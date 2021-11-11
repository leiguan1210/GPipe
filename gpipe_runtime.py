'''
Author: Lei Guan
Email: guanleics@gmail.com
Time: 20/03/2018
'''
import torch
import collections

IMAGE_CLASSIFICATION = "image_classification"
TRANSLATION = "translation"
SPEECH_TO_TEXT = "speech_to_text"


class ModulesWithDependencies:
    def __init__(self, modules_with_dependencies):
        self._modules = []
        self._all_input_names = []
        self._all_output_names = []
        for (module, input_names, output_names) in modules_with_dependencies:
            self._modules.append(module)
            self._all_input_names.append(input_names)
            self._all_output_names.append(output_names)

    def modules(self):
        return self._modules

    def all_input_names(self):
        return self._all_input_names

    def all_output_names(self):
        return self._all_output_names

    def is_input_tensor(self, tensor_name):
        for module_input_names in self._all_input_names:
            if tensor_name in module_input_names:
                return True
        return False


class StageRuntime(object):
    def __init__(self, model, rank, world_size, inputs_module_destinations,
                 configuration_maps, target_tensor_names, model_type):
        self.tensors = {}
        self.gradients = {}

        self.rank = rank
        self.world_size = world_size
        self.model_type = model_type
        self.target_tensor_names = target_tensor_names
        # inputs, outputs of stages
        self.input_names = {}
        self.output_names = {}
        # inputs, outputs of criterion
        self.criterion_input_names = {}
        self.criterion_output_names = {}
        # send/recv between stages
        self.send_names = {}
        self.recv_names = {}
        self.send_ranks = {}
        self.receive_ranks = {}

        self.fwd_idx = 0
        self.bwd_idx = 0

        self.optimizer = None

        self.is_criterion = self.rank == self.world_size - 1

        self.initialize(model, inputs_module_destinations, configuration_maps)

    def initialize(self, model, inputs_module_destinations, configuration_maps):
        modules_with_dependencies = ModulesWithDependencies(model)
        all_input_names = modules_with_dependencies.all_input_names()
        all_output_names = modules_with_dependencies.all_output_names()
        all_sub_modules = modules_with_dependencies.modules()

        self.input_names = all_input_names[self.rank]
        self.output_names = all_output_names[self.rank]
        self.sub_module = all_sub_modules[self.rank]
        self.sub_module = self.sub_module.cuda()

        self.criterion_input_names = all_input_names[-1]
        self.criterion_output_names = all_output_names[-1]
        self.criterion_module = all_sub_modules[-1]

        # print("rank", self.rank, self.input_names, self.output_names)

        module_to_stage_map = configuration_maps['module_to_stage_map']
        stage_to_rank_map = configuration_maps['stage_to_rank_map']
        stage_to_depth_map = configuration_maps['stage_to_depth_map']

        ### TODO send_names, recv_names
        stage_to_module_map = collections.defaultdict(list)
        for module in range(len(module_to_stage_map)):
            stage_to_module_map[module_to_stage_map[module]].append(module)

        rank_to_stage_map = {}
        for stage in stage_to_rank_map:
            for rank in stage_to_rank_map[stage]:
                rank_to_stage_map[rank] = stage

        self.stage = rank_to_stage_map[self.rank]
        modules = stage_to_module_map[self.stage]
        self.modules_with_dependencies = ModulesWithDependencies(
            [model[module] for module in modules])

        # print(len(model), module_to_stage_map)

        for i in range(len(model)):
            for j in range(i + 1, len(model)):
                for tensor_name in model[i][2]:
                    if tensor_name in model[j][1]:
                        if module_to_stage_map[i] == \
                                module_to_stage_map[j]:
                            continue
                        # For now, assume that each stage is served by only
                        # a single machine.
                        if module_to_stage_map[j] == self.stage:
                            self.receive_ranks[tensor_name] = \
                                stage_to_rank_map[module_to_stage_map[i]]
                        if module_to_stage_map[i] == self.stage:
                            self.send_ranks[tensor_name] = \
                                stage_to_rank_map[module_to_stage_map[j]]

        for model_inputs in inputs_module_destinations.keys():
            destination_stage = module_to_stage_map[
                inputs_module_destinations[model_inputs]]
            if destination_stage > self.stage:
                self.send_ranks[model_inputs] = \
                    self.ranks_in_next_stage

            if 0 < self.stage <= destination_stage:
                self.receive_ranks[model_inputs] = \
                    self.ranks_in_previous_stage

        print("rank", self.rank, self.input_names, self.output_names, self.send_ranks, self.receive_ranks)
        ## rank 0 ['input0'] ['out0'] {'out0': [1]} {}
        ## rank 1 ['out0'] ['out1'] {'out1': [2]} {'out0': [0]}
        ## rank 2 ['out1'] ['out2'] {'out2': [3]} {'out1': [1]}
        ## rank 3 ['out2'] ['out3'] {} {'out2': [2]}

    def set_loader(self, loader):
        if loader is not None:
            self.loader_iter = iter(loader)
        else:
            self.loader_iter = None

    def train(self):
        self.tensors = {}
        self.gradients = {}

        self.fwd_idx = 0
        self.bwd_idx = 0
        self.sub_module.train()

    def run_forward(self):
        # print("---1 before tensors", self.tensors.keys())
        self.receive_tensors_forward()
        # print("---2 before tensors", self.tensors.keys())

        self._run_forward(self.tensors)
        if self.rank == self.world_size - 1:
            self._run_criterion(self.tensors)
        # print("---3 before tensors", self.tensors.keys())

        self.update_fwd_idx()

    def _run_criterion(self, tensors):
        # If layer is criterion (loss).
        if self.model_type == SPEECH_TO_TEXT:
            output = tensors["output"].transpose(0, 1).float()
            output_sizes = tensors["output_sizes"].cpu()
            target = tensors["target"].cpu()
            target_sizes = tensors["target_length"].cpu()
            input0_size = tensors["input0_size"].cpu()
            module_outputs = [self.criterion_module(output, target, output_sizes, target_sizes) / input0_size[0]]
        else:
            # print("==== input_names", tensors, self.criterion_input_names)
            module_outputs = [self.criterion_module(tensors[input_name], tensors["target"])
                              for input_name in self.criterion_input_names]
            module_outputs = [sum(module_outputs)]

        for (output_name, module_output) in zip(self.criterion_output_names, module_outputs):
            tensors[output_name] = module_output

        self.output = tensors[self.criterion_input_names[0]]

        if self.is_criterion and self.model_type == TRANSLATION:
            loss_per_batch = tensors[self.criterion_input_names[0]] * tensors[self.criterion_input_name].size(1)
            loss_per_token = loss_per_batch / tensors["target_length"][0].item()
            self.loss = loss_per_token
        elif self.is_criterion:
            self.loss = tensors[self.criterion_output_names[0]]
        else:
            self.loss = 1

    def _run_forward(self, tensors):
        module_outputs = self.sub_module(*[tensors[input_name]
                                           for input_name in self.input_names])
        if not isinstance(module_outputs, tuple):
            module_outputs = (module_outputs,)

        module_outputs = list(module_outputs)

        # print("before tensors", tensors.keys())  before tensors dict_keys(['input0', 'target', 'out0'])
        for (output_name, module_output) in zip(self.output_names, module_outputs):
            tensors[output_name] = module_output
        # print("after tensors", tensors.keys())  after tensors dict_keys(['input0', 'target', 'out0'])

    def _run_backward(self, handler):
        self.receive_tensors_backward()

        self._do_backward(self.gradients)

        grad_tensors = {}
        for name in self.inputs:
            if self.inputs[name].requires_grad:
                grad_tensors[name] = self.inputs[name].grad.data.clone()

        self.send_tensors_backward(grad_tensors)
        self.update_bwd_idx()

    def run_backward(self, input_names, output_names):
        inputs = {}
        outputs = {}
        input_gradients = {}
        output_gradients = {}

        # get outputs
        for output_name in output_names:
            if output_name not in self.gradients:
                output_gradients[output_name] = None
            else:
                output_gradients[output_name] = self.gradients[output_name]
            if self.tensors[output_name].requires_grad:
                outputs[output_name] = self.tensors[output_name]

        # get inputs
        for input_name in input_names:
            if input_name not in output_names:
                inputs[input_name] = self.tensors[input_name]

        # print("outputs", outputs, output_gradients)

        # Hook to record input gradients
        def hook_wrapper(input_name):
            def hook(input_gradient):
                input_gradients[input_name] = input_gradient
            return hook

        for input_name in inputs:
            if input_name != "input0" and input_name != "input1" and input_name != "input2" \
                    and inputs[input_name].requires_grad:
                inputs[input_name].register_hook(hook_wrapper(input_name))

        # perform backward pass
        torch.autograd.backward(tuple([outputs[output_name] for output_name in outputs]),
                                grad_tensors=tuple([output_gradients[output_name]
                                                    for output_name in outputs]))

        # Input tensors don't need gradients.
        for input_name in inputs:
            if not inputs[input_name].requires_grad:
                self.gradients[input_name] = inputs[input_name]
                continue

            if input_name != "input0" and input_name != "input1" and input_name != "input2" and input_name != "input":
                self.gradients[input_name] = input_gradients[input_name]

    def receive_tensors_forward(self):
        if self.loader_iter is not None:
            input = next(self.loader_iter)
            if self.model_type == TRANSLATION:
                (input, target) = input
                src, src_length = input
                tgt, tgt_length = target

                self.tensors["input0"] = src.cuda(non_blocking=True)
                self.tensors["input1"] = torch.LongTensor(src_length).cuda(non_blocking=True)
                self.tensors["input2"] = tgt[:-1].cuda(non_blocking=True)
                self.tensors["target"] = tgt[1:].cuda().contiguous().view(-1)
                self.tensors["target_length"] = \
                    torch.tensor([int(sum(torch.LongTensor(tgt_length) - 1))],
                                 dtype=torch.int).cuda()
            elif self.model_type == IMAGE_CLASSIFICATION:
                (input, target) = input
                self.tensors["input0"] = input.cuda(non_blocking=True)
                self.tensors["target"] = target.cuda(non_blocking=True)
            elif self.model_type == SPEECH_TO_TEXT:
                input, target, input_percentages, target_sizes = input
                input_sizes = input_percentages.mul_(int(input.size(3))).int()
                self.tensors["input0"] = input.cuda(non_blocking=True)
                self.tensors["input1"] = input_sizes.cuda(non_blocking=True)
                self.tensors["target"] = target.cuda(non_blocking=True)
                self.tensors["target_length"] = target_sizes.cuda(
                    non_blocking=True)

    def update_fwd_idx(self):
        self.fwd_idx += 1

    def update_bwd_idx(self):
        self.bwd_idx += 1




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
    def __init__(self, model, rank, inputs_module_destinations,
                 configuration_maps, model_type):
        self.tensors = {}
        self.gradients = {}

        self.rank = rank
        self.model_type = model_type
        # inputs, outputs of stages
        self.input_names = {}
        self.output_names = {}
        # send/recv between stages
        self.send_names = {}
        self.recv_names = {}
        self.send_ranks = {}
        self.receive_ranks = {}

        self.initialize(model, inputs_module_destinations, configuration_maps)

    def initialize(self, model, inputs_module_destinations, configuration_maps):
        modules_with_dependencies = ModulesWithDependencies(model)
        all_input_names = modules_with_dependencies.all_input_names()
        all_output_names = modules_with_dependencies.all_output_names()
        all_sub_modules = modules_with_dependencies.modules()

        self.input_names = all_input_names[self.rank]
        self.output_names = all_output_names[self.rank]
        self.sub_module = all_sub_modules[self.rank]

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



    def do_forward(self):
        self.receive_tensors_forward()

        self._do_forward()

        self.send_tensors_forward(self.outputs.copy(), self.target_tensors)

        self.update_fwd_idx()

    def _do_forward(self, inputs):
        outputs = {}

        module_outputs = self.net(*[inputs[input_name] for input_name in self.input_names])
        if not isinstance(module_outputs, tuple):
            module_outputs = (module_outputs,)
        module_outputs = list(module_outputs)
        for (output_name, module_output) in zip(self.output_names, module_outputs):
            outputs[output_name] = module_output

        return outputs


    def do_backward(self, handler):
        self.receive_tensors_backward()

        self._do_backward(self.gradients)

        grad_tensors = {}
        for name in self.inputs:
            if self.inputs[name].requires_grad:
                grad_tensors[name] = self.inputs[name].grad.data.clone()

        self.send_tensors_backward(grad_tensors)
        self.update_bwd_idx()

    def _do_backward(self, outputs, grad_tensors):
        if self.last_stage:
            module_outputs = [self.criterion(
                self.outputs[name], self.target_tensors["target"])
                for name in self.output_names
            ]
            loss = sum(module_outputs)
            self.loss = loss
            loss.backward()

        else:
            torch.autograd.backward(
                tuple([self.outputs[name] for name in self.output_names]),
                grad_tensors=tuple([grad_tensors[name] for name in self.output_names])
            )

        grad_tensors = {}
        for name in self.inputs:
            if self.inputs[name].requires_grad:
                grad_tensors[name] = self.inputs[name].grad.data.clone()

        return grad_tensors

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
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
            elif self.model_type == SPEECH_TO_TEXT:
                input, target, input_percentages, target_sizes = input
                input_sizes = input_percentages.mul_(int(input.size(3))).int()
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = input_sizes.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
                self.tensors[-1]["target_length"] = target_sizes.cuda(
                    non_blocking=True)


    def update_fwd_idx(self):
        self.fwd_idx += 1

    def update_bwd_idx(self):
        self.bwd_idx += 1




import os
import torch
import numpy as np
from tqdm import tqdm

from model.SEWResNet import *
from model.LoRaFBSNet import *
from model.functional import *
from spikingjelly.activation_based import functional, neuron

from model.cornet import *
from model.monkeynet import *

from model.CORnet import *
from model.LoRaFBCNet import *


class Extraction:
    def __init__(self, model, model_name, stimulus_path, device="cuda:0"):
        self.model = model.to(device)
        self.stimulus_path = stimulus_path
        self.set_stimulus(stimulus_path)
        self.model_name = model_name
        self.device = device
    
        self.features = None
        self.batch_size = 1

    def load_model(self, model_name, checkpoint_path, _checkpoint_args, **kwargs):
        if _checkpoint_args:
            model = eval(model_name)(checkpoint_path=checkpoint_path, **kwargs)
        else:
            model = eval(model_name)(**kwargs)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])

        return model

    def set_stimulus(self, stimulus_path):
        self.stimulus_path = stimulus_path
        self.stimulus = torch.load(stimulus_path)
        self.stimulus_change = True

    def build_dataloader(self, batch_size):
        self.stimulus_dataset = torch.utils.data.TensorDataset(self.stimulus)
        self.n_stimulus = len(self.stimulus_dataset)
        self.stimulus_dataloader = torch.utils.data.DataLoader(self.stimulus_dataset, batch_size=batch_size)

    def shuffle_stimulus(self, window):
        stimulus = torch.zeros_like(self.stimulus)
        shuffle_index = np.zeros(stimulus.size(0), dtype=int)
        num_stimuli = stimulus.size(0)
        num_windows = int(np.ceil(num_stimuli / window))
        for i in range(num_windows):
            l = i * window
            r = min((i + 1) * window, num_stimuli)
            random_index = np.random.permutation(r - l)
            shuffle_index[l: r] = l + random_index
            stimulus[l: r] = self.stimulus[l + random_index]
        self.stimulus = stimulus
        self.stimulus_change = True
        return shuffle_index

    def replace_stimulus(self, replace_path, window):
        replace = torch.load(replace_path)
        stimulus = self.stimulus
        num_stimuli = stimulus.size(0)
        num_replace = int(np.ceil(num_stimuli / window))
        replace_index = np.random.randint(low=window, size=num_replace) + np.arange(0, num_replace * window, window)
        replace_index[-1] = min(replace_index[-1], num_stimuli - 1)
        stimulus[replace_index] = replace[replace_index]
        self.stimulus = stimulus
        self.stimulus_change = True
        return replace_index

    def hook_fn(self, module, inputs, outputs):
        pass

    def layer_extraction(self, layer_name, layer_dims):
        pass
    

class SNNStaticExtraction(Extraction):
    def __init__(self, model_name, checkpoint_path, stimulus_path, T, _3d=False, _checkpoint_args=False, device="cuda:0", **kwargs):
        model = self.load_model(model_name, checkpoint_path, _checkpoint_args, **kwargs)
        set_step_mode(model, 'm', (ConvRecurrentContainer, ))
        set_backend(model, 'cupy', neuron.BaseNode, (ConvRecurrentContainer, ))

        super().__init__(model, model_name, stimulus_path, device)
        self.T = T
        self._3d = _3d
        self._mean = True
    
    def hook_fn(self, module, inputs, outputs):
        self.features.append(outputs.data.cpu())

    def layer_extraction(self, layer_name, layer_dims):
        if self.stimulus_change:
            self.build_dataloader(self.batch_size)
            self.stimulus_change = False
        if self._mean:
            extraction = torch.zeros([self.n_stimulus] + layer_dims, dtype=torch.float)
        else:
            extraction = torch.zeros([self.n_stimulus] + [self.T] + layer_dims, dtype=torch.float)
        
        self.model.eval()
        functional.reset_net(self.model)
        with torch.inference_mode():
            hook = eval(f"self.model.{layer_name}").register_forward_hook(self.hook_fn)
            n = 0
            for inputs in tqdm(self.stimulus_dataloader):
                inputs = inputs[0].to(self.device)
                bs = len(inputs)
                inputs = inputs.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

                self.features = []
                self.model(inputs)
                features = torch.stack(self.features, dim=0)
                if features.size(0) == 1:
                    features = features.squeeze(0)
                if self._mean:
                    extraction[n: n + bs] = features.mean(dim=0)
                else:
                    extraction[n: n + bs] = features.transpose(0, 1)
                functional.reset_net(self.model)
                n += bs
            hook.remove()

        return extraction


class SNNMovieExtraction(Extraction):
    def __init__(self, model_name, checkpoint_path, stimulus_path, T, _3d=False, _checkpoint_args=False, device="cuda:0", **kwargs):
        model = self.load_model(model_name, checkpoint_path, _checkpoint_args, **kwargs)
        set_step_mode(model, 'm', (ConvRecurrentContainer, ))
        set_backend(model, 'cupy', neuron.BaseNode, (ConvRecurrentContainer, ))

        super().__init__(model, model_name, stimulus_path, device)
        self.T = T
        self._3d = _3d

    def hook_fn(self, module, inputs, outputs):
        self.features.append(outputs.data.cpu())

    def layer_extraction(self, layer_name, layer_dims):
        if self.stimulus_change:
            self.build_dataloader(self.batch_size * self.T)
            self.stimulus_change = False
        extraction = torch.zeros([self.n_stimulus] + layer_dims, dtype=torch.float)

        self.model.eval()
        functional.reset_net(self.model)
        with torch.inference_mode():
            hook = eval(f"self.model.{layer_name}").register_forward_hook(self.hook_fn)
            n = 0
            for inputs in tqdm(self.stimulus_dataloader):
                inputs = inputs[0].to(self.device)
                bs = len(inputs)
                ## Default of batch size is 1
                inputs = inputs.unsqueeze(1)

                self.features = []
                self.model(inputs)
                features = torch.stack(self.features, dim=0)
                if features.size(0) == 1:
                    features = features.squeeze(0)
                extraction[n: n + bs] = features.squeeze(1)
                n += bs
            hook.remove()

        return extraction


class CNNStaticExtraction(Extraction):
    def __init__(self, model_name, checkpoint_path, stimulus_path, T=1, _3d=False, _checkpoint_args=False, device="cuda:0", **kwargs):
        model = self.load_model(model_name, checkpoint_path, _checkpoint_args, **kwargs)
        super().__init__(model, model_name, stimulus_path, device)
        self.T = T
        self._3d = _3d
    
    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            self.features.append(outputs[0].data.cpu())
        else:
            self.features.append(outputs.data.cpu())

    def layer_extraction(self, layer_name, layer_dims):
        if self.stimulus_change:
            self.build_dataloader(self.batch_size)
            self.stimulus_change = False
        extraction = torch.zeros([self.n_stimulus] + layer_dims, dtype=torch.float)

        self.model.eval()
        if self.T > 1 and not self._3d:
            self.model.reset()
        with torch.inference_mode():
            hook = eval(f"self.model.{layer_name}").register_forward_hook(self.hook_fn)
            n = 0
            for inputs in tqdm(self.stimulus_dataloader):
                inputs = inputs[0].to(self.device)
                bs = len(inputs)
                if self.T > 1:
                    if self._3d:
                        inputs = inputs.unsqueeze(2).repeat(1, 1, self.T, 1, 1)
                    else:
                        inputs = inputs.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

                self.features = []
                self.model(inputs)
                features = torch.stack(self.features, dim=0)
                if self.T > 1:
                    if self._3d:
                        extraction[n: n + bs] = features.squeeze(0).mean(dim=2)
                    else:
                        if features.size(0) == 1:
                            features = features.squeeze(0)
                            extraction[n: n + bs] = features.view(bs, self.T, features.size(1), features.size(2), features.size(3)).mean(dim=1)
                        else:
                            extraction[n: n + bs] = features.mean(dim=0)
                        self.model.reset()
                else:
                    extraction[n: n + bs] = features[-1]
                n += bs
            hook.remove()

        return extraction


class CNNMovieExtraction(Extraction):
    def __init__(self, model_name, checkpoint_path, stimulus_path, T=1, _3d=False, _checkpoint_args=False, device="cuda:0", **kwargs):
        model = self.load_model(model_name, checkpoint_path, _checkpoint_args, **kwargs)
        super().__init__(model, model_name, stimulus_path, device)
        self.T = T
        self._3d = _3d
    
    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            self.features.append(outputs[0].data.cpu())
        else:
            self.features.append(outputs.data.cpu())

    def layer_extraction(self, layer_name, layer_dims):
        if self.stimulus_change:
            self.build_dataloader(self.batch_size * self.T)
            self.stimulus_change = False
        extraction = torch.zeros([self.n_stimulus] + layer_dims, dtype=torch.float)

        self.model.eval()
        if not self._3d:
            self.model.reset()
        with torch.inference_mode():
            hook = eval(f"self.model.{layer_name}").register_forward_hook(self.hook_fn)
            n = 0
            for inputs in tqdm(self.stimulus_dataloader):
                inputs = inputs[0].to(self.device)
                bs = len(inputs)
                ## Default of batch size is 1
                if self._3d:
                    inputs = inputs.unsqueeze(0).transpose(1, 2)
                else:
                    inputs = inputs.unsqueeze(1)

                self.features = []
                self.model(inputs)
                features = torch.stack(self.features, dim=0)
                if features.size(0) == 1:
                    features = features.squeeze(0)
                if self._3d:
                    extraction[n: n + bs] = features.squeeze(0).transpose(0, 1)
                else:
                    extraction[n: n + bs] = features.squeeze(1)
                n += bs
            hook.remove()

        return extraction

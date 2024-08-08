import os
import torch
import numpy as np


class VisualModel:
    def __init__(self, model_name, layers_info, extraction, shuffle=False, replace=False, window=0, noise_stimulus_path=None, front_len=0, _normalize=False):
        self.model_name = model_name
        self.layers_info = layers_info
        self.extraction = extraction
        self.stimulus_path = extraction.stimulus_path

        self.shuffle = shuffle
        self.replace = replace
        self.window = window
        self.noise_stimulus_path = noise_stimulus_path
        self.front_len = front_len
        if self.shuffle or self.replace:
            assert self.window > 0
        if self.replace:
            assert self.noise_stimulus_path is not None
        if self.front_len > 0:
            self.extraction.set_stimulus(self.stimulus_path)
            self.extraction.front_stimulus(self.front_len)

        self._normalize = _normalize
    
    def _z_score(self, x):
        _mean = np.mean(x)
        _std = np.std(x)
        x = (x - _mean) / (_std + 1e-10)
        
        return x

    def _process_model_data(self, x):
        x = x.numpy()
        x = x.reshape((x.shape[0], -1))
        if self._normalize:
            x = self._z_score(x)
        return x

    def __len__(self):
        return len(self.layers_info)
    
    def __getitem__(self, key):
        if self.shuffle:
            self.extraction.set_stimulus(self.stimulus_path)
            shuffle_index = self.extraction.shuffle_stimulus(self.window)
        if self.replace:
            self.extraction.set_stimulus(self.stimulus_path)
            replace_index = self.extraction.replace_stimulus(self.noise_stimulus_path, self.window)
        model_data = self.extraction.layer_extraction(self.layers_info[key][0], self.layers_info[key][1])
        model_data = self._process_model_data(model_data)
        if self.shuffle:
            return model_data, shuffle_index
        elif self.replace:
            return model_data, replace_index
        else:
            return model_data

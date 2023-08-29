import os
import numpy as np


class NeuralDataset:
    def __init__(self, dataset_name, brain_areas, data_dir, **kwargs):
        self.dataset_name = dataset_name
        self.brain_areas = brain_areas
        self.data_dir = data_dir
        
        self.neural_dataset = {}
        for i in range(len(self.brain_areas)):
            self.neural_dataset[self.brain_areas[i]] = eval(f"self.{dataset_name}")(self.brain_areas[i], **kwargs)

    def allen_natural_movie_one(self, brain_area, exclude, threshold, time_step=1, _mean_time_step=True):
        neural_data = np.load(os.path.join(self.data_dir, self.dataset_name, f"{brain_area}_{time_step}.npy"))

        neural_firing_rate = np.sum(np.sum(neural_data, axis=1), axis=0) / 30 / 20
        if exclude:
            neural_data = neural_data[:, :, neural_firing_rate >= threshold]

        if _mean_time_step:
            neural_data = np.sum(neural_data, axis=1)
        neural_data = neural_data / 20

        return neural_data
    
    def allen_natural_movie_three(self, brain_area, exclude, threshold, time_step=1, _mean_time_step=True):
        neural_data = np.load(os.path.join(self.data_dir, self.dataset_name, f"{brain_area}_{time_step}.npy"))

        neural_firing_rate = np.sum(np.sum(neural_data, axis=1), axis=0) / 120 / 10
        if exclude:
            neural_data = neural_data[:, :, neural_firing_rate >= threshold]

        if _mean_time_step:
            neural_data = np.sum(neural_data, axis=1)
        neural_data = neural_data / 10

        return neural_data

    def __len__(self):
        return len(self.brain_areas)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.neural_dataset[self.brain_areas[key]]
        elif isinstance(key, str):
            return self.neural_dataset[key]
        else:
            raise KeyError(f"Unknown key: {key}")
    
    def keys(self):
        return self.brain_areas

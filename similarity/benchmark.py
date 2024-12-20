import os
import time
from tqdm import tqdm
import torch
import numpy as np
from extraction import SNNStaticExtraction, CNNStaticExtraction


class Benchmark:
    def __init__(self, neural_dataset, metric, save_dir=None, suffix=""):
        self.neural_dataset = neural_dataset
        self.metric = metric
        self.save_dir = save_dir
        self.suffix = suffix

        self.brain_areas = self.neural_dataset.keys()
        self.num_areas = len(self.neural_dataset)

        self.model_name = None
        self.num_layers = 0
    
    def _preset_print(self):
        self.area_length = 0
        for brain_area in self.brain_areas:
            if len(brain_area) > self.area_length: self.area_length = len(brain_area)
        
        self.layer_length = 1
        num_layers = self.num_layers
        while num_layers // 10 > 0:
            self.layer_length += 1
            num_layers //= 10
        self.layer_length += len(self.model_name) + 1

    def _print_score(self, brain_area, model_layer, score, _time):
        print(f"%-{self.area_length}s %-{self.layer_length}s: %-9.6f  time: %.4fs" % (brain_area, model_layer, score, _time))

    def _print_one_layer_scores(self, scores):
        print("All scores:")

        for brain_area in self.brain_areas:
            print(f"%-11s" % (brain_area), end='')
        print()

        for i in range(scores.shape[0]):
            print(f"%-11.8f" % (scores[i]), end='')
        print()

    def _print_all_scores(self, scores):
        print("All scores:")

        print(f"%{self.layer_length}s" % (' '), end='')
        for brain_area in self.brain_areas:
            print(f" %-11s" % (brain_area), end='')
        print()

        for i in range(scores.shape[0]):
            print(f"%-{self.layer_length}s" % (f"{self.model_name}_{i + 1}"), end='')
            for j in range(scores.shape[1]):
                print(f" %-11.8f" % (scores[i, j]), end='')
            print()

    def _save_scores(self, scores):
        os.makedirs(self.save_dir, exist_ok=True)
        np.save(os.path.join(self.save_dir, f"{self.model_name}{self.suffix}.npy"), scores)

    def __call__(self, visual_model):
        pass


class MovieBenchmark(Benchmark):
    seed = 2023

    def __init__(self, neural_dataset, metric, save_dir=None, suffix="", trial_for_ablation=1, shuffle=False, replace=False, best_layer=False, trial_for_clip=1, clip_len=0, front_len=0):
        super().__init__(neural_dataset, metric, save_dir, suffix)
        self.front_len = front_len
        assert self.front_len >= 0 and self.front_len <= len(self.neural_dataset[0])
        if self.front_len == 0:
            self.num_stimuli = len(self.neural_dataset[0])
        else:
            self.num_stimuli = self.front_len
        self.no_first_frame_idx = np.ones(self.num_stimuli, dtype=bool)
        if neural_dataset.dataset_name != "allen_natural_scenes":
            self.no_first_frame_idx[0] = False
            self.num_stimuli -= 1

        self.trial_for_ablation = trial_for_ablation
        self.shuffle = shuffle
        self.replace = replace
        self.ablation = self.shuffle or self.replace
        self.best_layer = best_layer
        assert (self.best_layer and self.ablation) or (not self.best_layer)

        self.trial_for_clip = trial_for_clip
        self.clip_len = clip_len
        assert 0 <= self.clip_len <= self.num_stimuli - self.trial_for_clip + 1
    
    def __call__(self, visual_model):
        self.model_name = visual_model.model_name
        self.num_layers = len(visual_model)

        self._preset_print()
        if self.best_layer:
            scores = np.zeros((self.trial_for_ablation, self.num_areas))
            split_save_dir = self.save_dir.split('/')
            original_save_dir = os.path.join(*split_save_dir[0:4])
            assert os.path.isfile(os.path.join(original_save_dir, f"{self.model_name}.npy"))
            original_scores = np.load(os.path.join(original_save_dir, f"{self.model_name}.npy"))
            layers_index_for_area = np.argmax(original_scores, axis=0)
            layers_index = set(layers_index_for_area)
        else:
            scores = np.zeros((self.trial_for_ablation, self.trial_for_clip, self.num_layers, self.num_areas))
            layers_index = np.arange(self.num_layers)
        
        for layer_index in layers_index:
            np.random.seed(self.seed)
            if self.clip_len > 0:
                start_index = np.random.choice(self.num_stimuli - self.clip_len + 1, size=self.trial_for_clip, replace=False)
                end_index = start_index + self.clip_len
            else:
                start_index = np.zeros(self.trial_for_clip, dtype=int)
                end_index = np.ones(self.trial_for_clip, dtype=int) * self.num_stimuli
            for i in range(self.trial_for_ablation):
                print()
                if not self.ablation:
                    print(f"Extracting features of layer {layer_index + 1} of {self.model_name}")

                    model_data = visual_model[layer_index]
                    model_data = model_data[self.no_first_frame_idx]
                else:
                    print(f"Trial {i + 1}: Extracting features of layer {layer_index + 1} of {self.model_name}")
                
                    if self.shuffle:
                        model_data, shuffle_index = visual_model[layer_index]
                        if isinstance(visual_model.extraction, (SNNStaticExtraction, CNNStaticExtraction)):
                            no_first_frame_idx = self.no_first_frame_idx[shuffle_index]
                        else:
                            no_first_frame_idx = self.no_first_frame_idx
                        model_data = model_data[no_first_frame_idx]
                    if self.replace:
                        model_data, replace_index = visual_model[layer_index]
                        model_data = model_data[self.no_first_frame_idx]
                
                for area_index, brain_area in enumerate(self.brain_areas):
                    if self.best_layer and layers_index_for_area[area_index] != layer_index:
                        continue
                    start_time = time.time()
                    
                    neural_data = self.neural_dataset[area_index]
                    if not self.ablation:
                        if self.front_len > 0:
                            neural_data = neural_data[:self.front_len]
                        neural_data = neural_data[self.no_first_frame_idx]
                    else:
                        if self.shuffle:
                            neural_data = neural_data[shuffle_index]
                            neural_data = neural_data[no_first_frame_idx]
                        if self.replace:
                            neural_data = neural_data[self.no_first_frame_idx]

                    for j in range(self.trial_for_clip):
                        score = self.metric.score(model_data[start_index[j]: end_index[j]], neural_data[start_index[j]: end_index[j]])
                        self._print_score(brain_area, f"{self.model_name}_{layer_index + 1}", score, time.time() - start_time)

                        if self.best_layer:
                            scores[i, area_index] = score
                        else:
                            scores[i, j, layer_index, area_index] = score
        print()
        if self.best_layer:
            self._print_one_layer_scores(np.mean(scores, axis=0))
        else:
            self._print_all_scores(np.mean(np.mean(scores, axis=0), axis=0))
        
        if self.save_dir is not None:
            if not self.ablation:
                scores = np.mean(scores, axis=0)
            if self.clip_len == 0:
                if not self.ablation:
                    scores = np.mean(scores, axis=0)
                else:
                    if not self.best_layer:
                        scores = np.mean(scores, axis=1)
            self._save_scores(scores)

import os
import json
import time
import pickle
import torch
import numpy as np
from tqdm import tqdm

from dataset import NeuralDataset
from metric import TSRSAMetric, RegMetric


def main(dataset_name, metric_name):
    brain_areas = ['visp', 'visl', 'visrl', 'visal', 'vispm', 'visam']
    time_step = 1
    exclude = True
    threshold = 0.5

    neural_dataset = NeuralDataset(
        dataset_name=dataset_name,
        brain_areas=brain_areas,
        data_dir="/data/hlw20/neural-computational-model-source/neural_dataset",
        time_step=time_step,
        exclude=exclude,
        threshold=threshold
    )
    
    if metric_name == "TSRSA":
        metric = TSRSAMetric()
    elif metric_name == "Regression":
        metric = RegMetric()
    save_dir = os.path.join("results", metric_name, dataset_name)
    
    scores = np.zeros((100, len(brain_areas)))
    for area_index, brain_area in enumerate(brain_areas):
        neural_data = neural_dataset[area_index]
        num_neurons = neural_data.shape[1]
        for i in tqdm(range(100)):
            random_index = np.random.permutation(num_neurons)
            neural_data_1 = neural_data[:, random_index[:num_neurons // 2]][1:]
            neural_data_2 = neural_data[:, random_index[num_neurons // 2:]][1:]
            score = metric.score(neural_data_1, neural_data_2)
            scores[i, area_index] = score
        print(np.mean(scores[:, area_index]), np.std(scores[:, area_index]) / np.sqrt(100))
    print(np.mean(scores), np.mean(np.std(scores, axis=0) / np.sqrt(100)))
    np.save(os.path.join(save_dir, "ceiling_half_neuron.npy"), scores)


    ### for split-half-trial experiments, doesn't apply
    # with open(os.path.join("/data/hlw20/neural-computational-model-source/neural_dataset", dataset_name, "1.pkl"), 'rb') as f:
    #     specimens_spikes_data = pickle.load(f)
    # if dataset_name == "allen_natural_movie_one":
    #     num_stimulus = 900
    # elif dataset_name == "allen_natural_movie_three":
    #     num_stimulus = 3600
    # for area_index, area in enumerate(brain_areas):
    #     num_units = 0
    #     min_num_trial = 100
    #     for specimen_spikes_data in specimens_spikes_data:
    #         if specimen_spikes_data[area_index] is not None:
    #             num_units += specimen_spikes_data[area_index][0].shape[2]
    #             for j, stimulus_spikes_data in enumerate(specimen_spikes_data[area_index]):
    #                 if min_num_trial > stimulus_spikes_data.shape[0]:
    #                     min_num_trial = stimulus_spikes_data.shape[0]

    #     for i in tqdm(range(100)):
    #         neural_data_1 = np.zeros((num_stimulus, num_units))
    #         neural_data_2 = np.zeros((num_stimulus, num_units))
    #         random_index = np.random.permutation(min_num_trial)
    #         num_units = 0
    #         for specimen_spikes_data in specimens_spikes_data:
    #             area_spikes_data = specimen_spikes_data[area_index]
    #             if area_spikes_data is not None:
    #                 k = area_spikes_data[0].shape[2]
    #                 for j, stimulus_spikes_data in enumerate(area_spikes_data):
    #                     neural_data_1[j, num_units: num_units + k] += np.sum(np.mean(stimulus_spikes_data[random_index[:min_num_trial // 2]], axis=0), axis=0)
    #                     neural_data_2[j, num_units: num_units + k] += np.sum(np.mean(stimulus_spikes_data[random_index[min_num_trial // 2:]], axis=0), axis=0)
    #                 num_units += k
    #         neural_data_1 = neural_data_1[1:]
    #         neural_data_2 = neural_data_2[1:]
    #         score = metric.score(neural_data_1, neural_data_2)
    #         scores[i, area_index] = score
    #     print(np.mean(scores[:, area_index]), np.std(scores[:, area_index]) / np.sqrt(100))
    # print(np.mean(scores), np.mean(np.std(scores, axis=0) / np.sqrt(100)))
    # np.save(os.path.join(save_dir, "ceiling_half_trial.npy"), scores)


if __name__=="__main__":
    np.random.seed(2023)
    main("allen_natural_movie_one", "TSRSA")

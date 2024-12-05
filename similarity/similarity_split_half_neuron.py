import os
import json
import time
import pickle
import torch
import numpy as np
from tqdm import tqdm

from dataset import NeuralDataset
from metric import TSRSAMetric, RegMetric


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Neural Representation Similarity for Movie Stimuli")

    parser.add_argument("--neural-dataset", default="allen_natural_movie_one", type=str, choices=["allen_natural_movie_one", "allen_natural_movie_three"], help="name of neural dataset")
    parser.add_argument("--neural-dataset-dir", default="neural_dataset/", type=str, help="directory for storing neural dataset")

    parser.add_argument("--metric", default="TSRSA", type=str, choices=["TSRSA", "Regression"], help="name of similarity metric")

    parser.add_argument("--trial-for-clip", default=1, type=int, help="number of repetitions for experiments of different movie clip lengths")
    parser.add_argument("--clip-len", default=0, type=int, help="length of movie clip")

    parser.add_argument("--output-dir", default="results/", help="directory to save results of representational similarity")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    np.random.seed(2023)

    brain_areas = ['visp', 'visl', 'visrl', 'visal', 'vispm', 'visam']
    time_step = 1
    exclude = True
    threshold = 0.5

    neural_dataset = NeuralDataset(
        dataset_name=args.neural_dataset,
        brain_areas=brain_areas,
        data_dir=args.neural_dataset_dir,
        time_step=time_step,
        exclude=exclude,
        threshold=threshold
    )
    
    if args.metric == "TSRSA":
        metric = TSRSAMetric()
    elif args.metric == "Regression":
        metric = RegMetric()
    save_dir = os.path.join("results", args.metric, args.neural_dataset, "neural")
    os.makedirs(save_dir, exist_ok=True)
    
    if args.clip_len == 0:
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
        # with open(os.path.join(args.neural_dataset_dir, args.neural_dataset, "1.pkl"), 'rb') as f:
        #     specimens_spikes_data = pickle.load(f)
        # if args.neural_dataset == "allen_natural_movie_one":
        #     num_stimulus = 900
        # elif args.neural_dataset == "allen_natural_movie_three":
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
    else:
        scores = np.zeros((args.trial_for_clip, 100, len(brain_areas)))
        for area_index, brain_area in enumerate(brain_areas):
            neural_data = neural_dataset[area_index]
            neural_data = neural_data[1:]
            num_stimuli = neural_data.shape[0]
            num_neurons = neural_data.shape[1]
            np.random.seed(2023)
            start_index = np.random.choice(num_stimuli - args.clip_len + 1, size=args.trial_for_clip, replace=False)
            end_index = start_index + args.clip_len
            for i in tqdm(range(args.trial_for_clip)):
                for j in range(100):
                    random_index = np.random.permutation(num_neurons)
                    neural_data_1 = neural_data[start_index[i]: end_index[i], random_index[:num_neurons // 2]]
                    neural_data_2 = neural_data[start_index[i]: end_index[i], random_index[num_neurons // 2:]]
                    score = metric.score(neural_data_1, neural_data_2)
                    scores[i, j, area_index] = score
            print(np.mean(scores[:, :, area_index]), np.std(scores[:, :, area_index]) / np.sqrt(1000))
        print(np.mean(scores), np.mean(np.std(np.reshape(scores, (-1, 6)), axis=0) / np.sqrt(1000)))
        np.save(os.path.join(save_dir, f"stimulus_clip_ceiling_half_neuron_{args.clip_len}.npy"), scores)


if __name__=="__main__":
    main()

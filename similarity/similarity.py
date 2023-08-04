import os
import json
import time
import torch
import numpy as np

from dataset import NeuralDataset
from visualmodel import VisualModel
from metric import TSRSAMetric
from benchmark import MovieBenchmark
from extraction import SNNStaticExtraction, SNNMovieExtraction


def preset_neural_dataset(args):
    exclude = True
    threshold = 0.5
    neural_dataset = NeuralDataset(
        dataset_name="allen_natural_movie_one",
        brain_areas=['visp', 'visl', 'visrl', 'visal', 'vispm', 'visam'],
        data_dir=args.neural_dataset_dir,
        exclude=exclude,
        threshold=threshold
    )
    return neural_dataset, exclude, threshold


def build_extraction(args):
    stimulus_name = "stimulus_allen_natural_movie_one_224.pt"
    if args.train_dataset == "imagenet":
        T = 4
        extraction = SNNStaticExtraction(
            model_name=args.model,
            checkpoint_path=args.checkpoint_path,
            stimulus_path=os.path.join(args.stimulus_dir, stimulus_name),
            T=T,
            device=args.device
        )
    elif args.train_dataset == "ucf101":
        T = 16
        extraction = SNNMovieExtraction(
            model_name=args.model,
            checkpoint_path=args.checkpoint_path,
            stimulus_path=os.path.join(args.stimulus_dir, stimulus_name),
            T=T,
            device=args.device
        )
    return extraction, T


def save_dir_preset(args):
    save_dir = os.path.join(args.output_dir, args.train_dataset)
    if args.shuffle:
        save_dir = os.path.join(save_dir, "stimulus_shuffle")
    if args.replace:
        save_dir = os.path.join(save_dir, "stimulus_replace")
    
    suffix = ""
    if args.shuffle or args.replace:
        suffix = suffix + f"_{args.window}"

    return save_dir, suffix


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Neural Representation Similarity")

    parser.add_argument("--model", default="r_sew_resnet18", type=str, help="name of model")
    parser.add_argument("--train-dataset", default="ucf101", type=str, choices=["ucf101", "imagenet"], help="name of pretrain dataset")
    parser.add_argument("--checkpoint-path", default="model_checkpoint/ucf101/r_sew_resnet18.pth", type=str, help="path of pretrained model checkpoint")

    parser.add_argument("--neural-dataset-dir", default="neural_dataset/", type=str, help="directory for storing neural dataset")

    parser.add_argument("--stimulus-dir", default="stimulus/", type=str, help="directory for stimulus")
    parser.add_argument("--device", default="cuda:0", type=str, help="device for extracting features")

    parser.add_argument("--trial", default=1, type=int, help="number of repetitions for the shuffled frame experiment or the noise image replacement experiment")
    parser.add_argument("--shuffle", action="store_true", help="experiment for shuffled frame")
    parser.add_argument("--replace", action="store_true", help="experiment for noise image replacement")
    parser.add_argument("--window", default=0, type=int, help="number of frames per window for the shuffled frame experiment or the noise image replacement experiment")

    parser.add_argument("--output-dir", default="results/", help="directory to save results of representational similarity")

    args = parser.parse_args()
    return args


def main(args):
    extraction, args.T = build_extraction(args)
    with open(f"model_layers/{args.model}.json", 'r') as f:
        layers_info = json.load(f)
    layers_info = layers_info[:-1]
    if args.replace:
        noise_stimulus_path = os.path.join(args.stimulus_dir, "stimulus_allen_natural_movie_one_random_224.pt")
    else:
        noise_stimulus_path = None
    visual_model = VisualModel(
        model_name=args.model,
        layers_info=layers_info,
        extraction=extraction,
        shuffle=args.shuffle,
        replace=args.replace,
        window=args.window,
        noise_stimulus_path=noise_stimulus_path
    )

    neural_dataset, args.exclude, args.threshold = preset_neural_dataset(args)
    metric = TSRSAMetric()
    save_dir, suffix = save_dir_preset(args)
    benchmark = MovieBenchmark(
        neural_dataset=neural_dataset,
        metric=metric,
        save_dir=save_dir,
        suffix=suffix,
        trial=args.trial,
        shuffle=args.shuffle,
        replace=args.replace
    )
    print(args)
    benchmark(visual_model)


if __name__=="__main__":
    args = get_args()
    assert (not args.shuffle) or (not args.replace)
    main(args)

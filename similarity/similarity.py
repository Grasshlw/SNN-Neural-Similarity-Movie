import os
import json
import time
import torch
import numpy as np

from dataset import NeuralDataset
from visualmodel import VisualModel
from metric import TSRSAMetric
from benchmark import MovieBenchmark
from extraction import SNNStaticExtraction, SNNMovieExtraction, CNNStaticExtraction, CNNMovieExtraction


def preset_neural_dataset(args):
    time_step = 1
    exclude = True
    threshold = 0.5
    if args.neural_dataset == "allen_natural_scenes":
        time_step = 8
        threshold = 0.8
    
    neural_dataset = NeuralDataset(
        dataset_name=args.neural_dataset,
        brain_areas=['visp', 'visl', 'visrl', 'visal', 'vispm', 'visam'],
        data_dir=args.neural_dataset_dir,
        time_step=time_step,
        exclude=exclude,
        threshold=threshold
    )
    return neural_dataset, exclude, threshold


def build_extraction(args):
    stimulus_name = f"stimulus_{args.neural_dataset}_224.pt"
    model_args = {}
    _3d = False
    _checkpoint_args = False
    if args.train_dataset == "imagenet":
        if args.model in ['cornet_z', 'cornet_rt', 'cornet_s']:
            T = 1
            _checkpoint_args = True
            extraction_tool = CNNStaticExtraction
        else:
            T = 4
            extraction_tool = SNNStaticExtraction
            model_args['cnf'] = "ADD"
            model_args['num_classes'] = 1000
    elif args.train_dataset == "ucf101":
        if args.model in ['resnet_1p_ar', 'resnet_2p_ar', 'resnet_1p_cpc', 'resnet_2p_cpc', 'r_cornet_rt', 'r_resnet18']:
            if args.neural_dataset == "allen_natural_scenes":
                T = 4
                extraction_tool = CNNStaticExtraction
            else:
                T = 16
                extraction_tool = CNNMovieExtraction
            if args.model in ['resnet_1p_ar', 'resnet_2p_ar', 'resnet_1p_cpc', 'resnet_2p_cpc']:
                T = 5
                _3d = True
                _checkpoint_args = True
        else:
            if args.neural_dataset == "allen_natural_scenes":
                T = 4
                extraction_tool = SNNStaticExtraction
            else:
                T = 16
                extraction_tool = SNNMovieExtraction
            model_args['cnf'] = "ADD"
            model_args['num_classes'] = 101
    extraction = extraction_tool(
        model_name=args.model,
        checkpoint_path=args.checkpoint_path,
        stimulus_path=os.path.join(args.stimulus_dir, stimulus_name),
        T=T,
        _3d=_3d,
        _checkpoint_args=_checkpoint_args,
        device=args.device,
        **model_args
    )
    return extraction, T


def preset_metric(args):
    if args.metric == "TSRSA":
        metric = TSRSAMetric()
    return metric


def save_dir_preset(args):
    save_dir = os.path.join(args.output_dir, args.metric, args.neural_dataset, args.train_dataset)
    if args.shuffle:
        save_dir = os.path.join(save_dir, "stimulus_shuffle")
    if args.replace:
        save_dir = os.path.join(save_dir, f"stimulus_replace_{args.replace_type}")
    if args.shuffle or args.replace:
        if args.best_layer:
            save_dir = os.path.join(save_dir, "best_layer")
    
    suffix = ""
    if args.shuffle or args.replace:
        suffix = suffix + f"_{args.window}"

    return save_dir, suffix


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Neural Representation Similarity")

    parser.add_argument("--model", default="r_sew_resnet18", type=str, help="name of model for load")
    parser.add_argument("--model-name", default=None, type=str, help="name of model for save")
    parser.add_argument("--train-dataset", default="ucf101", type=str, choices=["ucf101", "imagenet"], help="name of pretrain dataset")
    parser.add_argument("--checkpoint-path", default="model_checkpoint/ucf101/r_sew_resnet18.pth", type=str, help="path of pretrained model checkpoint")

    parser.add_argument("--neural-dataset", default="allen_natural_movie_one", type=str, choices=["allen_natural_movie_one", "allen_natural_movie_three", "allen_natural_scenes"], help="name of neural dataset")
    parser.add_argument("--neural-dataset-dir", default="neural_dataset/", type=str, help="directory for storing neural dataset")

    parser.add_argument("--metric", default="TSRSA", type=str, choices=["TSRSA"], help="name of similarity metric")

    parser.add_argument("--stimulus-dir", default="stimulus/", type=str, help="directory for stimulus")
    parser.add_argument("--device", default="cuda:0", type=str, help="device for extracting features")

    parser.add_argument("--trial", default=1, type=int, help="number of repetitions for the shuffled frame experiment or the noise image replacement experiment")
    parser.add_argument("--shuffle", action="store_true", help="experiment for shuffled frame")
    parser.add_argument("--replace", action="store_true", help="experiment for noise image replacement")
    parser.add_argument("--replace-type", default="gaussian", type=str, choices=["gaussian", "uniform", "black", "static"], help="type of noise image for replacement")
    parser.add_argument("--window", default=0, type=int, help="number of frames per window for the shuffled frame experiment or the noise image replacement experiment")
    parser.add_argument("--best-layer", action="store_true", help="only conduct experiment for the best layer")

    parser.add_argument("--output-dir", default="results/", help="directory to save results of representational similarity")

    args = parser.parse_args()
    return args


def main(args):
    extraction, args.T = build_extraction(args)
    with open(f"model_layers/{args.model}.json", 'r') as f:
        layers_info = json.load(f)
    layers_info = layers_info[:-1]
    if args.replace:
        noise_stimulus_path = os.path.join(args.stimulus_dir, f"stimulus_{args.neural_dataset}_224_{args.replace_type}.pt")
    else:
        noise_stimulus_path = None
    visual_model = VisualModel(
        model_name=args.model_name,
        layers_info=layers_info,
        extraction=extraction,
        shuffle=args.shuffle,
        replace=args.replace,
        window=args.window,
        noise_stimulus_path=noise_stimulus_path
    )

    neural_dataset, args.exclude, args.threshold = preset_neural_dataset(args)
    metric = preset_metric(args)
    save_dir, suffix = save_dir_preset(args)
    benchmark = MovieBenchmark(
        neural_dataset=neural_dataset,
        metric=metric,
        save_dir=save_dir,
        suffix=suffix,
        trial=args.trial,
        shuffle=args.shuffle,
        replace=args.replace,
        best_layer=args.best_layer
    )
    print(args)
    benchmark(visual_model)


if __name__=="__main__":
    args = get_args()
    assert (not args.shuffle) or (not args.replace)
    if args.model_name is None:
        args.model_name = args.model
    main(args)

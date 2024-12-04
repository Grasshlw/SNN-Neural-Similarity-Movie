import os
import random
import warnings
import numpy as np
from tqdm import tqdm
import time
import datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.samplers import DistributedSampler, RandomClipSampler, UniformClipSampler
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import functional, neuron
import utils
import datasets
from transforms import ConvertBHWCtoBCHW
from model.LoRaFBSNet import *
from model.SEWResNet import *
from model.CORnet import *
from model.LoRaFBCNet import *
from model.SCORnet import *
from model.functional import set_step_mode, set_backend


_seed_ = 2022


def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="UCF101", type=str, choices=["UCF101", "UCF101_subset", "UCF101_random"], help="name of dataset")
    parser.add_argument("--data-path", default="/datasets/UCF101/UCF-101/", help="dataset path")
    parser.add_argument("--annotation-path", default="/datasets/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/", help="annotation path")
    parser.add_argument("--frames", default=16, type=int, help="number of frames per clip")
    parser.add_argument("--f-steps", default=16, type=int, help="number of frames between each clip")
    parser.add_argument("--cache-dataset", action="store_true", help="Cache the datasets for quicker initialization. It also serializes the transforms")
    
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs to train")
    parser.add_argument("--batch-size", default=32, type=int, help="number of images per gpu")
    
    parser.add_argument("--opt", default="sgd", type=str, choices=["sgd", "adam"], help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay (L2 penalty)")
    
    parser.add_argument("--by-iteration", action="store_true", help="convert scheduler to be per iteration, not per epoch")
    parser.add_argument("--lr-scheduler", default="cosa", type=str, choices=["step", "cosa", "exp"], help="lr scheduler")
    parser.add_argument("--lr-step", default=20, type=int, help="period of learning rate decay")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="multiplicative factor of learning rate decay")
    parser.add_argument("--lr-warmup-epochs", default=10, type=int, help="number of epochs to warmup")
    parser.add_argument("--lr-warmup-decay", default=0.001, type=float, help="decay of learning rate in warmup stage")
    
    parser.add_argument("--amp", action="store_true", help="use automatic mixed precision")
    parser.add_argument("--workers", default=8, type=int, help="number of data loading workers")
    parser.add_argument("--model-name", default="lorafb_snet18", type=str, help="name of model to train")
    parser.add_argument("--not-snn", action="store_true", help="model is not a snn")
    parser.add_argument("--not-recurrent", action="store_true", help="model is not a recurrent network")
    parser.add_argument("--pretrained", default=None, help="path of pretrained checkpoint")
    parser.add_argument("--output-path", default="logs/", help="path to save outputs")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    
    parser.add_argument("--local_rank", default=0, type=int, help="node rank for distributed training")
    args = parser.parse_args()
    return args


def set_deterministic():
    random.seed(_seed_)
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(_seed_)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def _get_cache_path(filepath, datapath):
    import hashlib
    
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(datapath, h[:10] + ".pt")
#     cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(args):
    if "UCF101" in args.dataset:
        return eval(f"load_UCF101")(args)

    
def load_UCF101(args):
    if args.dataset == "UCF101":
        dataset_loader = datasets.UCF101WithVideoID
    elif args.dataset == "UCF101_subset":
        dataset_loader = datasets.UCF101OneClip
    elif args.dataset == "UCF101_random":
        dataset_loader = datasets.UCF101RandomClip
    print("Loading data...")
    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(os.path.join(args.data_path, "train"), args.data_path)
    if args.cache_dataset and os.path.exists(cache_path):
        print(f"Loading train dataset from {cache_path}")
        train_set, _ = torch.load(cache_path)
    else:
        if args.dataset == "UCF101":
            transforms_train = transforms.Compose([
                ConvertBHWCtoBCHW(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize((256, 340)),
                transforms.RandomCrop((256, 256)),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            train_set = dataset_loader(
                root=args.data_path,
                annotation_path=args.annotation_path,
                frames_per_clip=args.frames,
                step_between_clips=args.f_steps,
                fold=1,
                train=True,
                transform=transforms_train
            )
        else:
            transforms_train = transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize((256, 340)),
                transforms.RandomCrop((256, 256)),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            train_set = dataset_loader(
                data_path=args.data_path,
                annotation_path=args.annotation_path,
                frames_per_clip=args.frames,
                downsample=1,
                fold=1,
                train=True,
                transform=transforms_train
            )
        if args.cache_dataset:
            print(f"Saving train dataset to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            utils.save_on_master((train_set, os.path.join(args.data_path, "train")), cache_path)
    print(f"Length of training data: {len(train_set)}")
    print(f"Took {time.time() - st}s")
    
    print("Loading validation data")
    st = time.time()
    cache_path = _get_cache_path(os.path.join(args.data_path, "test"), args.data_path)
    if args.cache_dataset and os.path.exists(cache_path):
        print(f"Loading test dataset from {cache_path}")
        test_set, _ = torch.load(cache_path)
    else:
        if args.dataset == "UCF101":
            transforms_test = transforms.Compose([
                ConvertBHWCtoBCHW(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize((256, 340)),
                transforms.CenterCrop((256, 256)),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            test_set = dataset_loader(
                root=args.data_path,
                annotation_path=args.annotation_path,
                frames_per_clip=args.frames,
                step_between_clips=args.f_steps,
                fold=1,
                train=False,
                transform=transforms_test
            )
        else:
            transforms_test = transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize((256, 340)),
                transforms.CenterCrop((256, 256)),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            test_set = dataset_loader(
                data_path=args.data_path,
                annotation_path=args.annotation_path,
                frames_per_clip=args.frames,
                downsample=1,
                fold=1,
                train=False,
                transform=transforms_test
            )
        if args.cache_dataset:
            print(f"Saving test dataset to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            utils.save_on_master((test_set, os.path.join(args.data_path, "test")), cache_path)
    print(f"Length of testing data: {len(test_set)}")
    print(f"Took {time.time() - st}s")
    
    print("Creating data loaders")
    g = torch.Generator()
    g.manual_seed(_seed_)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, seed=_seed_)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set, generator=g)
        test_sampler = torch.utils.data.SequentialSampler(test_set)
    
    return train_set, test_set, train_sampler, test_sampler


def load_model(args):
    return eval(f"{args.model_name}")(num_classes=101, cnf="ADD")


def get_logdir_name(args):
    if args.pretrained is None:
        logdir = f"{args.model_name}/"
    else:
        d = args.pretrained.split('/')
        logdir = f"{args.model_name}_imagenet_pretrained_{d[-2]}/"
    logdir += f"epochs{args.epochs}_bs{args.batch_size}_" \
              f"{args.opt}_lr{args.lr}_momentum{args.momentum}_wd{args.weight_decay}_"
    if args.lr_scheduler == "step":
        logdir += f"step_lrstep{args.lr_step}_gamma{args.lr_gamma}_"
    elif args.lr_scheduler == "cosa":
        logdir += f"cosa_"
    elif args.lr_scheduler == "exp":
        logdir += f"exp_gamma{args.lr_gamma}_"
    logdir += f"ws{args.world_size if args.distributed else 1}"
    return logdir


def set_optimizer(parameters, args):
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "adam":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def set_lr_scheduler(optimizer, iters_per_epoch, args):
    if args.lr_scheduler == "step":
        if args.by_iteration:
            lr_step = iters_per_epoch * args.lr_step
        else:
            lr_step = args.lr_step
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosa":
        if args.by_iteration:
            t_max = iters_per_epoch * (args.epochs - args.lr_warmup_epochs)
        else:
            t_max = args.epochs - args.lr_warmup_epochs
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif args.lr_scheduler == "exp":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    
    if args.lr_warmup_epochs > 0:
        if args.by_iteration:
            warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        else:
            warmup_iters = args.lr_warmup_epochs
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters])
    else:
        lr_scheduler = main_lr_scheduler
    return lr_scheduler


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, args, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=20, fmt="{global_avg:.4f}"))
    metric_logger.add_meter("acc", utils.SmoothedValue(window_size=20, fmt="{global_avg:.4f}"))
#     metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    
    header = f"Epoch: [{epoch}]"
    for i, (inputs, labels) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        inputs, labels = inputs.to(device), labels.to(device)
        if not args.not_recurrent:
            inputs = inputs.permute(1, 0, 2, 3, 4)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(inputs)
            if not args.not_recurrent:
                outputs = outputs.mean(0)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if not args.not_recurrent:
            if not args.not_snn:
                functional.reset_net(model)
                if args.model_name == "s_cornet":
                    if args.distributed:
                        model.module.reset_state()
                    else:
                        model.reset_state()
            else:
                if args.distributed:
                    model.module.reset_state()
                else:
                    model.reset_state()
        
        acc = utils.accuracy(outputs, labels)
        batch_size = labels.size(0)
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc"].update(acc.item(), n=batch_size)
#         metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        if args.by_iteration:
            lr_scheduler.step()
    
    metric_logger.synchronize_between_processes()
    train_loss, train_acc = metric_logger.meters["loss"].global_avg, metric_logger.meters["acc"].global_avg
    
    return train_loss, train_acc


def evaluate(model, criterion, data_loader, device, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=20, fmt="{global_avg:.4f}"))
    metric_logger.add_meter("acc", utils.SmoothedValue(window_size=20, fmt="{global_avg:.4f}"))
#     metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    
    header = f"Test:"
    num_processed_samples = 0
    
    with torch.inference_mode():
        for i, (inputs, labels) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            start_time = time.time()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if not args.not_recurrent:
                inputs = inputs.permute(1, 0, 2, 3, 4)
            
            outputs = model(inputs)
            if not args.not_recurrent:
                outputs = outputs.mean(0)
            loss = criterion(outputs, labels)
            
            if not args.not_recurrent:
                if not args.not_snn:
                    functional.reset_net(model)
                    if args.model_name == "s_cornet":
                        if args.distributed:
                            model.module.reset_state()
                        else:
                            model.reset_state()
                else:
                    if args.distributed:
                        model.module.reset_state()
                    else:
                        model.reset_state()
            
            acc = utils.accuracy(outputs, labels)
            batch_size = labels.size(0)
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc"].update(acc.item(), n=batch_size)
#             metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

            # FIXME need to take into account that the datasets could have been padded in distributed setup
            num_processed_samples += batch_size
            
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
         warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )
    
    metric_logger.synchronize_between_processes()
    test_loss, test_acc = metric_logger.meters["loss"].global_avg, metric_logger.meters["acc"].global_avg
    
    return test_loss, test_acc


def main():
    args = get_args()
    set_deterministic()
    utils.init_distributed_mode(args)
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_set, test_set, train_sampler, test_sampler = load_data(args)

    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=args.batch_size,
        sampler=test_sampler, 
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    
    print("Creating model")
    model = load_model(args)
    if not args.not_snn and args.model_name != "s_cornet":
        set_step_mode(model, 'm', (ConvRecurrentContainer, ))
        set_backend(model, 'cupy', neuron.BaseNode, (ConvRecurrentContainer, ))
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        checkpoint["model"].pop("fc.weight", None)
        checkpoint["model"].pop("fc.bias", None)
        model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = set_optimizer(model.parameters(), args)
    lr_scheduler = set_lr_scheduler(optimizer, len(train_loader), args)
    
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    if utils.is_main_process():
        logdir = get_logdir_name(args)
        logdir = os.path.join(args.output_path, args.dataset, logdir)
        os.makedirs(logdir, exist_ok=True)
        
        writer = SummaryWriter(logdir)
        with open(os.path.join(logdir, "args.txt"), 'w') as f:
            f.write(str(args))
        
        max_test_acc = -1.
    
    print("Start training...")
    st = time.time()
    for epoch in range(args.epochs):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss, train_accuracy = train_one_epoch(model, criterion, optimizer, lr_scheduler, train_loader, device, epoch, args, scaler=scaler)
        if not args.by_iteration:
            lr_scheduler.step()
        test_loss, test_accuracy = evaluate(model, criterion, test_loader, device, args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        if utils.is_main_process():
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_accuracy", train_accuracy, epoch)
            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("test_accuracy", test_accuracy, epoch)
            
            save_max_test_acc = False
            if test_accuracy > max_test_acc:
                max_test_acc = test_accuracy
                save_max_test_acc = True
            
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "test_acc": test_accuracy,
                "max_test_acc": max_test_acc,
            }
            if scaler is not None:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(logdir, f"checkpoint_latest.pth"))
            if save_max_test_acc:
                utils.save_on_master(checkpoint, os.path.join(logdir, f"checkpoint_max_test_acc.pth"))
        
        print(f"Total: train_acc={train_accuracy:.4f}  train_loss={train_loss:.4f}  test_acc={test_accuracy:.4f}  test_loss={test_loss:.4f}  total time={total_time_str}")
        print()
    
    total_time = time.time() - st
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Took {total_time_str}")


if __name__ == "__main__":
    main()

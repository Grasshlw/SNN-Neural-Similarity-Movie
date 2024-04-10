import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from PIL import Image


class UCF101OneClip(Dataset):
    def __init__(self, data_path, annotation_path, frames_per_clip, downsample, fold=1, train=True, transform=None, max_num_clips=1):
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.frames_per_clip = frames_per_clip
        self.downsample = downsample
        self.fold = fold
        self.train = train
        self.transform = transform
        
        class_info = pd.read_csv(os.path.join(annotation_path, "classInd.txt"), header=None, sep=' ')
        self.class_info = {}
        for i in range(len(class_info)):
            self.class_info[class_info.iloc[i, 1]] = class_info.iloc[i, 0]
        if self.train:
            file_name = os.path.join(annotation_path, f"trainframelist0{self.fold}.txt")
        else:
            file_name = os.path.join(annotation_path, f"testframelist0{self.fold}.txt")
        video_info = pd.read_csv(file_name, header=None, sep=' ')
        
        drop_index = []
        for i in range(len(video_info)):
            frames_num = video_info.iloc[i, 1]
            if frames_num <= self.frames_per_clip * self.downsample:
                drop_index.append(i)
        self.video_info = video_info.drop(drop_index)
    
    def __getitem__(self, index):
        video_path = self.video_info.iloc[index, 0]
        frames_num = self.video_info.iloc[index, 1]
        
        start_index = np.random.choice(frames_num - self.frames_per_clip * self.downsample)
        clips_index = start_index + np.arange(self.frames_per_clip) * self.downsample
        
        clips = []
        for idx in clips_index:
            img = Image.open(os.path.join(self.data_path, video_path, "%04d.jpg" % (idx + 1)))
            clips.append(self.transform(img))
        
        clips = torch.stack(clips, 0)
        
        class_name = video_path.split('/')[0]
        label = self.class_info[class_name] - 1
        
        return clips, label
    
    def __len__(self):
        return len(self.video_info)


class UCF101RandomClip(Dataset):
    def __init__(self, data_path, annotation_path, frames_per_clip, downsample, fold=1, train=True, transform=None, max_num_clips=1):
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.frames_per_clip = frames_per_clip
        self.downsample = downsample
        self.fold = fold
        self.train = train
        self.transform = transform
        
        class_info = pd.read_csv(os.path.join(annotation_path, "classInd.txt"), header=None, sep=' ')
        self.class_info = {}
        for i in range(len(class_info)):
            self.class_info[class_info.iloc[i, 1]] = class_info.iloc[i, 0]
        if self.train:
            file_name = os.path.join(annotation_path, f"trainframelist0{self.fold}.txt")
        else:
            file_name = os.path.join(annotation_path, f"testframelist0{self.fold}.txt")
        video_info = pd.read_csv(file_name, header=None, sep=' ')
        
        drop_index = []
        for i in range(len(video_info)):
            frames_num = video_info.iloc[i, 1]
            if frames_num <= self.frames_per_clip * self.downsample:
                drop_index.append(i)
        self.video_info = video_info.drop(drop_index)
    
    def __getitem__(self, index):
        video_path = self.video_info.iloc[index, 0]
        frames_num = self.video_info.iloc[index, 1]
        
        clips_index = np.random.choice(frames_num, size=self.frames_per_clip, replace=False)
        
        clips = []
        for idx in clips_index:
            img = Image.open(os.path.join(self.data_path, video_path, "%04d.jpg" % (idx + 1)))
            clips.append(self.transform(img))
        
        clips = torch.stack(clips, 0)
        
        class_name = video_path.split('/')[0]
        label = self.class_info[class_name] - 1
        
        return clips, label
    
    def __len__(self):
        return len(self.video_info)

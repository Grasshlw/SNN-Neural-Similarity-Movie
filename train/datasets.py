import torchvision


class UCF101WithVideoID(torchvision.datasets.UCF101):
    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, video_idx, label

# from colorVideoDataset import ColorVideoDataset
# from torch.utils.data import DataLoader
# from torchinfo import summary

import torch
import torch.nn as nn 

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

def get_3dResNet() -> nn.Module:    
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    num_classes = 8
    in_features = model.blocks[5].proj.in_features
    model.blocks[5].proj = nn.Linear(in_features, num_classes)
    
    # Unfreeze the new linear layer for training
    for param in model.blocks[5].proj.parameters():
        param.requires_grad = True
    print(model)
    return model

def get_resnet_transformer() -> Compose:
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8
    sampling_rate = 8
    frames_per_second = 30

    # Note that this transform is specific to the slow_R50 model.
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate)/frames_per_second
    return transform


def load_videos_from_direct(file_path, transform: Compose, subset_ratio=0.1):
    dataset = ColorVideoDataset(file_path)
    num_samples = int(len(dataset) * subset_ratio)
    
    def collate_fn(batch):
        videos = []
        for video_data, _, _ in batch:
            video_data = video_data.permute(1, 0, 2, 3)  # Change shape from (T, C, H, W) to (C, T, H, W)
            video_data = transform({"video": video_data})["video"]
            videos.append(video_data)
        return torch.stack(videos)
    
    from torch.utils.data import Subset
    indices = list(range(num_samples))
    subset_dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(subset_dataset, batch_size=32, collate_fn=collate_fn)
    
    # Return the dataloader for iteration, or process all at once
    all_videos = []
    for batch in dataloader:
        all_videos.append(batch)
    
    return torch.cat(all_videos, dim=0)

if __name__ == "__main__":
    model = get_3dResNet()
    transform = get_resnet_transformer()
    video_tensor = load_videos_from_direct('./colors', transform, subset_ratio=0.1)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    video_tensor = video_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    # print(video_tensor.shape)
    model.eval()
    with torch.no_grad():
        outputs = model(video_tensor)
    print(outputs.shape)
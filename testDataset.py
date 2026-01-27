import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import pytorch_lightning as pl
import torchvision

class TestVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=30):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        
        # Get all class folders (0, 1, 2, 3, 4, 5, 6, 7)
        self.class_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_folders)}
        
        # Build dataset index
        self.video_paths = []
        self.labels = []
        
        for class_folder in self.class_folders:
            class_path = os.path.join(root_dir, class_folder)
            label = self.class_to_idx[class_folder]
            
            # Get all sequence folders for this class
            seq_folders = sorted([s for s in os.listdir(class_path) if s.startswith('seq_')])
            
            for seq_folder in seq_folders:
                seq_path = os.path.join(class_path, seq_folder)
                self.video_paths.append(seq_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Get all frame files in the sequence folder
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            
            # Load image
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms if provided
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            frames.append(frame)
        
        # Convert to tensor
        video_frames = torch.stack(frames)
        
        # Handle sequence length
        if self.sequence_length is not None:
            if len(video_frames) >= self.sequence_length:
                video_frames = video_frames[:self.sequence_length]
            else:
                padding_needed = self.sequence_length - len(video_frames)
                last_frame = video_frames[-1:]
                padding = last_frame.repeat(padding_needed, 1, 1, 1)
                video_frames = torch.cat([video_frames, padding], dim=0)
        
        return video_frames, label
    
    def get_num_classes(self):
        return len(self.class_folders)
    

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str=None):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.full_dataset = TestVideoDataset(root_dir=self.data_dir, transform=transform)

    def test_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
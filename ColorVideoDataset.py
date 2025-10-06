from torch.utils.data import Dataset
import os
import cv2
import torch

class ColorVideoDataset(Dataset):
    """
    PyTorch Dataset for color video classification.
    Each video is a sequence of frames, labeled by the color folder it belongs to.
    """
    
    def __init__(self, root_dir, transform=None, sequence_length=30):
        """
        Args:
            root_dir (string): Directory with all the color folders.
            transform (callable, optional): Optional transform to be applied on frames.
            sequence_length (int, optional): Fixed length for video sequences. If None, uses all frames.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        
        # Get all color categories
        self.color_folders = [f for f in os.listdir(root_dir) if f.startswith('colors_')]
        self.color_to_idx = {color.replace('colors_', ''): idx for idx, color in enumerate(self.color_folders)}
        self.idx_to_color = {idx: color.replace('colors_', '') for idx, color in enumerate(self.color_folders)}
        
        # Build dataset index
        self.video_paths = []
        self.labels = []
        
        for color_folder in self.color_folders:
            color_path = os.path.join(root_dir, color_folder)
            color_name = color_folder.replace('colors_', '')
            label = self.color_to_idx[color_name]
            
            # Get all video folders for this color
            video_folders = [v for v in os.listdir(color_path) if v.startswith(color_name + '_Video_')]
            
            for video_folder in video_folders:
                video_path = os.path.join(color_path, video_folder)
                self.video_paths.append(video_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            video_frames: Tensor of shape (T, C, H, W) where T is sequence length
            label: Integer label for the color category
            video_info: Dictionary with metadata
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Get all frame files in the video folder
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            
            # Load image
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Apply transforms if provided
            if self.transform:
                frame = self.transform(frame)
            else:
                # Convert to tensor and normalize to [0, 1]
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            frames.append(frame)
            del frame
        
        # Convert to tensor
        video_frames = torch.stack(frames)  # Shape: (T, C, H, W)
        del frames
        
        # Handle sequence length
        if self.sequence_length is not None:
            if len(video_frames) >= self.sequence_length:
                # Randomly sample frames or take first N frames
                video_frames = video_frames[:self.sequence_length]
            else:
                # Pad with last frame if sequence is shorter
                padding_needed = self.sequence_length - len(video_frames)
                last_frame = video_frames[-1:]
                padding = last_frame.repeat(padding_needed, 1, 1, 1)
                video_frames = torch.cat([video_frames, padding], dim=0)
                del padding, last_frame
        
        # Create metadata
        video_info = {
            'video_path': video_path,
            'color': self.idx_to_color[label],
            'num_frames': len(frame_files),
            'video_folder': os.path.basename(video_path)
        }
        
        return video_frames, label, video_info
    
    def get_class_names(self):
        """Returns list of color class names"""
        return [self.idx_to_color[i] for i in range(len(self.idx_to_color))]
    
    def get_num_classes(self):
        """Returns number of color classes"""
        return len(self.color_to_idx)
import torch
import torch.nn as nn 
from transformers import VideoMAEImageProcessor, VideoMAEModel
from colorVideoDataset import ColorVideoDataset
from torchinfo import summary

class VideoMAEClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(VideoMAEClassifier, self).__init__()
        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        for param in self.model.parameters():
            param.requires_grad = False
        # Classifier head
        self.fc_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

    
    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        sequence_output = outputs.last_hidden_state 

        output = sequence_output.mean(1)
        output = self.fc_norm(output)

        logits = self.classifier(output)  # (batch_size, num_classes)
        return logits
    

if __name__ == "__main__":
    # Load dataset with return_tensors=False to get numpy arrays
    dataset = ColorVideoDataset(
        root_dir='./colors',
        sequence_length=16,  # VideoMAE typically uses 16 frames
        return_tensors=False  # Return numpy arrays instead of tensors
    )

    # Get a single video sample
    video_frames, label, video_info = dataset[0]

    print(f"\nVideo info: {video_info}")
    print(f"Label: {label} ({dataset.idx_to_color[label]})")
    print(f"Number of frames: {len(video_frames)}")
    print(f"First frame shape: {video_frames[0].shape}")

    # Load VideoMAE processor and model
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEClassifier(num_classes=8)
    # print(summary(model, input_size=(1, 16, 3, 224, 224)))
    # Process the video frames
    inputs = processor(video_frames, return_tensors="pt")

    print(f"\nInput tensor shape: {inputs['pixel_values'].shape}")

    # Perform a forward pass
    outputs = model(inputs['pixel_values'])

    last_hidden_states = outputs  # (batch_size, num_classes)
    print(f"Output hidden states shape: {list(last_hidden_states.shape)}")
    print(f"\nInference successful! Hidden state: [batch_size, num_patches, hidden_size]")
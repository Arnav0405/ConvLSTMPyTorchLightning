import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel
from colorVideoDataset import ColorVideoDataset
from torchinfo import summary

# Load dataset with return_tensors=False to get numpy arrays
dataset = ColorVideoDataset(
    root_dir='./colors',
    sequence_length=16,  # VideoMAE typically uses 16 frames
    return_tensors=False  # Return numpy arrays instead of tensors
)


print(f"\nVideo info: {video_info}")
print(f"Label: {label} ({dataset.idx_to_color[label]})")
print(f"Number of frames: {len(video_frames)}")
print(f"First frame shape: {video_frames[0].shape}")

# Load VideoMAE processor and model
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
# print(summary(model, input_size=(1, 16, 3, 224, 224)))
# Process the video frames
inputs = processor(video_frames, return_tensors="pt")

print(f"\nInput tensor shape: {inputs['pixel_values'].shape}")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(f"Output hidden states shape: {list(last_hidden_states.shape)}")
print(f"\nInference successful! Hidden state: [batch_size, num_patches, hidden_size]")
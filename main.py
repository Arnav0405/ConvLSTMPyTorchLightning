from torchinfo import summary 
import torch
from models.EncoderDecoderCLSTM import EncoderDecoderCLSTM
from torch.nn import functional as F

def main():
    num_classes = 8
    model = EncoderDecoderCLSTM(nf=64, in_chan=3)
    print("=== Model Summary with torchinfo (using input_data) ===")
    batch_size = 4  # Multiple videos
    seq_len = 30    # 30 frames
    channels = 3    # RGB channels
    height = 64     
    width = 64      

    x = torch.randn(batch_size, seq_len, channels, height, width)
    true_labels = torch.randint(0, num_classes, (batch_size,))
    modules = list(model.modules())
    module_0 = modules[0]
    
    summary(module_0, input_data=[x], verbose=1)

    # Forward pass
    print("\n=== Forward Pass Test ===")
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output breakdown: (batch={output.shape[0]}, num_classes={output.shape[1]})")
    
    # Convert logits to probabilities
    probabilities = F.softmax(output, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)
    
    print(f"\nLogits:\n{output}")
    print(f"\nProbabilities:\n{probabilities}")
    print(f"\nPredicted classes: {predicted_classes}")
    print(f"True labels:       {true_labels}")
    
    # Calculate accuracy
    accuracy = (predicted_classes == true_labels).float().mean()
    print(f"\nRandom accuracy: {accuracy:.2%}")
    
    # Test loss calculation
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, true_labels)
    print(f"Cross-entropy loss: {loss:.4f}")
if __name__ == "__main__":
    main()

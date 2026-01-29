import torch
from torch.utils.data import TensorDataset

def get_random_dataset(num_samples=1000, input_size=784, output_size=10):
    # Generate random input data
    x = torch.randn(num_samples, input_size)
    # Generate random labels
    y = torch.randint(0, output_size, (num_samples,))
    dataset = TensorDataset(x, y)
    return dataset

def dataloader_from_dataset(dataset=get_random_dataset(), batch_size=32, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
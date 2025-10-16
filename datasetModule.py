import torchvision
import torch
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from colorVideoDataset import ColorVideoDataset

def video_collate_fn(batch):
        videos, labels, infos = zip(*batch)
        videos = torch.stack(videos).float()
        labels = torch.tensor(labels).long()
        return videos, labels, infos

class GestureDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, num_workers=15):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str=None):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        full_dataset = ColorVideoDataset(root_dir=self.data_dir, transform=transform)
        self.train_dataset, val_set = random_split(
                full_dataset,
                [int(0.8 * len(full_dataset)), len(full_dataset) - int(0.8 * len(full_dataset))], 
                generator=torch.Generator().manual_seed(42)
            )
        
        self.val_dataset, self.test_dataset = random_split(
             val_set, 
             [int(0.5) * len(val_set), len(val_set) - int(0.5 * len(val_set))]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=video_collate_fn, persistent_workers=True, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=video_collate_fn, persistent_workers=True, pin_memory=False)
    
    def test_dataloader(self):
         return DataLoader(self.test_dataset, batch_size= self.batch_size, shuffle=False, collate_fn=video_collate_fn, persistent_workers=True)
    
    def get_class_names(self):
        temp_dataset = ColorVideoDataset(root_dir=self.data_dir)
        return temp_dataset.get_class_names()
    

if __name__ == "__main__":
    mod = GestureDataModule(data_dir="./colors")
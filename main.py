import torchvision
import torch
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ColorVideoDataset import ColorVideoDataset
from training import ConvLSTM_GestureRecognitionModel

class GestureDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, num_workers=15):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str=None):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((64, 64))
        ])
        full_dataset = ColorVideoDataset(root_dir=self.data_dir, transform=transform)
        self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [int(0.8 * len(full_dataset)), len(full_dataset) - int(0.8 * len(full_dataset))], 
                generator=torch.Generator().manual_seed(42)
            )

    def video_collate_fn(self, batch):
        videos, labels, infos = zip(*batch)
        videos = torch.stack(videos)
        labels = torch.tensor(labels)
        return videos, labels, infos

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.video_collate_fn, persistent_workers=False, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.video_collate_fn, persistent_workers=False, pin_memory=False)

def main(num_epochs):
    data_mod = GestureDataModule(data_dir='./colors', batch_size=2) 
    data_mod.setup()
    model = ConvLSTM_GestureRecognitionModel(num_classes=8)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='tcnlstm-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(max_epochs=num_epochs, accelerator='cuda',
                      callbacks=[checkpoint_callback, early_stop_callback],
                      precision=16)
    trainer.fit(model, datamodule=data_mod)

    torch.save(model.state_dict(), "ConvLstm_final.pth")
    
if __name__ == "__main__":
    EPOCHS = 200
    main(EPOCHS)

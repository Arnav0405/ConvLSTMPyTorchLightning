import os
import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.TCN_With_LSTM import TCN_LSTM

class TCNLSTMClassifier(pl.LightningModule):
    def __init__(self, num_classes=8, in_channels=3, conv_channels=[64, 128], num_blocks=2,
                 spatial_kernel=3, temporal_kernel=3, lstm_hidden=256, lstm_layers=2, learning_rate=1e-3):
        super().__init__()
        self.model = TCN_LSTM(
            in_channels=in_channels,
            conv_channels=conv_channels,
            num_blocks=num_blocks,
            spatial_kernel=spatial_kernel,
            temporal_kernel=temporal_kernel,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            num_classes=num_classes
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if x.shape[1] != 3:  # If not (B, C, T, H, W), permute
            x = x.permute(0, 2, 1, 3, 4)
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class GestureDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
        ])
        self.train_dataset = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=transform)
        self.val_dataset = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def run_training():
    data_dir = 'path_to_your_data'  # Update this path
    batch_size = 4
    num_classes = 8
    num_epochs = 100

    data_module = GestureDataModule(data_dir=data_dir, batch_size=batch_size)
    model = TCNLSTMClassifier(num_classes=num_classes)

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

    trainer = Trainer(
        max_epochs=num_epochs, 
        gpus=1 if torch.cuda.is_available() else 0, 
        callbacks=[checkpoint_callback, early_stop_callback]
        )
    
    trainer.fit(model, datamodule=data_module)
    
    torch.save(model.state_dict(), "tcnlstm_final.pth")

if __name__ == "__main__":
    run_training()
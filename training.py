import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.EncoderDecoderCLSTM import EncoderDecoderCLSTM


class ConvLSTM_GestureRecognitionModel(pl.LightningModule):
    def __init__(self, num_classes=8, nf=64, in_chan=3, learning_rate=1e-3):
        super(ConvLSTM_GestureRecognitionModel, self).__init__()
        self.model = EncoderDecoderCLSTM(nf=nf, in_chan=in_chan)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

    def forward(self, x):
        x = x.to(device='cuda')
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 1, 4, 2, 3)  # Rearrange to (B, T, C, H, W)
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)

        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
def run_training():
    data_dir = 'path_to_your_data'  # Update this path
    batch_size = 8
    num_classes = 8
    num_epochs = 200

    # data_module = GestureDataModule(data_dir=data_dir, batch_size=batch_size)
    model = ConvLSTM_GestureRecognitionModel(num_classes=num_classes)

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

    trainer = Trainer(max_epochs=num_epochs, gpus=1 if torch.cuda.is_available() else 0,
                      callbacks=[checkpoint_callback, early_stop_callback])
    # trainer.fit(model, datamodule=data_module)

    # torch.save(model.state_dict(), "ConvLstm_final.pth")
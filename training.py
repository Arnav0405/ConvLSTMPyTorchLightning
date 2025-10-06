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
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        # Debug: Print original shape
        print(f"Original x shape: {x.shape}")
        
        # Fix the permutation - if your dataset outputs (B, T, H, W, C), then:
        if x.shape[-1] == 3:  # If channels are last
            x = x.permute(0, 1, 4, 2, 3)  # (B, T, H, W, C) -> (B, T, C, H, W)
        elif x.shape[2] == 3:  # If channels are already in position 2
            # No permutation needed, already (B, T, C, H, W)
            pass
        else:
            # Your specific case - if it's (B, C, T, H, W), convert to (B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        
        print(f"After permutation x shape: {x.shape}")
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)

        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        # Apply the same fix here
        if x.shape[-1] == 3:
            x = x.permute(0, 1, 4, 2, 3)
        elif x.shape[2] == 3:
            pass
        else:
            x = x.permute(0, 2, 1, 3, 4)
        
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

        del y_hat, preds
        return {'val_loss': loss, 'val_acc': acc}

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
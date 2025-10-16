import os
import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.TCN_With_LSTM import TCN_LSTM
from datasetModule import GestureDataModule

class TCNLSTMClassifier(pl.LightningModule):
    def __init__(self, num_blocks, num_classes=8, in_channels=3, conv_channels=[64, 128],
                 spatial_kernel=3, temporal_kernel=3, lstm_hidden=256, lstm_layers=2, input_size=(128, 128), learning_rate=1e-3):
        super().__init__()
        self.model = TCN_LSTM(
            in_channels=in_channels,
            conv_channels=conv_channels,
            num_blocks=num_blocks,
            spatial_kernel=spatial_kernel,
            temporal_kernel=temporal_kernel,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            num_classes=num_classes,
            input_size=input_size
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
    
    def on_fit_start(self):
        # Initialize inner model dynamically with a sample batch
        sample_batch = next(iter(self.trainer.datamodule.train_dataloader()))[0]
        sample_batch = sample_batch.to(self.device)
        self.model.initialize_lstm(sample_batch)


    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        opt = self.optimizers()
        if opt is not None:
            lr = opt.param_groups[0]['lr']
            self.log("learning_rate", lr, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        
        del y_hat, x, y
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

        del y_hat, preds
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def run_training(EPOCHS):
    data_mod = GestureDataModule(data_dir='./colors', batch_size=16) 
    data_mod.setup()
    num_classes = 8

    model = TCNLSTMClassifier(num_classes=num_classes, num_blocks=2, learning_rate=1e-4)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='tcn_lstm-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(max_epochs=EPOCHS, accelerator='cuda',
                      callbacks=[checkpoint_callback, early_stop_callback],
                      precision=16, limit_val_batches=0.3, num_sanity_val_steps=0)
    
    trainer.fit(model, datamodule=data_mod)
    
    torch.save(model.state_dict(), "tcnlstm_final.pth")

if __name__ == "__main__":
    run_training(100)
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar

from models.EncoderDecoderCLSTM import EncoderDecoderCLSTM

class ConvLSTM_GestureRecognitionModel(pl.LightningModule):
    def __init__(self, num_classes=8, nf=64, in_chan=3, learning_rate=1.905e-05):
        super().__init__()
        self.model = EncoderDecoderCLSTM(nf=nf, in_chan=in_chan)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Timing
        self.train_epoch_start_time = None
        self.train_batch_start_time = None

        # Accuracy tracking
        self.train_correct = 0
        self.train_total = 0

    # --------------------------- Timing & Logging ---------------------------

    def on_train_epoch_start(self):
        self.train_epoch_start_time = time.time()
        self.train_correct = 0
        self.train_total = 0

        opt = self.optimizers()
        if opt is not None:
            lr = opt.param_groups[0]['lr']
            self.log("learning_rate", lr, prog_bar=True)

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.train_epoch_start_time
        train_acc = self.train_correct / self.train_total if self.train_total > 0 else 0.0

        # Log once per epoch
        self.log("train_epoch_time_sec", epoch_time, prog_bar=True)
        self.log("train_acc", train_acc, prog_bar=True)

    def on_train_batch_start(self, batch, batch_idx):
        self.train_batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_time = time.time() - self.train_batch_start_time
        self.log("train_batch_time_sec", batch_time, prog_bar=False)

    # --------------------------- Forward ---------------------------

    def forward(self, x):
        return self.model(x)

    # --------------------------- Training Step ---------------------------

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        # Ensure (B, T, C, H, W)
        if x.shape[-1] == 3:  
            x = x.permute(0, 1, 4, 2, 3)
        elif x.shape[2] != 3:  
            x = x.permute(0, 2, 1, 3, 4)
        
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)

        # Accumulate training accuracy stats
        self.train_correct += correct
        self.train_total += total

        # Single clean log per step
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    # --------------------------- Validation Step ---------------------------

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        if x.shape[-1] == 3:
            x = x.permute(0, 1, 4, 2, 3)
        elif x.shape[2] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        # Log once per validation step (aggregated automatically per epoch)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

        return {'val_loss': loss, 'val_acc': acc}

    # --------------------------- Optimizer ---------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class CustomProgressBar(TQDMProgressBar):
    """Custom progress bar to show epoch and batch timing."""
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description("Training")
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("Validating")
        return bar

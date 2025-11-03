import torchvision
import torch
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datasetModule import GestureDataModule
from training import ConvLSTM_GestureRecognitionModel, CustomProgressBar

def main(num_epochs):
    data_mod = GestureDataModule(data_dir='./colors', batch_size=16) 
    data_mod.setup()
    model = ConvLSTM_GestureRecognitionModel(num_classes=8, learning_rate=1.9e-5)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='convlstm-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    progress_bar = CustomProgressBar()

    trainer = Trainer(max_epochs=num_epochs, accelerator='cuda',
                      callbacks=[checkpoint_callback, early_stop_callback, progress_bar], 
                      enable_progress_bar=True,
                      precision=16, limit_val_batches=0.3, num_sanity_val_steps=0)
    trainer.fit(model, datamodule=data_mod)
    
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model, datamodule=data_mod)
    # print(lr_finder.suggestion())       # 1.9054607179632464e-05

    torch.save(model.state_dict(), "ConvLstm_final.pth")
    
if __name__ == "__main__":
    EPOCHS = 150
    main(EPOCHS)

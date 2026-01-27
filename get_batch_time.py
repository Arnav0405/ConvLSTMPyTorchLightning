import time
from time import perf_counter
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from datasetModule import GestureDataModule
from training import ConvLSTM_GestureRecognitionModel

def _get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def train_on_sample(sample_size: int,
                    epochs: int = 1,
                    batch_size: int = 8,
                    num_workers: int = 4,
                    verbose: bool = False) -> Tuple[float, ConvLSTM_GestureRecognitionModel]:
    """
    Train the model on a random subset of the training dataset of size `sample_size`.
    Returns (elapsed_seconds, trained_model).
    """
    # prepare datamodule and datasets
    dm = GestureDataModule(data_dir='./colors', batch_size=batch_size)
    dm.setup()

    full_train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    dataset = full_train_loader.dataset
    n_total = len(dataset)
    n = min(sample_size, n_total)
    if n == 0:
        raise ValueError("sample_size must be > 0 and dataset must be non-empty")

    indices = torch.randperm(n_total)[:n].tolist()
    train_subset = Subset(dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # instantiate model
    model = ConvLSTM_GestureRecognitionModel(num_classes=8, learning_rate=1.9e-5)

    # trainer (no progress bar to reduce noise)
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=_get_device(),
        devices=1 if torch.cuda.is_available() else None,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False
    )

    if verbose:
        print(f"Training on subset {n}/{n_total} samples, epochs={epochs}, batch_size={batch_size}")

    start = perf_counter()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = perf_counter() - start

    if verbose:
        print(f"Elapsed: {elapsed:.3f}s")

    return elapsed, model

def measure_training_times(sample_sizes: List[int],
                           epochs: int = 1,
                           batch_size: int = 8,
                           num_workers: int = 4) -> List[Tuple[int, float]]:
    """
    For each sample size in sample_sizes, train and record elapsed time.
    Returns list of tuples (sample_size, elapsed_seconds).
    """
    results = []
    for s in sample_sizes:
        elapsed, _ = train_on_sample(sample_size=s, epochs=epochs, batch_size=batch_size,
                                     num_workers=num_workers, verbose=True)
        results.append((s, elapsed))
    return results

def inference_on_dataloader(model: torch.nn.Module,
                            dataloader: DataLoader,
                            device: str = None,
                            warmup_batches: int = 1) -> Tuple[float, int, float]:
    """
    Run inference over dataloader using model and measure time.
    Returns (elapsed_seconds, total_samples, seconds_per_sample)
    """
    device = device or _get_device()
    model.to(device).eval()
    total_samples = 0

    with torch.no_grad():
        # Warmup
        if warmup_batches > 0:
            batches = 0
            for xb, _ in dataloader:
                xb = xb.to(device)
                _ = model(xb)
                batches += 1
                if batches >= warmup_batches:
                    break

        start = perf_counter()
        for xb, _ in dataloader:
            xb = xb.to(device)
            _ = model(xb)
            total_samples += xb.size(0)
        elapsed = perf_counter() - start

    secs_per_sample = elapsed / total_samples if total_samples > 0 else float('inf')
    return elapsed, total_samples, secs_per_sample

if __name__ == "__main__":
    # Example usage: measure training time as sample increases
    SAMPLE_SIZES = [50, 100, 200, 400]   # adjust to dataset size
    EPOCHS = 1
    BATCH_SIZE = 8

    print("Measuring training times for sample sizes:", SAMPLE_SIZES)
    times = measure_training_times(SAMPLE_SIZES, epochs=EPOCHS, batch_size=BATCH_SIZE)

    for s, t in times:
        print(f"Sample {s:5d} -> {t:.3f} s (total training time for {EPOCHS} epoch(s))")

    # Example inference on a small validation subset
    dm = GestureDataModule(data_dir='./colors', batch_size=BATCH_SIZE)
    dm.setup()
    test_loader = dm.test_dataloader()
    # Train a tiny model on 100 samples to obtain weights for the demo
    _, trained_model = train_on_sample(sample_size=100, epochs=1, batch_size=BATCH_SIZE, verbose=False)
    elapsed, n_samples, sec_per = inference_on_dataloader(trained_model, test_loader)
    print(f"Inference: {n_samples} samples in {elapsed:.3f}s -> {sec_per*1000:.3f} ms/sample")
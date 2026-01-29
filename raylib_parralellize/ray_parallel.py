import os
import tempfile

import ray
from ray import tune
from ray.train.torch import TorchTrainer, prepare_model, prepare_data_loader
from ray.train.torch.config import TorchConfig
from ray.train import ScalingConfig

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"     # For Windows compatibility
ray.init()

def train_func(config):
    from models import model1, model2, model3
    from dataloaders import dataloader_from_dataset

    model_name = config["model_name"]
    input_size = config["input_size"]
    output_size = config["output_size"]
    
    # Instantiate the model
    if model_name == "model1":
        model = model1(input_size, output_size)
    elif model_name == "model2":
        model = model2(input_size, output_size)
    elif model_name == "model3":
        model = model3(input_size, output_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    # Prepare model for distributed training
    model = prepare_model(model)
    dataloader = dataloader_from_dataset()
    train_loader = prepare_data_loader(dataloader) 
        
    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    
    for epoch in range(50):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        for x, y in train_loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # [3] Report metrics and checkpoint.
        metrics = {"loss": loss.item(), "epoch": epoch}
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.module.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pt")
            )
            ray.train.report(
                metrics,
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
        if ray.train.get_context().get_world_rank() == 0:
            print(metrics)

# Define the search space for different models
config = {
    "model_name": tune.choice(["model1", "model2", "model3"]),
    "input_size": tune.choice([784]),  # Can be made variable if needed
    "output_size": tune.choice([10]),
}

scaling_config = ScalingConfig(num_workers=2, use_gpu=False)

torch_config = TorchConfig(backend="gloo")  # For Windows compatibility

trainer = TorchTrainer(
    train_func,
    train_loop_config={"model_name": ["model1", "model2", "model3"], "input_size": 784, "output_size": 10},
    scaling_config=scaling_config,
    torch_config=torch_config
)

result = trainer.fit()

ray.shutdown()
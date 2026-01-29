import sys
import os
import torch.distributed as dist
from random import randint
import torch

os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(randint(20000, 55555))

device = torch.device("cuda")
n_gpus = 1
rank = 0

dist.init_process_group(
	backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
	init_method="env://",
	world_size=n_gpus if device.type == "cuda" else 1,
	rank=rank if device.type == "cuda" else 0,
)

print("done")

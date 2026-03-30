import os
import time

import torch
import torch.distributed as dist
import yaml
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from src.model import CatDogCNN


def load_config(config_path="configs/config.yaml"):
    """Load training configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_ddp():
    """Initialize distributed training environment."""
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_ddp():
    """Destroy distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train():
    """Train model using Distributed Data Parallel + AMP."""
    config = load_config()

    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    train_dir = config["dataset"]["train_dir"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    image_size = config["training"]["image_size"]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    model = CatDogCNN().to(device)
    model = DDP(model, device_ids=[local_rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ✅ AMP (REQUIRED FOR MM3)
    scaler = torch.amp.GradScaler("cuda")

    if rank == 0:
        print(f"Running with {world_size} GPU(s)")
        print(f"Device: {device}")

    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # ✅ AMP forward pass
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            # ✅ AMP backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # ✅ Aggregate loss across GPUs
        loss_tensor = torch.tensor(running_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_loss = loss_tensor.item() / world_size / len(train_loader)
            print(f"[DDP] Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time

    if rank == 0:
        print(f"Total training time: {total_time:.2f} seconds")

    cleanup_ddp()


if __name__ == "__main__":
    train()
# flake8: noqa: E402
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import CatDogCNN


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train(ds_config_path="configs/ds_config_zero3.json"):
    import deepspeed

    config = load_config()

    train_dir = config["dataset"]["train_dir"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    image_size = config["training"]["image_size"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(train_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = CatDogCNN()
    loss_fn = nn.CrossEntropyLoss()

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config_path,
    )

    print(f"Using device: {model_engine.device}", flush=True)
    print(f"DeepSpeed config: {ds_config_path}", flush=True)

    model_engine.train()

    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in loader:
            images = images.to(model_engine.device, non_blocking=True)
            labels = labels.to(model_engine.device, non_blocking=True)

            outputs = model_engine(images)
            loss = loss_fn(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)

        print(
            f"[DeepSpeed] Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}",
            flush=True
        )

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds", flush=True)


if __name__ == "__main__":
    train()

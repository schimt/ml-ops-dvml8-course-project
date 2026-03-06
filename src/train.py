import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CatDogCNN


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train():
    config = load_config()

    train_dir = config["dataset"]["train_dir"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    image_size = config["training"]["image_size"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CatDogCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cat_dog_cnn.pth")
    print("Model saved to models/cat_dog_cnn.pth")


if __name__ == "__main__":
    train()
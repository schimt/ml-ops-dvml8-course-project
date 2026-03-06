import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CatDogCNN


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test():
    config = load_config()

    test_dir = config["dataset"]["test_dir"]
    batch_size = config["training"]["batch_size"]
    image_size = config["training"]["image_size"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load("models/cat_dog_cnn.pth", map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    test()
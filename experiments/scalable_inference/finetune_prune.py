# flake8: noqa: E402
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.utils.prune as prune
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.scalable_inference.infer import evaluate, get_test_loader
from src.model import CatDogCNN


def get_train_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder("data/train", transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_model(path):
    model = CatDogCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def apply_pruning(model, amount):
    for module in model.modules():
        if isinstance(
            module,
            (torch.nn.Conv2d, torch.nn.Linear),
        ):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


def remove_pruning(model):
    for module in model.modules():
        if isinstance(
            module,
            (torch.nn.Conv2d, torch.nn.Linear),
        ):
            try:
                prune.remove(module, "weight")
            except (AttributeError, ValueError):
                pass
    return model


def train(model, loader, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    for _ in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    return model


if __name__ == "__main__":
    model_path = "models/cat_dog_cnn.pth"

    train_loader = get_train_loader(32)
    test_loader = get_test_loader(32)

    model = load_model(model_path)
    model = apply_pruning(model, 0.8)
    model = remove_pruning(model)

    print("Before fine-tuning:")
    acc_before, _, _ = evaluate(model, test_loader)
    print(f"Accuracy: {acc_before:.4f}")

    model = train(model, train_loader, epochs=3)

    print("After fine-tuning:")
    acc_after, _, _ = evaluate(model, test_loader)
    print(f"Accuracy: {acc_after:.4f}")

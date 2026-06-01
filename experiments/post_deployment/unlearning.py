# flake8: noqa: E402
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def get_subset(dataset, condition):
    indices = [i for i, (_, y) in enumerate(dataset) if condition(y)]
    return Subset(dataset, indices)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def train(model, loader, optimizer, epochs=3):
    criterion = nn.CrossEntropyLoss()
    model.train()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


# task 1: train on all digits
transform = transforms.ToTensor()
dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

full_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("training on all digits...")
train(model, full_loader, optimizer, epochs=3)

loader_7 = DataLoader(
    get_subset(dataset, lambda y: y == 7),
    batch_size=64,
)
loader_non7 = DataLoader(
    get_subset(dataset, lambda y: y != 7),
    batch_size=64,
)

acc_all_before = evaluate(model, full_loader)
acc_7_before = evaluate(model, loader_7)
acc_non7_before = evaluate(model, loader_non7)

print("\nbefore unlearning")
print(f"all: {acc_all_before:.4f}")
print(f"class 7: {acc_7_before:.4f}")
print(f"non-7: {acc_non7_before:.4f}")


# task 2: unlearn class 7
print("\nunlearning class 7...")

criterion = nn.CrossEntropyLoss()
unlearn_optimizer = optim.Adam(model.parameters(), lr=0.0001)

for x, y in loader_7:
    x, y = x.to(device), y.to(device)
    unlearn_optimizer.zero_grad()
    loss = criterion(model(x), y)
    (-0.2 * loss).backward()
    unlearn_optimizer.step()


# task 3: evaluate
acc_all_after = evaluate(model, full_loader)
acc_7_after = evaluate(model, loader_7)
acc_non7_after = evaluate(model, loader_non7)

print("\nafter unlearning")
print(f"all: {acc_all_after:.4f}")
print(f"class 7: {acc_7_after:.4f}")
print(f"non-7: {acc_non7_after:.4f}")

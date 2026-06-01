import os
import random

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Simple MLP for MNIST classification."""

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


def get_digit_subset(dataset, min_digit, max_digit):
    indices = [
        i for i, (_, label) in enumerate(dataset)
        if min_digit <= label <= max_digit
    ]
    return Subset(dataset, indices)


def train(model, loader, optimizer, epochs=2):
    criterion = nn.CrossEntropyLoss()
    model.train()

    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def train_with_tracking(model, loader, optimizer, old_loader, epochs=2):
    old_accuracy_history = []

    for epoch in range(epochs):
        train(model, loader, optimizer, epochs=1)
        old_accuracy = evaluate(model, old_loader)
        old_accuracy_history.append(old_accuracy)
        print(
            f"Epoch {epoch + 1} while training on 5-9: "
            f"accuracy on 0-4 = {old_accuracy:.4f}"
        )

    return old_accuracy_history


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def create_replay_loader(new_dataset, memory, batch_size=64):
    combined_data = list(new_dataset) + memory
    return DataLoader(combined_data, batch_size=batch_size, shuffle=True)


def save_forgetting_plot(history):
    os.makedirs("artifacts", exist_ok=True)

    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Epoch while training on digits 5-9")
    plt.ylabel("Accuracy on old task digits 0-4")
    plt.title("Catastrophic Forgetting")
    plt.grid()
    plt.savefig("artifacts/mm7_forgetting.png")
    plt.close()


def main():
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    data_0_4 = get_digit_subset(train_dataset, 0, 4)
    data_5_9 = get_digit_subset(train_dataset, 5, 9)

    loader_0_4 = DataLoader(data_0_4, batch_size=64, shuffle=True)
    loader_5_9 = DataLoader(data_5_9, batch_size=64, shuffle=True)

    # -------------------------
    # Naive sequential training
    # -------------------------
    naive_model = Net().to(device)
    naive_optimizer = optim.Adam(naive_model.parameters(), lr=0.001)

    train(naive_model, loader_0_4, naive_optimizer, epochs=2)
    acc_0_4_before = evaluate(naive_model, loader_0_4)

    print("\n--- Naive Sequential Training ---")
    print(f"Accuracy on 0-4 after initial training: {acc_0_4_before:.4f}")

    forgetting_history = train_with_tracking(
        naive_model,
        loader_5_9,
        naive_optimizer,
        loader_0_4,
        epochs=3,
    )

    acc_0_4_after_naive = evaluate(naive_model, loader_0_4)
    acc_5_9_after_naive = evaluate(naive_model, loader_5_9)

    print("\nAfter naive training on 5-9:")
    print(f"Accuracy on old digits 0-4: {acc_0_4_after_naive:.4f}")
    print(f"Accuracy on new digits 5-9: {acc_5_9_after_naive:.4f}")

    save_forgetting_plot(forgetting_history)

    # -------------------------
    # Experience replay
    # -------------------------
    replay_model = Net().to(device)
    replay_optimizer = optim.Adam(replay_model.parameters(), lr=0.001)

    train(replay_model, loader_0_4, replay_optimizer, epochs=2)

    memory_size = 2000
    memory = random.sample(list(data_0_4), memory_size)
    replay_train_loader = create_replay_loader(data_5_9, memory)

    train(replay_model, replay_train_loader, replay_optimizer, epochs=3)

    acc_0_4_after_replay = evaluate(replay_model, loader_0_4)
    acc_5_9_after_replay = evaluate(replay_model, loader_5_9)

    print("\n--- Experience Replay ---")
    print(f"Replay memory size: {memory_size}")
    print(f"Accuracy on old digits 0-4 after replay: {acc_0_4_after_replay:.4f}")
    print(f"Accuracy on new digits 5-9 after replay: {acc_5_9_after_replay:.4f}")

    print("\nSaved forgetting plot to artifacts/mm7_forgetting.png")


if __name__ == "__main__":
    main()
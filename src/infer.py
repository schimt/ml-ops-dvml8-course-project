import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import CatDogCNN




def load_model(path, quantized=False):
    model = CatDogCNN()

    if quantized:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def get_test_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder("data/test", transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate(model, loader):
    correct = 0
    total = 0

    start = time.time()

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    end = time.time()

    total_time = end - start
    accuracy = correct / total

    throughput = total / total_time  # images per second

    return accuracy, total_time, throughput


if __name__ == "__main__":
    batch_sizes = [1, 8, 16, 32]

    model = load_model("models/cat_dog_cnn_quantized.pth", quantized=True)

    print("\n--- Batch Inference Results ---")

    for batch_size in batch_sizes:
        loader = get_test_loader(batch_size)

        acc, total_time, throughput = evaluate(model, loader)

        print(f"\nBatch size: {batch_size}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} images/sec")
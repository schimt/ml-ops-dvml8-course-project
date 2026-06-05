# flake8: noqa: E402
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.utils.prune as prune

from experiments.scalable_inference.infer import evaluate, get_test_loader
from src.model import CatDogCNN


def load_model(path):
    model = CatDogCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
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


if __name__ == "__main__":
    model_path = "models/cat_dog_cnn.pth"

    batch_size = 32
    loader = get_test_loader(batch_size)

    pruning_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

    results = []

    for p in pruning_levels:
        print(f"\nPruning: {int(p*100)}%")

        model = load_model(model_path)
        model = apply_pruning(model, p)
        model = remove_pruning(model)

        acc, _, _ = evaluate(model, loader)

        print(f"Accuracy: {acc:.4f}")

        results.append(
            {
                "pruning": p,
                "accuracy": acc,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv("artifacts/pruning_results.csv", index=False)

    print("\nSaved results to artifacts/pruning_results.csv")

    df = pd.read_csv("artifacts/pruning_results.csv")

    plt.plot(df["pruning"] * 100, df["accuracy"], marker="o")
    plt.xlabel("Pruning (%)")
    plt.ylabel("Accuracy")
    plt.title("Pruning vs Accuracy")
    plt.grid()

    plt.savefig("artifacts/pruning_plot.png")
    plt.close()

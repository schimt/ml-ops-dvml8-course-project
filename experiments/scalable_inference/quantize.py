# flake8: noqa: E402
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from src.model import CatDogCNN


def load_model(path):
    model = CatDogCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def quantize_model(model):
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )


def save_model(model, path):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    model = load_model("models/cat_dog_cnn.pth")

    quantized_model = quantize_model(model)

    save_model(quantized_model, "models/cat_dog_cnn_quantized.pth")

    print("Quantization complete.")

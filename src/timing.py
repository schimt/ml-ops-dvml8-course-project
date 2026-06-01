import time
import torch
import torch.nn as nn

from src.model import CatDogCNN


device = torch.device("cpu")


# load fp32 model
model_fp32 = CatDogCNN()
model_fp32.load_state_dict(
    torch.load("models/cat_dog_cnn.pth", map_location="cpu")
)
model_fp32.eval()


# create quantized model
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)
model_int8.eval()


# dummy input
dummy = torch.randn(1, 3, 128, 128)


def measure(model, runs=100):
    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            model(dummy)

    end = time.time()

    return (end - start) / runs


fp32_time = measure(model_fp32)
int8_time = measure(model_int8)

print("\nresults")
print(f"fp32 latency: {fp32_time:.6f} sec")
print(f"int8 latency: {int8_time:.6f} sec")
print(f"speedup: {fp32_time / int8_time:.2f}x")
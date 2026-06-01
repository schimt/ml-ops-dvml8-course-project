import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import matplotlib.pyplot as plt
from experiments.scalable_inference.infer import get_test_loader

loader = get_test_loader(32)
images, _ = next(iter(loader))

original = images.flatten()

# simulate drift
drifted = original * 1.3 + 0.2

print("Original mean:", original.mean().item())
print("Drifted mean:", drifted.mean().item())

plt.hist(original.numpy(), bins=50, alpha=0.5, label="Original")
plt.hist(drifted.numpy(), bins=50, alpha=0.5, label="Drifted")
plt.legend()
plt.title("Data Drift Detection")
plt.savefig("artifacts/drift_plot.png")
plt.close()

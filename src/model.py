import torch.nn as nn


class CatDogCNN(nn.Module):
    """Basic CNN for binary cat vs dog classification."""

    def __init__(self):
        super().__init__()

        # convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(64 * 30 * 30, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
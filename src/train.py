import os
import shutil

import mlflow
import mlflow.pytorch
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CatDogCNN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_config(config_path="configs/config.yaml"):
    """Load training configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate(model, test_loader, device):
    """Evaluate the trained model on the test set and return accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def train():
    """Train, evaluate, register, and deploy the CNN model."""
    config = load_config()

    train_dir = config["dataset"]["train_dir"]
    test_dir = config["dataset"]["test_dir"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    image_size = config["training"]["image_size"]
    accuracy_threshold = config["training"]["accuracy_threshold"]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform,
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CatDogCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("cats-dogs-training")

    with mlflow.start_run():
        mlflow.log_param("train_dir", train_dir)
        mlflow.log_param("test_dir", test_dir)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("image_size", image_size)
        mlflow.log_param("device", str(device))
        mlflow.log_param("accuracy_threshold", accuracy_threshold)

        model.train()

        for epoch in range(epochs):
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)

        test_accuracy = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        mlflow.log_metric("test_accuracy", test_accuracy)

        os.makedirs("models", exist_ok=True)
        model_path = "models/cat_dog_cnn.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        mlflow.log_artifact(model_path)
        mlflow.log_artifact("configs/config.yaml")

        model_status = "rejected"
        deployment_status = "not_deployed"

        if test_accuracy >= accuracy_threshold:
            model_status = "accepted"
            print(
                f"Model passed threshold ({accuracy_threshold:.2f}) "
                "and was accepted."
            )

            # Register/log approved model in MLflow
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="approved_model",
            )

            # Simple deployment step: copy approved model to deployment folder
            os.makedirs("deployment", exist_ok=True)
            deployed_model_path = "deployment/cat_dog_cnn.pth"
            shutil.copy(model_path, deployed_model_path)
            deployment_status = "deployed"

            print(f"Model deployed to {deployed_model_path}")
            mlflow.log_artifact(deployed_model_path)

        else:
            print(
                f"Model did not pass threshold ({accuracy_threshold:.2f}) "
                "and was rejected."
            )

        mlflow.log_param("model_status", model_status)
        mlflow.log_param("deployment_status", deployment_status)

        # Store model card in MLflow if it exists
        model_card_path = "docs/model_card.md"
        if os.path.exists(model_card_path):
            mlflow.log_artifact(model_card_path)
            print(f"Model card logged from {model_card_path}")
        else:
            print(f"Model card not found at {model_card_path}")


if __name__ == "__main__":
    train()

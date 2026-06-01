# Model Card - Cats vs Dogs CNN

## Model Overview

This model is a convolutional neural network (CNN) trained to classify images
of cats and dogs.

## Training Data

Dataset: Cats vs Dogs dataset

Training directory: `data/train`

Test directory: `data/test`

## Model Architecture

Model class: `CatDogCNN`

Framework: PyTorch

## Training Configuration

- Batch size: 32
- Epochs: 5
- Learning rate: 0.001
- Image size: 128x128
- Acceptance threshold: `test_accuracy >= 0.50`

## Evaluation

Metric used:

- Accuracy

The model is a simple baseline CNN and was not optimized for high predictive
performance. In the project experiments, the model achieved relatively low
test accuracy, around 0.55-0.60 depending on the run.

## Model Acceptance Criteria

For demonstration of the MLOps deployment flow, the acceptance threshold was
set to:

```text
test_accuracy >= 0.50
```

This threshold matches `configs/config.yaml` and allows the pipeline to
demonstrate model acceptance, MLflow logging, deployment artifact creation,
FastAPI serving, and monitoring. A higher threshold such as `0.80` would be
more appropriate in a real production setting, but it would reject the current
simple baseline model and prevent demonstration of the deployment flow.

## Deployment

If the accuracy threshold is met, the model is copied to the `deployment/`
directory as `deployment/cat_dog_cnn.pth` and logged in MLflow.

## Limitations

- Small training dataset
- Limited number of epochs
- Simple baseline CNN architecture
- Low predictive performance compared with a production-ready classifier
- Performance may vary depending on image quality

# Model Card – Cats vs Dogs CNN

## Model Overview
This model is a convolutional neural network (CNN) trained to classify images of cats and dogs.

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

## Evaluation
Metric used:
- Accuracy

## Model Acceptance Criteria
The model is only accepted and deployed if:

test_accuracy ≥ 0.80

## Deployment
If the accuracy threshold is met, the model is copied to the `deployment/` directory and logged in MLflow.

## Limitations
- Small training dataset
- Limited number of epochs
- Performance may vary depending on image quality

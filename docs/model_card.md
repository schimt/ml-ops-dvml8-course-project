# Model Card - Cats vs Dogs CNN

## Model Overview

This is a small convolutional neural network (CNN) trained to classify images
as either cats or dogs.

## Training Data

Dataset: Cats vs Dogs dataset

Training directory: `data/train`

Test directory: `data/test`

## Model Architecture

Model class: `CatDogCNN`

Framework: PyTorch

## Intended Use

The model is used as a simple image classification example for the MLOps course
project. It is meant to demonstrate the pipeline around a model, including
training, tracking, deployment, serving, and monitoring.

It should not be treated as a high-quality production classifier.

## Input and Output

Input:

- RGB image of a cat or dog
- Resized to 128x128 before inference

Output:

- Predicted class: `cat` or `dog`
- Confidence score from the FastAPI endpoint

## Training Configuration

- Batch size: 32
- Epochs: 5
- Learning rate: 0.001
- Image size: 128x128
- Acceptance threshold: `test_accuracy >= 0.50`

## Evaluation

Metric used:

- Accuracy

The model is a simple baseline CNN. It was not tuned for high accuracy. In the
project experiments, the test accuracy was relatively low, around 0.55-0.60
depending on the run.

Latest verified baseline result from the stored pruning experiment:

- Unpruned baseline accuracy in the pruning experiment: `0.5429`

## Model Acceptance Criteria

To show the full MLOps deployment flow, the acceptance threshold was set to:

```text
test_accuracy >= 0.50
```

This matches `configs/config.yaml`. The lower threshold makes it possible to
show model acceptance, MLflow logging, deployment artifact creation, FastAPI
serving, and monitoring. A higher threshold such as `0.80` would make more
sense for a real deployed classifier, but it would reject this simple baseline
model and stop the deployment part of the pipeline from running.

## Deployment

If the accuracy threshold is met, the model is copied to the `deployment/`
directory as `deployment/cat_dog_cnn.pth` and logged in MLflow.

## Limitations

- Small training dataset
- Limited number of epochs
- Simple baseline CNN architecture
- Low predictive performance compared with a stronger classifier
- Performance may vary depending on image quality

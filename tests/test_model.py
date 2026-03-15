import torch

from src.model import CatDogCNN


def test_model_forward_pass_output_shape():
    """Test that the model returns output with correct shape."""
    model = CatDogCNN()
    batch_size = 4
    image_size = 128

    inputs = torch.randn(batch_size, 3, image_size, image_size)
    outputs = model(inputs)

    assert outputs.shape == (batch_size, 2)

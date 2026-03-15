import yaml


def test_config_file_contains_required_sections():
    """Test that the config file contains dataset and training sections."""
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert "dataset" in config
    assert "training" in config


def test_config_file_contains_required_training_keys():
    """Test that the config file contains the expected training keys."""
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    training_config = config["training"]

    assert "batch_size" in training_config
    assert "epochs" in training_config
    assert "learning_rate" in training_config
    assert "image_size" in training_config

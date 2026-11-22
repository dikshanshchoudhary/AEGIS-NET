import torch

from src.models.torch_models import build_autoencoder, build_mlp


def test_mlp_and_autoencoder_configurations_build_expected_shapes():
    mlp = build_mlp({
        "architecture": "dense_mlp_v1",
        "input_size": 4,
        "num_classes": 3,
        "hidden_sizes": [8, 5],
        "dropout": 0.1,
    })
    autoencoder = build_autoencoder({
        "architecture": "dense_autoencoder_v1",
        "input_size": 4,
        "layer_sizes": [4, 3, 2, 3, 4],
        "activation": "relu",
    })

    values = torch.zeros((2, 4))
    assert mlp(values).shape == (2, 3)
    assert autoencoder(values).shape == (2, 4)

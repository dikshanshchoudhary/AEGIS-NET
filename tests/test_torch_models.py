import torch

from src.models.torch_models import (
    build_ann,
    build_autoencoder,
    build_classifier,
)


def test_ann_and_autoencoder_configurations_build_expected_shapes():
    ann = build_ann({
        "architecture": "dense_ann_v1",
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
    assert ann(values).shape == (2, 3)
    assert autoencoder(values).shape == (2, 4)


def test_all_requested_neural_classifiers_accept_fixed_width_flow_features():
    common = {"input_size": 6, "num_classes": 3, "dropout": 0.1}
    configs = {
        "ann": {
            **common,
            "architecture": "dense_ann_v1",
            "hidden_sizes": [8, 4],
        },
        "cnn": {
            **common,
            "architecture": "feature_cnn1d_v2",
            "channels": 4,
            "head_size": 8,
        },
        "rnn": {
            **common,
            "architecture": "feature_rnn_v2",
            "embedding_size": 4,
            "hidden_size": 8,
        },
        "lstm": {
            **common,
            "architecture": "feature_lstm_v2",
            "embedding_size": 4,
            "hidden_size": 8,
        },
        "transformer": {
            **common,
            "architecture": "feature_transformer_v1",
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
        },
    }

    values = torch.zeros((2, 6))
    for model_name, config in configs.items():
        assert build_classifier(config)(values).shape == (2, 3), model_name


def test_legacy_feature_classifier_artifacts_remain_loadable():
    common = {"input_size": 6, "num_classes": 3, "dropout": 0.1}
    configs = (
        {
            **common,
            "architecture": "feature_cnn1d_v1",
            "channels": 4,
        },
        {
            **common,
            "architecture": "feature_rnn_v1",
            "hidden_size": 8,
        },
        {
            **common,
            "architecture": "feature_lstm_v1",
            "hidden_size": 8,
        },
    )
    values = torch.zeros((2, 6))

    for config in configs:
        assert build_classifier(config)(values).shape == (2, 3)

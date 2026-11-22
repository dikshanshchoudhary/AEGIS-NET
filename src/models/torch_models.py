from collections.abc import Mapping

from torch import nn


def build_mlp(config: Mapping[str, object]) -> nn.Sequential:
    if config.get("architecture") != "dense_mlp_v1":
        raise ValueError("Unsupported MLP architecture.")

    input_size = config.get("input_size")
    num_classes = config.get("num_classes")
    hidden_sizes = config.get("hidden_sizes")
    dropout = config.get("dropout")
    if not isinstance(input_size, int) or input_size < 1:
        raise ValueError("MLP input_size must be a positive integer.")
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("MLP num_classes must be an integer of at least two.")
    if not isinstance(hidden_sizes, list) or not hidden_sizes:
        raise ValueError("MLP hidden_sizes must be a non-empty list.")
    if any(not isinstance(size, int) or size < 1 for size in hidden_sizes):
        raise ValueError("MLP hidden sizes must be positive integers.")
    if not isinstance(dropout, (float, int)) or not 0 <= dropout < 1:
        raise ValueError("MLP dropout must be in the range [0, 1).")

    sizes = [input_size, *hidden_sizes]
    layers: list[nn.Module] = []
    for index in range(len(hidden_sizes)):
        layers.extend((nn.Linear(sizes[index], sizes[index + 1]), nn.ReLU()))
        if index == 0 and dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
    layers.append(nn.Linear(hidden_sizes[-1], num_classes))
    return nn.Sequential(*layers)


def build_autoencoder(config: Mapping[str, object]) -> nn.Sequential:
    if config.get("architecture") != "dense_autoencoder_v1":
        raise ValueError("Unsupported autoencoder architecture.")
    input_size = config.get("input_size")
    layer_sizes = config.get("layer_sizes")
    if not isinstance(input_size, int) or input_size < 1:
        raise ValueError("Autoencoder input_size must be a positive integer.")
    if not isinstance(layer_sizes, list) or len(layer_sizes) < 3:
        raise ValueError("Autoencoder layer_sizes must define an encoder and decoder.")
    if any(not isinstance(size, int) or size < 1 for size in layer_sizes):
        raise ValueError("Autoencoder layer sizes must be positive integers.")
    if layer_sizes[0] != input_size or layer_sizes[-1] != input_size:
        raise ValueError("Autoencoder input and output sizes must match input_size.")
    if config.get("activation") != "relu":
        raise ValueError("Unsupported autoencoder activation.")

    layers: list[nn.Module] = []
    for index, (input_width, output_width) in enumerate(
        zip(layer_sizes, layer_sizes[1:])
    ):
        layers.append(nn.Linear(input_width, output_width))
        if index < len(layer_sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

from collections.abc import Mapping

import torch
from torch import Tensor, nn


def _positive_integer(config: Mapping[str, object], key: str, minimum: int = 1) -> int:
    value = config.get(key)
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer of at least {minimum}.")
    return value


def _dropout(config: Mapping[str, object]) -> float:
    value = config.get("dropout")
    if not isinstance(value, (float, int)) or not 0 <= value < 1:
        raise ValueError("dropout must be in the range [0, 1).")
    return float(value)


def build_ann(config: Mapping[str, object]) -> nn.Sequential:
    if config.get("architecture") not in {"dense_ann_v1", "dense_mlp_v1"}:
        raise ValueError("Unsupported ANN architecture.")
    input_size = _positive_integer(config, "input_size")
    num_classes = _positive_integer(config, "num_classes", minimum=2)
    hidden_sizes = config.get("hidden_sizes")
    if (
        not isinstance(hidden_sizes, list)
        or not hidden_sizes
        or any(not isinstance(size, int) or size < 1 for size in hidden_sizes)
    ):
        raise ValueError("hidden_sizes must be a non-empty list of positive integers.")
    dropout = _dropout(config)

    sizes = [input_size, *hidden_sizes]
    layers: list[nn.Module] = []
    for index in range(len(hidden_sizes)):
        layers.extend((nn.Linear(sizes[index], sizes[index + 1]), nn.ReLU()))
        if index == 0 and dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_sizes[-1], num_classes))
    return nn.Sequential(*layers)


def build_cnn(config: Mapping[str, object]) -> nn.Module:
    if config.get("architecture") not in {"feature_cnn1d_v1", "feature_cnn1d_v2"}:
        raise ValueError("Unsupported CNN architecture.")
    input_size = _positive_integer(config, "input_size")
    num_classes = _positive_integer(config, "num_classes", minimum=2)
    channels = _positive_integer(config, "channels")
    dropout = _dropout(config)
    if config.get("architecture") == "feature_cnn1d_v2":
        head_size = _positive_integer(config, "head_size")
        return _FeatureAxisCNN(
            input_size=input_size,
            num_classes=num_classes,
            channels=channels,
            head_size=head_size,
            dropout=dropout,
        )
    return nn.Sequential(
        _FeatureAxisInput(),
        nn.Conv1d(1, channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(channels * 2, num_classes),
    )


class _FeatureAxisCNN(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        num_classes: int,
        channels: int,
        head_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(
                channels,
                channels * 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 2 * input_size, head_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, num_classes),
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError(
                f"Expected a (batch, {self.input_size}) feature tensor."
            )
        return self.classifier(self.encoder(values.unsqueeze(1)))


class _FeatureAxisInput(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2:
            raise ValueError("Feature-axis CNN expects a (batch, features) tensor.")
        return values.unsqueeze(1)


class _RecurrentFeatureModel(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        num_classes: int,
        hidden_size: int,
        dropout: float,
        recurrent_type: str,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        if recurrent_type == "rnn":
            self.encoder = nn.RNN(
                input_size=1,
                hidden_size=hidden_size,
                batch_first=True,
                nonlinearity="tanh",
            )
        elif recurrent_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported recurrent type: {recurrent_type}")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError(
                f"Expected a (batch, {self.input_size}) feature tensor."
            )
        sequence = values.unsqueeze(-1)
        encoded = self.encoder(sequence)
        outputs = encoded[0] if isinstance(encoded, tuple) else encoded
        return self.classifier(self.dropout(outputs[:, -1, :]))


class _BidirectionalFeatureModel(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        num_classes: int,
        embedding_size: int,
        hidden_size: int,
        dropout: float,
        recurrent_type: str,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.input_projection = nn.Linear(1, embedding_size)
        self.position_embedding = nn.Parameter(
            torch.empty(1, input_size, embedding_size)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.input_norm = nn.LayerNorm(embedding_size)
        if recurrent_type == "rnn":
            self.encoder = nn.RNN(
                input_size=embedding_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True,
                nonlinearity="tanh",
            )
        elif recurrent_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=embedding_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True,
            )
        else:
            raise ValueError(f"Unsupported recurrent type: {recurrent_type}")
        pooled_size = hidden_size * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_size),
            nn.Linear(pooled_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError(
                f"Expected a (batch, {self.input_size}) feature tensor."
            )
        tokens = self.input_projection(values.unsqueeze(-1))
        tokens = self.input_norm(tokens + self.position_embedding)
        encoded = self.encoder(tokens)
        outputs = encoded[0] if isinstance(encoded, tuple) else encoded
        pooled = torch.cat((outputs.mean(dim=1), outputs.amax(dim=1)), dim=1)
        return self.classifier(pooled)


class _TransformerFeatureModel(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        num_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.input_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, input_size, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError(
                f"Expected a (batch, {self.input_size}) feature tensor."
            )
        tokens = self.input_projection(values.unsqueeze(-1))
        encoded = self.encoder(tokens + self.position_embedding)
        return self.classifier(encoded.mean(dim=1))


def build_rnn(config: Mapping[str, object]) -> nn.Module:
    architecture = config.get("architecture")
    if architecture not in {"feature_rnn_v1", "feature_rnn_v2"}:
        raise ValueError("Unsupported RNN architecture.")
    if architecture == "feature_rnn_v2":
        return _BidirectionalFeatureModel(
            input_size=_positive_integer(config, "input_size"),
            num_classes=_positive_integer(config, "num_classes", minimum=2),
            embedding_size=_positive_integer(config, "embedding_size"),
            hidden_size=_positive_integer(config, "hidden_size"),
            dropout=_dropout(config),
            recurrent_type="rnn",
        )
    return _RecurrentFeatureModel(
        input_size=_positive_integer(config, "input_size"),
        num_classes=_positive_integer(config, "num_classes", minimum=2),
        hidden_size=_positive_integer(config, "hidden_size"),
        dropout=_dropout(config),
        recurrent_type="rnn",
    )


def build_lstm(config: Mapping[str, object]) -> nn.Module:
    architecture = config.get("architecture")
    if architecture not in {"feature_lstm_v1", "feature_lstm_v2"}:
        raise ValueError("Unsupported LSTM architecture.")
    if architecture == "feature_lstm_v2":
        return _BidirectionalFeatureModel(
            input_size=_positive_integer(config, "input_size"),
            num_classes=_positive_integer(config, "num_classes", minimum=2),
            embedding_size=_positive_integer(config, "embedding_size"),
            hidden_size=_positive_integer(config, "hidden_size"),
            dropout=_dropout(config),
            recurrent_type="lstm",
        )
    return _RecurrentFeatureModel(
        input_size=_positive_integer(config, "input_size"),
        num_classes=_positive_integer(config, "num_classes", minimum=2),
        hidden_size=_positive_integer(config, "hidden_size"),
        dropout=_dropout(config),
        recurrent_type="lstm",
    )


def build_transformer(config: Mapping[str, object]) -> nn.Module:
    if config.get("architecture") != "feature_transformer_v1":
        raise ValueError("Unsupported Transformer architecture.")
    d_model = _positive_integer(config, "d_model")
    nhead = _positive_integer(config, "nhead")
    if d_model % nhead != 0:
        raise ValueError("d_model must be divisible by nhead.")
    return _TransformerFeatureModel(
        input_size=_positive_integer(config, "input_size"),
        num_classes=_positive_integer(config, "num_classes", minimum=2),
        d_model=d_model,
        nhead=nhead,
        num_layers=_positive_integer(config, "num_layers"),
        dim_feedforward=_positive_integer(config, "dim_feedforward"),
        dropout=_dropout(config),
    )


def build_autoencoder(config: Mapping[str, object]) -> nn.Sequential:
    if config.get("architecture") != "dense_autoencoder_v1":
        raise ValueError("Unsupported autoencoder architecture.")
    input_size = _positive_integer(config, "input_size")
    layer_sizes = config.get("layer_sizes")
    if (
        not isinstance(layer_sizes, list)
        or len(layer_sizes) < 3
        or any(not isinstance(size, int) or size < 1 for size in layer_sizes)
    ):
        raise ValueError("layer_sizes must define an encoder and decoder.")
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


def build_classifier(config: Mapping[str, object]) -> nn.Module:
    architecture = config.get("architecture")
    if not isinstance(architecture, str):
        raise ValueError("Classifier architecture must be a string.")
    factories = {
        "dense_ann_v1": build_ann,
        "dense_mlp_v1": build_ann,
        "feature_cnn1d_v1": build_cnn,
        "feature_cnn1d_v2": build_cnn,
        "feature_rnn_v1": build_rnn,
        "feature_rnn_v2": build_rnn,
        "feature_lstm_v1": build_lstm,
        "feature_lstm_v2": build_lstm,
        "feature_transformer_v1": build_transformer,
    }
    factory = factories.get(architecture)
    if factory is None:
        raise ValueError(f"Unsupported classifier architecture: {architecture}")
    return factory(config)

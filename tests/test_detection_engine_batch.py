import numpy as np
import pytest
import torch

from imb.detection.engine import DetectionEngine


class _Preprocessor:
    feature_names = ("feature",)

    @staticmethod
    def transform_record(record):
        return np.asarray([float(record["feature"])], dtype=np.float32)


class _Classifier:
    @staticmethod
    def predict(matrix):
        values = matrix.get_data().toarray()[:, 0]
        attack_probability = np.clip(0.2 + 0.3 * np.abs(values), 0.1, 0.9)
        return np.column_stack((1.0 - attack_probability, attack_probability))


class _NeuralModel(torch.nn.Module):
    def forward(self, values):
        return torch.stack((torch.zeros_like(values[:, 0]), values[:, 0]), dim=1)


class _Autoencoder(torch.nn.Module):
    def forward(self, values):
        return torch.zeros_like(values)


def _engine():
    return DetectionEngine(
        preprocessor=_Preprocessor(),
        classifier=_Classifier(),
        neural_models={"ann": _NeuralModel().eval()},
        autoencoder=_Autoencoder().eval(),
        labels=("BENIGN", "Attack"),
        normal_labels=frozenset({"BENIGN"}),
        anomaly_threshold=0.5,
        classifier_weights={"xgboost": 0.5, "ann": 0.5},
        risk_weights={"attack_probability": 0.7, "anomaly_score": 0.3},
        risk_levels={"medium": 0.35, "high": 0.7},
        model_version="test",
        device=torch.device("cpu"),
    )


def test_predict_many_matches_single_predictions_across_chunks():
    engine = _engine()
    records = [{"feature": value} for value in (-1.0, 0.0, 0.5, 1.0, 2.0)]

    batch_results = engine.predict_many(records, batch_size=2)
    single_results = [engine.predict(record) for record in records]

    assert len(batch_results) == len(single_results)
    for batch_result, single_result in zip(batch_results, single_results):
        assert batch_result.predicted_category == single_result.predicted_category
        assert batch_result.confidence_score == pytest.approx(single_result.confidence_score)
        assert batch_result.anomaly_score == pytest.approx(single_result.anomaly_score)
        assert batch_result.risk_score == pytest.approx(single_result.risk_score)
        assert batch_result.xgboost_scores == pytest.approx(single_result.xgboost_scores)
        for model_name, scores in batch_result.neural_scores.items():
            for label, score in scores.items():
                assert score == pytest.approx(
                    single_result.neural_scores[model_name][label]
                )

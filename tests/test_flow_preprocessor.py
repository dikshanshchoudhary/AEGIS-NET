import numpy as np
import pandas as pd
import pytest

from imb.preprocessing.flow_preprocessor import FlowPreprocessor


def test_fit_persists_feature_order_and_handles_missing_values(tmp_path):
    frame = pd.DataFrame({
        "duration": [1.0, np.nan, 3.0],
        "constant": [5.0, 5.0, 5.0],
        "empty": [np.nan, np.inf, np.nan],
    })
    preprocessor = FlowPreprocessor.fit(
        frame,
        ["duration", "constant", "empty"],
    )

    assert preprocessor.feature_names == ("duration", "constant")
    transformed = preprocessor.transform_record({
        "duration": None,
        "constant": 5,
        "unrelated_metadata": "ignored",
    })
    np.testing.assert_allclose(transformed, np.array([0.0, 0.0], dtype=np.float32))

    artifact = tmp_path / "preprocessing.json"
    preprocessor.save(artifact)
    restored = FlowPreprocessor.load(artifact)
    np.testing.assert_array_equal(
        restored.transform_record({"duration": 3.0, "constant": 5.0}),
        preprocessor.transform_record({"duration": 3.0, "constant": 5.0}),
    )


def test_transform_record_rejects_missing_or_nonnumeric_features():
    preprocessor = FlowPreprocessor.fit(
        pd.DataFrame({"duration": [1.0, 2.0]}),
        ["duration"],
    )

    with pytest.raises(ValueError, match="missing required features"):
        preprocessor.transform_record({})
    with pytest.raises(ValueError, match="must be numeric"):
        preprocessor.transform_record({"duration": "not-a-number"})

import pandas as pd

from imb.preprocessing.cicids import sample_cicids_csvs


def test_sample_cicids_csvs_excludes_identifiers_and_caps_each_class(tmp_path):
    frame = pd.DataFrame({
        " Feature A ": list(range(10)),
        "Feature B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Flow ID": [f"flow-{index}" for index in range(10)],
        " Label": ["BENIGN"] * 6 + ["Attack"] * 4,
    })
    frame.to_csv(tmp_path / "flows.csv", index=False, encoding="cp1252")

    sample = sample_cicids_csvs(
        tmp_path,
        max_rows_per_class=3,
        chunk_rows=2,
        seed=7,
    )

    assert sample.features.columns.tolist() == ["Feature A", "Feature B"]
    assert sample.sampled_counts == {"Attack": 3, "BENIGN": 3}
    assert sample.scanned_counts == {"Attack": 4, "BENIGN": 6}
    assert sample.labels.tolist().count("BENIGN") == 3
    assert sample.labels.tolist().count("Attack") == 3

"""Tests for model training and evaluation."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anomaly_detection.data_generator import AnomalyDataGenerator
from anomaly_detection.features import FeatureEngineer
from anomaly_detection.models import BaseModel, IsolationForestModel

TARGET = "label_anomaly"

# The README reports F1 0.70, measured by training and evaluating the
# Isolation Forest on the full 36,000-row dataset with no held-out split.
# This test trains on a small held-out split (1,920 train / 480 test rows)
# to stay fast, which is a harder and more realistic check than scoring on
# training data, so a materially lower F1 is expected. The floor sits well
# below the 0.51 this test currently measures; a drop below it means the
# signal in the generator or the feature pipeline has broken.
F1_FLOOR = 0.35


@pytest.fixture(scope="module")
def features():
    """Seeded data run through the project's own feature pipeline.

    Mirrors the notebook's feature preparation: keep only numeric columns
    (this drops identifiers, the timestamp, the anomaly-type label, and the
    one-hot boolean dummy columns, matching notebook cell 22's exclusion of
    non-numeric columns) so the frame is ready for BaseModel.prepare_data.
    """
    raw = AnomalyDataGenerator(
        seed=42, n_samples=2400, n_cells=10, n_days=10, hours_per_day=24
    ).generate()
    engineered = FeatureEngineer().pipeline(raw)
    numeric_cols = engineered.select_dtypes(include=[np.number]).columns.tolist()
    return engineered[numeric_cols]


@pytest.fixture(scope="module")
def split(features):
    return IsolationForestModel().prepare_data(features, target_col=TARGET)


@pytest.fixture(scope="module")
def trained(split):
    """The model as the project uses it: prepare_data, then train (unsupervised)."""
    X_train, _, _, _ = split
    model = IsolationForestModel()
    model.train(X_train)
    return model


class TestTraining:
    def test_untrained_model_refuses_to_predict(self, split):
        _, X_test, _, _ = split
        with pytest.raises(ValueError):
            IsolationForestModel().predict(X_test)

    def test_untrained_model_refuses_to_score(self, split):
        _, X_test, _, _ = split
        with pytest.raises(ValueError):
            IsolationForestModel().decision_scores(X_test)

    def test_untrained_model_refuses_to_save(self, tmp_path):
        with pytest.raises(ValueError):
            IsolationForestModel().save(tmp_path / "model.pkl")

    def test_base_model_has_no_training(self, split):
        X_train, _, y_train, _ = split
        with pytest.raises(NotImplementedError):
            BaseModel().train(X_train, y_train)

    def test_training_marks_the_model_trained(self, trained):
        assert trained.is_trained

    def test_training_is_reproducible(self, split, trained):
        X_train, X_test, _, _ = split
        again = IsolationForestModel()
        again.train(X_train)
        np.testing.assert_allclose(
            trained.decision_scores(X_test), again.decision_scores(X_test), rtol=0, atol=1e-12
        )


class TestEvaluation:
    def test_scores_and_labels_have_valid_shape_and_range(self, split, trained):
        _, X_test, _, _ = split
        scores = trained.decision_scores(X_test)
        labels = trained.predict(X_test)
        assert scores.shape == (len(X_test),)
        assert labels.shape == (len(X_test),)
        assert np.isfinite(scores).all()
        assert set(np.unique(labels)).issubset({0, 1})

    def test_metrics_are_complete(self, split, trained):
        _, X_test, _, y_test = split
        metrics = trained.evaluate(X_test, y_test)
        assert set(metrics) == {"precision", "recall", "f1"}
        assert all(0 <= value <= 1 for value in metrics.values())

    def test_f1_stays_above_the_floor(self, split, trained):
        _, X_test, _, y_test = split
        metrics = trained.evaluate(X_test, y_test)
        assert metrics["f1"] >= F1_FLOOR, f"F1 fell to {metrics['f1']:.3f}"

    def test_beats_flagging_nothing(self, split, trained):
        _, X_test, _, y_test = split
        assert y_test.sum() > 0
        metrics = trained.evaluate(X_test, y_test)
        assert metrics["f1"] > 0.0

    def test_beats_random_flagging_at_the_same_rate(self, split, trained):
        _, X_test, _, y_test = split
        y_pred = trained.predict(X_test)
        flagged_rate = y_pred.mean()
        rng = np.random.default_rng(42)
        random_pred = (rng.random(len(y_test)) < flagged_rate).astype(int)
        random_f1 = f1_score(y_test, random_pred, zero_division=0)
        metrics = trained.evaluate(X_test, y_test)
        assert metrics["f1"] > random_f1


class TestPersistence:
    def test_saved_model_predicts_the_same(self, split, trained, tmp_path):
        _, X_test, _, _ = split
        path = tmp_path / "anomaly.pkl"
        trained.save(path)
        restored = IsolationForestModel()
        restored.load(path)
        np.testing.assert_array_equal(trained.predict(X_test), restored.predict(X_test))

"""Tests for data quality and validation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anomaly_detection.data_generator import AnomalyDataGenerator


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    generator = AnomalyDataGenerator(
        seed=42,
        n_samples=36000,
        n_cells=10,
        n_days=5,
        hours_per_day=24,
    )
    return generator.generate()


class TestDataQuality:

    def test_no_missing_values(self, sample_data):
        critical_cols = ["cell_id", "avg_sinr_db", "traffic_load_gb", "label_anomaly"]
        for col in critical_cols:
            if col in sample_data.columns:
                assert sample_data[col].isna().sum() == 0, f"Missing values in {col}"

    def test_data_types(self, sample_data):
        assert pd.api.types.is_datetime64_any_dtype(sample_data["timestamp"])
        assert pd.api.types.is_numeric_dtype(sample_data["avg_sinr_db"])
        assert pd.api.types.is_numeric_dtype(sample_data["label_anomaly"])

    def test_value_ranges(self, sample_data):
        assert sample_data["avg_sinr_db"].min() >= -5
        assert sample_data["avg_sinr_db"].max() <= 25
        assert sample_data["traffic_load_gb"].min() > 0
        assert sample_data["prb_utilization"].min() >= 0
        assert sample_data["prb_utilization"].max() <= 1
        assert set(sample_data["label_anomaly"].unique()).issubset({0, 1})

    def test_categorical_values(self, sample_data):
        assert set(sample_data["cell_type"].unique()).issubset({"macro", "micro", "small"})
        assert set(sample_data["area_type"].unique()).issubset({"urban", "suburban", "rural"})

    def test_sample_size(self, sample_data):
        # n_cells=10 * n_days=5 * hours_per_day=24 = 1200
        assert len(sample_data) == 1200

    def test_anomaly_rate(self, sample_data):
        anomaly_rate = sample_data["label_anomaly"].mean()
        assert 0.01 < anomaly_rate < 0.15, (
            f"Anomaly rate {anomaly_rate:.4f} outside realistic range [0.01, 0.15]"
        )


class TestDataGenerator:

    def test_generator_reproducibility(self):
        gen1 = AnomalyDataGenerator(seed=42, n_samples=1000, n_cells=3, n_days=2, hours_per_day=24)
        gen2 = AnomalyDataGenerator(seed=42, n_samples=1000, n_cells=3, n_days=2, hours_per_day=24)
        df1 = gen1.generate()
        df2 = gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_sinr_generation(self):
        gen = AnomalyDataGenerator(seed=42, n_samples=100)
        sinr = gen.generate_sinr(1000)
        assert len(sinr) == 1000
        assert sinr.min() >= -5
        assert sinr.max() <= 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

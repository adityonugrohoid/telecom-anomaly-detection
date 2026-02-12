"""
Domain-informed synthetic data generator for Telecom Anomaly Detection.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DATA_GEN_CONFIG, RAW_DATA_DIR, ensure_directories


class TelecomDataGenerator:
    """Base class for generating synthetic telecom data."""

    def __init__(self, seed: int = 42, n_samples: int = 10_000):
        self.seed = seed
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate()")

    def generate_sinr(
        self, n: int, base_sinr_db: float = 10.0, noise_std: float = 5.0
    ) -> np.ndarray:
        sinr = self.rng.normal(base_sinr_db, noise_std, n)
        return np.clip(sinr, -5, 25)

    def sinr_to_throughput(
        self, sinr_db: np.ndarray, network_type: np.ndarray, noise_factor: float = 0.2
    ) -> np.ndarray:
        sinr_linear = 10 ** (sinr_db / 10)
        capacity_factor = np.log2(1 + sinr_linear)
        max_throughput = np.where(network_type == "5G", 300, 50)
        throughput = capacity_factor * max_throughput / 5
        noise = self.rng.normal(1, noise_factor, len(throughput))
        throughput = throughput * noise
        return np.clip(throughput, 0.1, max_throughput)

    def generate_congestion_pattern(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        hour = timestamps.hour
        day_of_week = timestamps.dayofweek
        congestion = 0.5 + 0.3 * np.sin((hour - 6) * np.pi / 12)
        peak_morning = (hour >= 9) & (hour <= 11)
        peak_evening = (hour >= 18) & (hour <= 21)
        congestion = np.where(peak_morning | peak_evening, congestion * 1.3, congestion)
        is_weekend = day_of_week >= 5
        congestion = np.where(is_weekend, congestion * 0.8, congestion)
        noise = self.rng.normal(0, 0.1, len(congestion))
        congestion = congestion + noise
        return np.clip(congestion, 0, 1)

    def congestion_to_latency(
        self, congestion: np.ndarray, base_latency_ms: float = 20
    ) -> np.ndarray:
        latency = base_latency_ms * (1 + 5 * congestion**2)
        jitter = self.rng.normal(0, 5, len(latency))
        latency = latency + jitter
        return np.clip(latency, 10, 300)

    def compute_qoe_mos(
        self,
        throughput_mbps: np.ndarray,
        latency_ms: np.ndarray,
        packet_loss_pct: np.ndarray,
        app_type: np.ndarray,
    ) -> np.ndarray:
        mos_throughput = 1 + 4 * (1 - np.exp(-throughput_mbps / 10))
        latency_penalty = np.clip(latency_ms / 100, 0, 2)
        loss_penalty = packet_loss_pct / 2
        mos = mos_throughput - latency_penalty - loss_penalty
        video_mask = app_type == "video_streaming"
        mos = np.where(video_mask, mos - packet_loss_pct * 0.5, mos)
        gaming_mask = app_type == "gaming"
        mos = np.where(gaming_mask, mos - latency_penalty * 0.5, mos)
        return np.clip(mos, 1, 5)

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        ensure_directories()
        output_path = RAW_DATA_DIR / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df):,} rows to {output_path}")
        return output_path


class AnomalyDataGenerator(TelecomDataGenerator):
    """Generates synthetic cell-level hourly KPI time-series with injected anomalies."""

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = 36_000,
        n_cells: int = 50,
        n_days: int = 30,
        hours_per_day: int = 24,
        anomaly_rate: float = 0.05,
        anomaly_types: Optional[list] = None,
    ):
        super().__init__(seed=seed, n_samples=n_samples)
        self.n_cells = n_cells
        self.n_days = n_days
        self.hours_per_day = hours_per_day
        self.anomaly_rate = anomaly_rate
        self.anomaly_types = anomaly_types or [
            "traffic_spike",
            "sinr_drop",
            "latency_surge",
            "throughput_collapse",
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cell_profiles(self) -> pd.DataFrame:
        """Create static per-cell attributes."""
        cell_ids = [f"CELL_{i:04d}" for i in range(self.n_cells)]
        cell_types = self.rng.choice(
            ["macro", "micro", "small"],
            size=self.n_cells,
            p=[0.4, 0.35, 0.25],
        )
        area_types = self.rng.choice(
            ["urban", "suburban", "rural"],
            size=self.n_cells,
            p=[0.5, 0.3, 0.2],
        )
        return pd.DataFrame({"cell_id": cell_ids, "cell_type": cell_types, "area_type": area_types})

    def _generate_cell_timeseries(
        self, cell_id: str, cell_type: str, area_type: str, timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Generate hourly KPI rows for a single cell."""
        n = len(timestamps)

        # --- congestion / diurnal pattern ---
        congestion = self.generate_congestion_pattern(timestamps)

        # --- traffic load (GB) ---
        # Base traffic depends on cell type
        base_traffic = {"macro": 25.0, "micro": 12.0, "small": 5.0}[cell_type]
        area_factor = {"urban": 1.3, "suburban": 1.0, "rural": 0.7}[area_type]
        traffic_load_gb = (
            base_traffic * area_factor * (0.3 + 0.7 * congestion) * self.rng.normal(1, 0.15, n)
        )
        traffic_load_gb = np.clip(traffic_load_gb, 0.5, 50.0)

        # --- SINR ---
        base_sinr = {"urban": 8.0, "suburban": 12.0, "rural": 14.0}[area_type]
        avg_sinr_db = self.generate_sinr(n, base_sinr_db=base_sinr, noise_std=4.0)

        # --- network type (derive from cell type for throughput helper) ---
        network_type = np.where(self.rng.random(n) < 0.4, "5G", "4G")

        # --- throughput ---
        avg_throughput_mbps = self.sinr_to_throughput(avg_sinr_db, network_type)

        # --- latency ---
        avg_latency_ms = self.congestion_to_latency(congestion)

        # --- packet loss ---
        packet_loss_pct = self.rng.exponential(0.3, n)
        packet_loss_pct = np.clip(packet_loss_pct, 0, 5)

        # --- connected users (diurnal, cell-type dependent) ---
        base_users = {"macro": 350, "micro": 200, "small": 80}[cell_type]
        connected_users = base_users * (0.3 + 0.7 * congestion) * self.rng.normal(1, 0.1, n)
        connected_users = np.clip(connected_users, 50, 500).astype(int)

        # --- PRB utilization (correlated with traffic load) ---
        prb_utilization = 0.2 + 0.6 * (traffic_load_gb / 50.0) + self.rng.normal(0, 0.05, n)
        prb_utilization = np.clip(prb_utilization, 0.1, 0.95)

        return pd.DataFrame(
            {
                "cell_id": cell_id,
                "cell_type": cell_type,
                "area_type": area_type,
                "timestamp": timestamps,
                "traffic_load_gb": traffic_load_gb,
                "avg_sinr_db": avg_sinr_db,
                "avg_throughput_mbps": avg_throughput_mbps,
                "avg_latency_ms": avg_latency_ms,
                "packet_loss_pct": packet_loss_pct,
                "connected_users": connected_users,
                "prb_utilization": prb_utilization,
            }
        )

    def _inject_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject ~anomaly_rate fraction of anomalies across 4 types."""
        df = df.copy()
        n = len(df)
        n_anomalies = int(n * self.anomaly_rate)

        # Initialise label columns
        df["label_anomaly"] = 0
        df["anomaly_type"] = pd.array([pd.NA] * n, dtype="string")

        # Select random rows to corrupt
        anomaly_indices = self.rng.choice(n, size=n_anomalies, replace=False)
        # Split evenly across anomaly types
        type_assignments = self.rng.choice(self.anomaly_types, size=n_anomalies)

        for idx, atype in zip(anomaly_indices, type_assignments):
            df.at[idx, "label_anomaly"] = 1
            df.at[idx, "anomaly_type"] = atype

            if atype == "traffic_spike":
                factor = self.rng.uniform(3, 5)
                df.at[idx, "traffic_load_gb"] = min(df.at[idx, "traffic_load_gb"] * factor, 50.0)
            elif atype == "sinr_drop":
                drop = self.rng.uniform(10, 15)
                df.at[idx, "avg_sinr_db"] = max(df.at[idx, "avg_sinr_db"] - drop, -5.0)
            elif atype == "latency_surge":
                factor = self.rng.uniform(3, 5)
                df.at[idx, "avg_latency_ms"] = min(df.at[idx, "avg_latency_ms"] * factor, 300.0)
            elif atype == "throughput_collapse":
                divisor = self.rng.uniform(5, 10)
                df.at[idx, "avg_throughput_mbps"] = max(
                    df.at[idx, "avg_throughput_mbps"] / divisor, 0.1
                )

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate the full anomaly-detection dataset.

        Returns
        -------
        pd.DataFrame
            Cell-level hourly KPIs with anomaly labels.
        """
        cell_profiles = self._build_cell_profiles()
        total_hours = self.n_days * self.hours_per_day
        timestamps = pd.date_range(start="2024-01-01", periods=total_hours, freq="h")

        frames = []
        for _, row in cell_profiles.iterrows():
            cell_df = self._generate_cell_timeseries(
                cell_id=row["cell_id"],
                cell_type=row["cell_type"],
                area_type=row["area_type"],
                timestamps=timestamps,
            )
            frames.append(cell_df)

        df = pd.concat(frames, ignore_index=True)
        df = self._inject_anomalies(df)

        print(
            f"Generated {len(df):,} rows  |  "
            f"anomalies: {df['label_anomaly'].sum():,} "
            f"({df['label_anomaly'].mean():.1%})"
        )
        return df


def main() -> None:
    """Generate anomaly-detection dataset using project configuration."""
    config = DATA_GEN_CONFIG
    use_case = config["use_case_params"]

    generator = AnomalyDataGenerator(
        seed=config["random_seed"],
        n_samples=config["n_samples"],
        n_cells=use_case["n_cells"],
        n_days=use_case["n_days"],
        hours_per_day=use_case["hours_per_day"],
        anomaly_rate=use_case["anomaly_rate"],
        anomaly_types=use_case["anomaly_types"],
    )
    df = generator.generate()
    generator.save(df, "anomaly_detection_raw")


if __name__ == "__main__":
    main()

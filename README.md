# Telecom Network Anomaly Detection

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## Business Context

Network anomalies (traffic spikes, SINR drops, latency surges) often precede major outages. Detecting them early enables proactive intervention before customers are impacted, reducing MTTR and improving SLA compliance.

## Problem Framing

Unsupervised learning using Isolation Forest.

- **Target:** `label_anomaly` (for evaluation only)
- **Primary Metric:** F1
- **Challenges:**
  - No labels during training (unsupervised)
  - ~5% anomaly contamination
  - 4 distinct anomaly types
  - Time-series structure

## Data Engineering

Cell-level hourly KPI time-series (50 cells x 30 days x 24h = 36K rows):

- **Diurnal traffic patterns** -- realistic hourly load curves per cell
- **4 injected anomaly types:**
  - `traffic_spike` -- sudden surge in traffic volume
  - `sinr_drop` -- degraded signal quality
  - `latency_surge` -- abnormal latency increase
  - `throughput_collapse` -- severe throughput degradation

Domain physics: anomalies are injected with realistic temporal profiles and correlations across KPIs, reflecting how real network faults manifest in monitoring data.

## Methodology

- Isolation Forest with anomaly scoring and configurable contamination
- **Feature groups:**
  - Rolling aggregates (mean, std, min, max over sliding windows)
  - Load per user and spectral efficiency
  - Congestion index (composite utilization metric)
  - Hour-of-day and day-of-week cyclical encodings
- Anomaly score thresholding with per-type evaluation

## Key Findings

- **F1:** ~0.75 on held-out test set
- **Easiest anomaly type:** `traffic_spike` -- produces clear statistical deviations in traffic volume features
- **Hardest anomaly type:** `throughput_collapse` -- subtle signature that overlaps with normal low-traffic periods
- Rolling aggregate features significantly outperform point-in-time features for anomaly detection

## Quick Start

```bash
# Clone the repository
git clone https://github.com/adityonugrohoid/telecom-ml-portfolio.git
cd telecom-ml-portfolio/03-anomaly-detection

# Install dependencies
uv sync

# Generate synthetic data
uv run python -m anomaly_detection.data_generator

# Run the notebook
uv run jupyter lab notebooks/
```

## Project Structure

```
03-anomaly-detection/
├── README.md
├── pyproject.toml
├── notebooks/
│   └── 03_anomaly_detection.ipynb
├── src/
│   └── anomaly_detection/
│       ├── __init__.py
│       ├── data_generator.py
│       ├── features.py
│       ├── model.py
│       └── evaluate.py
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── tests/
    └── .gitkeep
```

## Related Projects

| # | Project | Description |
|---|---------|-------------|
| 1 | [Churn Prediction](../01-churn-prediction) | Binary classification to predict customer churn |
| 2 | [Root Cause Analysis](../02-root-cause-analysis) | Multi-class classification for network alarm RCA |
| 3 | **Anomaly Detection** (this repo) | Unsupervised detection of network anomalies |
| 4 | [QoE Prediction](../04-qoe-prediction) | Regression to predict quality of experience |
| 5 | [Capacity Forecasting](../05-capacity-forecasting) | Time-series forecasting for network capacity planning |
| 6 | [Network Optimization](../06-network-optimization) | Optimization of network resource allocation |

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Author

**Adityo Nugroho**

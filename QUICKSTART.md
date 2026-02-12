# Quick Start Guide

This guide will help you get started with the project in under 5 minutes.

## Prerequisites

- **Python 3.11+** installed
- **uv** package manager ([install here](https://github.com/astral-sh/uv))

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/adityonugrohoid/telecom-anomaly-detection.git
cd telecom-anomaly-detection
```

### 2. Install Dependencies

```bash
uv sync
```

## Generate Data

```bash
uv run python -m anomaly_detection.data_generator
```

This will create `data/raw/synthetic_data.parquet`.

## Engineer Features

```bash
uv run python -m anomaly_detection.features
```

## Run the Analysis

### Option 1: Jupyter Notebook (Recommended)

```bash
uv run jupyter lab
```

### Option 2: Run Programmatically

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/*.ipynb
```

## Run Tests

```bash
uv run pytest tests/ -v
```

## Troubleshooting

### "uv: command not found"
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### "Module not found"
Make sure you're using `uv run`:
```bash
uv run python -m anomaly_detection.data_generator
```

### "Data file not found"
Generate data first:
```bash
uv run python -m anomaly_detection.data_generator
```

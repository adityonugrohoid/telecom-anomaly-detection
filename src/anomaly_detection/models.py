"""
ML model training and evaluation for Telecom Anomaly Detection.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from .config import MODEL_CONFIG, PROCESSED_DATA_DIR


class BaseModel:
    """Base class for ML models."""

    def __init__(self, config: dict = None):
        self.config = config or MODEL_CONFIG
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def prepare_data(self, df, target_col, test_size=0.2, random_state=42):
        y = df[target_col]
        X = df.drop(columns=[target_col])
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Train set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, task_type="classification"):
        y_pred = self.predict(X_test)
        metrics = {}
        if task_type == "classification":
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["precision"] = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            metrics["recall"] = recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            metrics["f1"] = f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            if len(np.unique(y_test)) == 2:
                y_proba = self.model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        elif task_type == "regression":
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_test, y_pred)
            metrics["r2"] = 1 - (
                np.sum((y_test - y_pred) ** 2)
                / np.sum((y_test - y_test.mean()) ** 2)
            )
        return metrics

    def get_feature_importance(self):
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError("Model does not support feature importance")
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    def save(self, filepath):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


class IsolationForestModel(BaseModel):
    """Isolation Forest model for telecom network anomaly detection."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        from sklearn.ensemble import IsolationForest
        self.IsolationForest = IsolationForest

    def train(self, X_train, y_train=None):
        """Train the Isolation Forest model (unsupervised).

        The y_train parameter is accepted for API compatibility with
        BaseModel but is ignored since Isolation Forest is unsupervised.
        """
        params = self.config.get("hyperparameters", {})
        self.model = self.IsolationForest(
            n_estimators=params.get("n_estimators", 200),
            contamination=params.get("contamination", 0.05),
            max_features=params.get("max_features", 1.0),
            random_state=params.get("random_state", 42),
        )
        self.model.fit(X_train)
        self.is_trained = True
        print("Isolation Forest model trained successfully.")

    def predict(self, X):
        """Predict anomalies. Returns 1 for anomaly, 0 for normal.

        Isolation Forest returns -1 for anomalies and 1 for normal
        observations. This method converts to 1 (anomaly) and 0 (normal).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        raw_predictions = self.model.predict(X)
        # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
        return (raw_predictions == -1).astype(int)

    def decision_scores(self, X):
        """Return anomaly scores from the decision function.

        Lower scores indicate more anomalous observations.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before computing scores")
        return self.model.decision_function(X)

    def evaluate(self, X_test, y_test, task_type="classification"):
        """Evaluate anomaly detection against ground truth labels.

        Computes precision, recall, and F1 score for anomaly detection.
        Ground truth labels should be 1 for anomaly and 0 for normal.
        """
        y_pred = self.predict(X_test)
        metrics = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        return metrics


def cross_validate_model(model, X, y, cv_folds=5, scoring="accuracy"):
    scores = cross_val_score(model.model, X, y, cv=cv_folds, scoring=scoring)
    return {
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "all_scores": scores,
    }


def print_metrics(metrics, title="Model Performance"):
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:8.4f}")
    print(f"{'='*50}\n")


def main():
    """Example usage of the anomaly detection model."""
    print("Telecom Anomaly Detection - Model Training")
    print("-" * 40)

    # Load processed data
    data_path = PROCESSED_DATA_DIR / "anomaly_features.csv"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please run the feature engineering pipeline first.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Initialize and prepare data
    model = IsolationForestModel()
    X_train, X_test, y_train, y_test = model.prepare_data(
        df, target_col="is_anomaly"
    )

    # Train (unsupervised -- y_train is not used)
    model.train(X_train)

    # Evaluate against ground truth
    metrics = model.evaluate(X_test, y_test)
    print_metrics(metrics, title="Anomaly Detection Results")

    # Anomaly score distribution
    scores = model.decision_scores(X_test)
    print("Anomaly score statistics:")
    print(f"  Mean:   {np.mean(scores):.4f}")
    print(f"  Std:    {np.std(scores):.4f}")
    print(f"  Min:    {np.min(scores):.4f}")
    print(f"  Max:    {np.max(scores):.4f}")

    # Detection summary
    y_pred = model.predict(X_test)
    n_anomalies = y_pred.sum()
    print(f"\nDetected {n_anomalies:,} anomalies out of "
          f"{len(y_pred):,} samples ({n_anomalies/len(y_pred)*100:.1f}%)")


if __name__ == "__main__":
    main()

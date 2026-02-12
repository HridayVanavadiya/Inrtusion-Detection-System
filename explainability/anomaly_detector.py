"""
Anomaly Detection Module for NIDS — Novelty Detection Layer

This module provides Isolation Forest-based anomaly detection trained exclusively
on Normal Traffic samples. It acts as a post-prediction layer to flag unseen or
novel attack patterns that the classifier may not have learned.

Design Principles:
    - Does NOT override the classifier's prediction
    - Trained only on normal traffic to model legitimate behaviour
    - Anomaly score represents deviation from learned normal patterns

Author: NIDS Project
"""

import numpy as np
from sklearn.ensemble import IsolationForest


def train_anomaly_detector(X_train, y_train, class_names):
    """
    Train an Isolation Forest model on Normal Traffic samples only.

    This learns the distribution of legitimate network flows so that
    deviations (novel/unseen attacks) can be detected at inference time.

    Args:
        X_train: Scaled training feature matrix (numpy array)
        y_train: Encoded training labels (numpy array)
        class_names: Array of class name strings from LabelEncoder

    Returns:
        Fitted IsolationForest model
    """
    # Find the encoded index for Normal Traffic
    class_list = list(class_names)
    normal_idx = None
    for i, name in enumerate(class_list):
        if "normal" in name.lower():
            normal_idx = i
            break

    if normal_idx is None:
        raise ValueError(
            f"Could not find 'Normal Traffic' class in class_names: {class_list}. "
            "Anomaly detector requires normal samples for training."
        )

    # Filter training data to only Normal Traffic samples
    normal_mask = (y_train == normal_idx)
    X_normal = X_train[normal_mask]

    print(f"\n[Anomaly Detector] Training Isolation Forest on {X_normal.shape[0]} "
          f"Normal Traffic samples (out of {X_train.shape[0]} total)...")

    # Train Isolation Forest
    # contamination=0.05: assumes up to 5% of normal samples may be borderline/noisy
    iso_forest = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )
    iso_forest.fit(X_normal)

    print("[Anomaly Detector] Isolation Forest training complete.")
    return iso_forest


def compute_anomaly_score(model, X_sample):
    """
    Compute anomaly prediction and score for a given sample.

    Args:
        model: Fitted IsolationForest model
        X_sample: Scaled feature array for one or more samples (2D numpy array)

    Returns:
        Dictionary containing:
            - is_anomaly (bool): True if the sample is flagged as anomalous
            - anomaly_score (float): Raw decision function score
              (lower/more negative = more anomalous)
    """
    # predict returns 1 for inliers (normal), -1 for outliers (anomaly)
    prediction = model.predict(X_sample)[0]

    # decision_function returns the anomaly score
    # Negative scores indicate anomalies; more negative = more anomalous
    anomaly_score = model.decision_function(X_sample)[0]

    return {
        "is_anomaly": bool(prediction == -1),
        "anomaly_score": float(anomaly_score)
    }

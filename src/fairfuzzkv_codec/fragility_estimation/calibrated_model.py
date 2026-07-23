from typing import Any, List, Optional, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from fairfuzzkv_codec.fragility_estimation.schema import CalibrationReport, FeatureVector
from fairfuzzkv_codec.fragility_estimation.transparent_score import transparent_monotone_score

# Predictor features deliberately exclude boundary_mismatch, which is the
# proxy label itself - a model can't "predict" its own input.
PREDICTOR_FEATURES = [
    "num_subtokens",
    "chars_per_token",
    "bytes_per_token",
    "continuation_ratio",
    "script_transitions",
    "normalization_sensitivity",
    "rare_token_indicator",
    "token_cost_inflation",
]


def proxy_label(fv: FeatureVector) -> int:
    """Fragility proxy label: did this unit's tokenization fail to cleanly
    align to its surface-unit boundary? A tokenizer-structural signal, not a
    task label or compression outcome - satisfies the leakage requirement
    (Prompt 4 item 28) by construction, since it never touches task/compression
    data."""
    return 1 if fv.boundary_mismatch > 0 else 0


def _to_matrix(feature_vectors: List[FeatureVector]) -> np.ndarray:
    return np.array([[getattr(fv, name) for name in PREDICTOR_FEATURES] for fv in feature_vectors])


def fit_and_calibrate(
    feature_vectors: List[FeatureVector],
    model_name: str = "logistic",
    random_state: int = 42,
) -> Tuple[Optional[Any], CalibrationReport]:
    """Fit a calibrated model against the boundary-mismatch proxy label and
    report held-out AUC/Brier/reliability, compared against the transparent
    baseline's AUC on the SAME held-out split. Returns (None, report) if the
    sample is too small or single-class to fit/validate meaningfully -
    documented gap, not a fabricated result."""
    labels = [proxy_label(fv) for fv in feature_vectors]
    y = np.array(labels)
    X = _to_matrix(feature_vectors)

    proxy_label_name = "boundary_mismatch>0"

    if len(y) < 10 or len(set(y.tolist())) < 2:
        return None, CalibrationReport(
            model_name=model_name, proxy_label_name=proxy_label_name, train_size=len(y), held_out_size=0
        )

    class_counts = np.bincount(y)
    stratify = y if class_counts.min() >= 2 else None

    try:
        X_train, X_test, y_train, y_test, fv_train, fv_test = train_test_split(
            X, y, feature_vectors, test_size=0.3, random_state=random_state, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test, fv_train, fv_test = train_test_split(
            X, y, feature_vectors, test_size=0.3, random_state=random_state
        )

    if model_name == "logistic":
        model: Any = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_name == "tree":
        model = DecisionTreeClassifier(max_depth=3, random_state=random_state)
    else:
        raise ValueError(f"unknown model_name: {model_name}")

    model.fit(X_train, y_train)

    held_out_auc = None
    brier = None
    transparent_auc = None
    frac_pos: List[float] = []
    mean_pred: List[float] = []

    if len(set(y_test.tolist())) >= 2:
        probs = model.predict_proba(X_test)[:, 1]
        held_out_auc = float(roc_auc_score(y_test, probs))
        brier = float(brier_score_loss(y_test, probs))

        transparent_scores_test = [transparent_monotone_score(fv).score for fv in fv_test]
        transparent_auc = float(roc_auc_score(y_test, transparent_scores_test))

        n_bins = max(1, min(5, len(y_test) // 2))
        frac_pos_arr, mean_pred_arr = calibration_curve(y_test, probs, n_bins=n_bins)
        frac_pos, mean_pred = list(frac_pos_arr), list(mean_pred_arr)

    beats_baseline = held_out_auc is not None and transparent_auc is not None and held_out_auc > transparent_auc

    report = CalibrationReport(
        model_name=model_name,
        proxy_label_name=proxy_label_name,
        train_size=len(y_train),
        held_out_size=len(y_test),
        held_out_auc=held_out_auc,
        held_out_brier_score=brier,
        transparent_baseline_auc=transparent_auc,
        beats_transparent_baseline=beats_baseline,
        reliability_curve_bin_true_frequency=frac_pos,
        reliability_curve_bin_predicted_probability=mean_pred,
    )
    return model, report

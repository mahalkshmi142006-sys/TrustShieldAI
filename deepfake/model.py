"""
TrustShieldAI Stage 3.3
Deepfake classifier training and prediction.

The classifier must be trained using real labelled data.
This module does NOT create fake training examples.
"""

from pathlib import Path

import cv2
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

from .forensics import (
    extract_forensic_features
)


# ============================================================
# FEATURE SCHEMA
# ============================================================

FEATURE_NAMES = [

    "sharpness",
    "brightness",
    "contrast",
    "edge_density",
    "noise_residual_std",

    "saturation_mean",

    "fft_low_energy",
    "fft_mid_energy",
    "fft_high_energy",
    "fft_high_mean",
    "fft_high_std",
    "fft_high_ratio",

    "face_count",
    "face_area_ratio",
    "face_sharpness",
    "face_fft_high_ratio"
]


MODEL_PATH = (
    Path(__file__).resolve().parent
    /
    "deepfake_rf.joblib"
)


# ============================================================
# FEATURE VECTOR
# ============================================================

def feature_vector(features):

    values = []

    for name in FEATURE_NAMES:

        value = features.get(
            name,
            0.0
        )

        if value is None:
            value = 0.0

        values.append(
            float(value)
        )

    return values


# ============================================================
# DATASET CREATION
# ============================================================

def build_dataset(dataset_root):

    dataset_root = Path(
        dataset_root
    )

    class_folders = [

        (
            dataset_root / "real",
            0
        ),

        (
            dataset_root / "fake",
            1
        )
    ]

    rows = []
    labels = []
    files = []

    extensions = {

        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    for folder, label in class_folders:

        if not folder.exists():
            continue

        for path in sorted(
            folder.rglob("*")
        ):

            if (
                path.suffix.lower()
                not in extensions
            ):
                continue

            image = cv2.imread(
                str(path),
                cv2.IMREAD_COLOR
            )

            if image is None:
                continue

            try:

                features = (
                    extract_forensic_features(
                        image
                    )
                )

                rows.append(
                    feature_vector(
                        features
                    )
                )

                labels.append(
                    label
                )

                files.append(
                    str(path)
                )

            except Exception:

                continue

    if len(labels) == 0:

        raise ValueError(
            "No readable images were found."
        )

    if len(set(labels)) < 2:

        raise ValueError(
            "Training data must contain "
            "both real and fake images."
        )

    return (
        np.asarray(
            rows,
            dtype=np.float32
        ),

        np.asarray(
            labels
        ),

        files
    )


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

def train(
    dataset_dir,
    model_path=MODEL_PATH,
    test_size=0.25,
    random_state=42
):

    X, y, files = build_dataset(
        dataset_dir
    )

    class_counts = np.bincount(
        y
    )

    if len(class_counts) < 2:

        raise ValueError(
            "Both real and fake classes are required."
        )

    if min(class_counts) < 2:

        raise ValueError(
            "Each class needs at least "
            "2 readable images."
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,
            y,

            test_size=test_size,

            random_state=random_state,

            stratify=y
        )
    )

    classifier = RandomForestClassifier(

        n_estimators=300,

        class_weight="balanced",

        random_state=random_state,

        min_samples_leaf=2,

        n_jobs=-1
    )

    classifier.fit(
        X_train,
        y_train
    )

    predictions = classifier.predict(
        X_test
    )

    probabilities = (
        classifier.predict_proba(
            X_test
        )[:, 1]
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            y_test,
            predictions,

            average="binary",

            zero_division=0
        )
    )

    metrics = {

        "dataset_size":
            int(len(y)),

        "train_size":
            int(len(y_train)),

        "test_size":
            int(len(y_test)),

        "real_count":
            int(
                (y == 0).sum()
            ),

        "fake_count":
            int(
                (y == 1).sum()
            ),

        "accuracy":
            round(
                float(
                    accuracy_score(
                        y_test,
                        predictions
                    )
                ),
                4
            ),

        "precision":
            round(
                float(precision),
                4
            ),

        "recall":
            round(
                float(recall),
                4
            ),

        "f1":
            round(
                float(f1),
                4
            ),

        "roc_auc":
            round(
                float(
                    roc_auc_score(
                        y_test,
                        probabilities
                    )
                ),
                4
            ),

        "confusion_matrix":
            confusion_matrix(
                y_test,
                predictions
            ).tolist(),

        "feature_names":
            FEATURE_NAMES,

        "classifier":
            "RandomForestClassifier",

        "random_state":
            random_state
    }

    bundle = {

        "model":
            classifier,

        "feature_names":
            FEATURE_NAMES,

        "metrics":
            metrics
    }

    model_path = Path(
        model_path
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        bundle,
        model_path
    )

    return metrics


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model(
    model_path=MODEL_PATH
):

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        return None

    return joblib.load(
        model_path
    )


# ============================================================
# PREDICT
# ============================================================

def predict(
    features,
    model_path=MODEL_PATH
):

    bundle = load_model(
        model_path
    )

    if bundle is None:

        return None

    vector = np.asarray(

        [
            feature_vector(
                features
            )
        ],

        dtype=np.float32
    )

    probability = float(

        bundle["model"]
        .predict_proba(
            vector
        )[0, 1]
    )

    return {

        "fake_probability":
            round(
                probability,
                4
            ),

        "authentic_probability":
            round(
                1 - probability,
                4
            ),

        "model":
            bundle[
                "metrics"
            ].get(
                "classifier",
                "RandomForestClassifier"
            ),

        "trained_dataset_size":
            bundle[
                "metrics"
            ].get(
                "dataset_size"
            )
    }

"""
TrustShieldAI Stage 3.4
Deepfake Model Evaluation and Explainability

Provides:
- Classification metrics
- Confusion matrix
- ROC-AUC
- ROC curve data
- Random Forest feature importance
- Human-readable evaluation report
"""

from pathlib import Path
import json

import cv2
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from deepfake.model import (
    FEATURE_NAMES,
    feature_vector,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "deepfake_dataset"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "deepfake"
    / "deepfake_rf.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "stage3_4_evaluation.json"
)

ROC_PATH = (
    OUTPUT_DIR
    / "stage3_4_roc_data.json"
)

IMPORTANCE_PATH = (
    OUTPUT_DIR
    / "stage3_4_feature_importance.json"
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def load_dataset():

    X = []
    y = []
    filenames = []

    class_folders = [
        ("real", 0),
        ("fake", 1),
    ]

    for class_name, label in class_folders:

        folder = DATASET_DIR / class_name

        if not folder.exists():
            continue

        for path in sorted(folder.rglob("*")):

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(
                str(path),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                continue

            try:

                from deepfake.forensics import (
                    extract_forensic_features
                )

                features = extract_forensic_features(
                    image
                )

                X.append(
                    feature_vector(features)
                )

                y.append(label)

                filenames.append(
                    str(path)
                )

            except Exception:
                continue

    if len(X) == 0:
        raise ValueError(
            "No readable dataset images found."
        )

    if len(set(y)) < 2:
        raise ValueError(
            "Both real and fake images are required."
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        filenames,
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 60)
    print("TRUSTSHIELD AI - STAGE 3.4")
    print("DEEPFAKE MODEL EVALUATION")
    print("=" * 60)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    bundle = joblib.load(
        MODEL_PATH
    )

    classifier = bundle["model"]

    X, y, filenames = load_dataset()

    predictions = classifier.predict(
        X
    )

    probabilities = classifier.predict_proba(
        X
    )[:, 1]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(
            y,
            probabilities
        )
    )

    roc_data = {

        "false_positive_rate":
            false_positive_rate.tolist(),

        "true_positive_rate":
            true_positive_rate.tolist(),

        "thresholds":
            thresholds.tolist(),

        "roc_auc":
            float(roc_auc),
    }

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importances = classifier.feature_importances_

    feature_importance = []

    for name, importance in zip(
        FEATURE_NAMES,
        importances
    ):

        feature_importance.append({

            "feature":
                name,

            "importance":
                round(
                    float(importance),
                    6
                ),
        })

    feature_importance.sort(
        key=lambda item:
        item["importance"],
        reverse=True
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y,
        predictions,
        target_names=[
            "Real",
            "Fake"
        ],
        output_dict=True,
        zero_division=0,
    )

    # --------------------------------------------------------
    # EVALUATION RESULT
    # --------------------------------------------------------

    result = {

        "stage":
            "3.4",

        "description":
            "Deepfake classifier evaluation and explainability",

        "dataset_size":
            int(len(y)),

        "real_count":
            int((y == 0).sum()),

        "fake_count":
            int((y == 1).sum()),

        "accuracy":
            round(
                float(accuracy),
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

        "f1_score":
            round(
                float(f1),
                4
            ),

        "roc_auc":
            round(
                float(roc_auc),
                4
            ),

        "confusion_matrix":
            cm.tolist(),

        "classification_report":
            report,

        "classifier":
            "RandomForestClassifier",

        "feature_count":
            len(FEATURE_NAMES),

        "feature_importance":
            feature_importance,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2
        )

    with open(
        ROC_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            roc_data,
            file,
            indent=2
        )

    with open(
        IMPORTANCE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            feature_importance,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()
    print("Evaluation completed.")
    print()

    print(
        f"Dataset size : {len(y)}"
    )

    print(
        f"Accuracy     : {accuracy:.4f}"
    )

    print(
        f"Precision    : {precision:.4f}"
    )

    print(
        f"Recall       : {recall:.4f}"
    )

    print(
        f"F1 Score     : {f1:.4f}"
    )

    print(
        f"ROC-AUC      : {roc_auc:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print("Top forensic features:")

    for item in feature_importance[:10]:

        print(
            f"  {item['feature']:<25}"
            f"{item['importance']:.6f}"
        )

    print()
    print("Saved files:")
    print(
        REPORT_PATH
    )

    print(
        ROC_PATH
    )

    print(
        IMPORTANCE_PATH
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()


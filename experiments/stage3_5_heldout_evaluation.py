
"""
TrustShieldAI Stage 3.5
Proper Held-Out Deepfake Evaluation

Protocol:
    70% training
    15% validation
    15% independent test

The test set is never used to train the final classifier.

Outputs:
    - Final test metrics
    - Confusion matrix
    - ROC-AUC
    - ROC curve data
    - Feature importance
    - JSON evaluation report
"""

from pathlib import Path
import json

import cv2
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import train_test_split

from deepfake.model import (
    FEATURE_NAMES,
    feature_vector,
)
from deepfake.forensics import (
    extract_forensic_features,
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "deepfake"
    / "deepfake_rf_stage3_5.joblib"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "stage3_5_heldout_results.json"
)

ROC_PATH = (
    OUTPUT_DIR
    / "stage3_5_heldout_roc.json"
)

IMPORTANCE_PATH = (
    OUTPUT_DIR
    / "stage3_5_feature_importance.json"
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


RANDOM_STATE = 42


# ============================================================
# LOAD ALL IMAGES
# ============================================================

def load_dataset():

    X = []
    y = []
    files = []

    classes = [
        ("real", 0),
        ("fake", 1),
    ]

    print("Loading dataset...")

    for class_name, label in classes:

        folder = DATASET_DIR / class_name

        if not folder.exists():

            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        image_paths = sorted(
            [
                path
                for path in folder.rglob("*")
                if (
                    path.is_file()
                    and
                    path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]
        )

        print(
            f"{class_name.capitalize()} images: "
            f"{len(image_paths)}"
        )

        for index, path in enumerate(image_paths, start=1):

            image = cv2.imread(
                str(path),
                cv2.IMREAD_COLOR
            )

            if image is None:
                continue

            try:

                features = extract_forensic_features(
                    image
                )

                X.append(
                    feature_vector(features)
                )

                y.append(label)

                files.append(
                    str(path)
                )

            except Exception as error:

                print(
                    f"Skipping {path.name}: {error}"
                )

    if len(X) == 0:

        raise ValueError(
            "No readable images found."
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    if len(set(y)) != 2:

        raise ValueError(
            "Both Real and Fake classes are required."
        )

    return X, y, files


# ============================================================
# SPLIT DATA
# ============================================================

def create_splits(X, y, files):

    # First:
    # 85% temporary training+validation
    # 15% independent test

    X_temp, X_test, y_temp, y_test, files_temp, files_test = (
        train_test_split(
            X,
            y,
            files,
            test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # From the remaining 85%:
    # 70% overall training
    # 15% overall validation

    validation_fraction = (
        0.15 / 0.85
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
        files_train,
        files_validation,
    ) = train_test_split(
        X_temp,
        y_temp,
        files_temp,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        files_train,
        files_validation,
        files_test,
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_classifier(X_train, y_train):

    classifier = RandomForestClassifier(

        n_estimators=300,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        min_samples_leaf=2,

        n_jobs=-1,
    )

    classifier.fit(
        X_train,
        y_train
    )

    return classifier


# ============================================================
# EVALUATE
# ============================================================

def evaluate_classifier(
    classifier,
    X_test,
    y_test,
):

    predictions = classifier.predict(
        X_test
    )

    probabilities = classifier.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    return (
        predictions,
        probabilities,
        {
            "accuracy": round(
                float(accuracy),
                4
            ),

            "precision": round(
                float(precision),
                4
            ),

            "recall": round(
                float(recall),
                4
            ),

            "f1_score": round(
                float(f1),
                4
            ),

            "roc_auc": round(
                float(roc_auc),
                4
            ),

            "confusion_matrix":
                cm.tolist(),

            "classification_report":
                classification_report(
                    y_test,
                    predictions,
                    target_names=[
                        "Real",
                        "Fake"
                    ],
                    output_dict=True,
                    zero_division=0,
                ),
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("TRUSTSHIELD AI - STAGE 3.5")
    print("PROPER HELD-OUT DEEPFAKE EVALUATION")
    print("=" * 60)
    print()

    X, y, files = load_dataset()

    print()
    print("Creating reproducible data split...")

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        files_train,
        files_validation,
        files_test,
    ) = create_splits(
        X,
        y,
        files
    )

    print()
    print("Dataset split:")
    print(
        f"Training   : {len(y_train)}"
    )
    print(
        f"Validation : {len(y_validation)}"
    )
    print(
        f"Test       : {len(y_test)}"
    )

    print()
    print("Training Random Forest...")

    classifier = train_classifier(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_predictions = classifier.predict(
        X_validation
    )

    validation_probabilities = (
        classifier.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    validation_auc = roc_auc_score(
        y_validation,
        validation_probabilities
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print()
    print("Evaluating on completely held-out test set...")

    (
        predictions,
        probabilities,
        test_metrics,
    ) = evaluate_classifier(
        classifier,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(
            y_test,
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
            test_metrics["roc_auc"],
    }

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_values = (
        classifier.feature_importances_
    )

    feature_importance = []

    for name, importance in zip(
        FEATURE_NAMES,
        importance_values
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
    # SAVE MODEL
    # --------------------------------------------------------

    bundle = {

        "model":
            classifier,

        "feature_names":
            FEATURE_NAMES,

        "protocol":
            {
                "training_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "random_state":
                    RANDOM_STATE,
            },

        "training_size":
            len(y_train),

        "validation_size":
            len(y_validation),

        "test_size":
            len(y_test),

        "test_metrics":
            test_metrics,
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        bundle,
        MODEL_PATH
    )

    # --------------------------------------------------------
    # COMPLETE REPORT
    # --------------------------------------------------------

    result = {

        "stage":
            "3.5",

        "description":
            "Proper held-out deepfake evaluation",

        "protocol":
            {
                "training_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "random_state":
                    RANDOM_STATE,
            },

        "dataset_size":
            len(y),

        "training_size":
            len(y_train),

        "validation_size":
            len(y_validation),

        "test_size":
            len(y_test),

        "training_real":
            int((y_train == 0).sum()),

        "training_fake":
            int((y_train == 1).sum()),

        "validation_real":
            int((y_validation == 0).sum()),

        "validation_fake":
            int((y_validation == 1).sum()),

        "test_real":
            int((y_test == 0).sum()),

        "test_fake":
            int((y_test == 1).sum()),

        "validation_accuracy":
            round(
                float(validation_accuracy),
                4
            ),

        "validation_roc_auc":
            round(
                float(validation_auc),
                4
            ),

        "test_metrics":
            test_metrics,

        "feature_importance":
            feature_importance,

        "model":
            "RandomForestClassifier",

        "feature_count":
            len(FEATURE_NAMES),

        "note":
            "The independent test set was not used "
            "to train the classifier.",
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

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
    # DISPLAY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("STAGE 3.5 RESULTS")
    print("=" * 60)

    print(
        f"Training size   : {len(y_train)}"
    )

    print(
        f"Validation size : {len(y_validation)}"
    )

    print(
        f"Test size       : {len(y_test)}"
    )

    print()

    print(
        f"Test Accuracy   : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Precision  : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall     : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1         : "
        f"{test_metrics['f1_score']:.4f}"
    )

    print(
        f"Test ROC-AUC    : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print()
    print("Confusion Matrix:")

    print(
        np.asarray(
            test_metrics["confusion_matrix"]
        )
    )

    print()
    print("Top forensic features:")

    for item in feature_importance[:10]:

        print(
            f"  {item['feature']:<25}"
            f"{item['importance']:.6f}"
        )

    print()
    print("Saved model:")

    print(
        MODEL_PATH
    )

    print()
    print("Saved evaluation:")

    print(
        REPORT_PATH
    )

    print()
    print("Saved ROC data:")

    print(
        ROC_PATH
    )

    print()
    print("Saved feature importance:")

    print(
        IMPORTANCE_PATH
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()


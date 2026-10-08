"""Shortcut ablation: does the deepfake classifier depend on dataset-framing features?
Runs 5-fold stratified CV (repeated 3x) for several feature subsets."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deepfake.model import build_dataset, FEATURE_NAMES
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

X, y, files = build_dataset(ROOT / "data" / "deepfake_dataset")
idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
FACE_FRAMING = ["face_count", "face_area_ratio"]
FACE_ALL = ["face_count", "face_area_ratio", "face_sharpness", "face_fft_high_ratio"]
subsets = {
    "A. all 16 features (original)": FEATURE_NAMES,
    "B. drop face_count + face_area_ratio": [f for f in FEATURE_NAMES if f not in FACE_FRAMING],
    "C. drop all 4 face features": [f for f in FEATURE_NAMES if f not in FACE_ALL],
    "D. ONLY face_area_ratio (shortcut check)": ["face_area_ratio"],
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
out = {}
for name, cols in subsets.items():
    Xs = X[:, [idx[c] for c in cols]]
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    r = cross_validate(clf, Xs, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    out[name] = {k: [round(float(r[f"test_{k}"].mean()), 4), round(float(r[f"test_{k}"].std()), 4)]
                 for k in ["accuracy", "f1", "roc_auc"]}
    print(f"{name:45s} acc={out[name]['accuracy'][0]:.3f}±{out[name]['accuracy'][1]:.3f}  "
          f"f1={out[name]['f1'][0]:.3f}  auc={out[name]['roc_auc'][0]:.3f}")
(ROOT / "experiments" / "shortcut_ablation_results.json").write_text(json.dumps(out, indent=2))

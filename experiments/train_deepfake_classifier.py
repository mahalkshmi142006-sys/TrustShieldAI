"""
TrustShieldAI Stage 3.3

Train and evaluate the deepfake Random Forest classifier.

Dataset structure:

data/
└── deepfake_dataset/
    ├── real/
    │   ├── image1.jpg
    │   └── image2.jpg
    │
    └── fake/
        ├── image1.jpg
        └── image2.jpg
"""

import argparse
import json
from pathlib import Path

from deepfake.model import train


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


parser = argparse.ArgumentParser(
    description=
    "Train TrustShieldAI deepfake classifier."
)


parser.add_argument(

    "--dataset",

    default=str(
        ROOT
        /
        "data"
        /
        "deepfake_dataset"
    )
)


parser.add_argument(

    "--model",

    default=str(
        ROOT
        /
        "deepfake"
        /
        "deepfake_rf.joblib"
    )
)


args = parser.parse_args()


metrics = train(

    args.dataset,

    args.model
)


output_path = (
    ROOT
    /
    "experiments"
    /
    "deepfake_evaluation_results.json"
)


output_path.write_text(

    json.dumps(
        metrics,
        indent=2
    ),

    encoding="utf-8"
)


print()
print("=" * 60)
print("TRUSTSHIELD AI - DEEPFAKE CLASSIFIER")
print("=" * 60)
print()

print(
    json.dumps(
        metrics,
        indent=2
    )
)

print()

print(
    f"Saved model: {args.model}"
)

print(
    f"Saved evaluation: {output_path}"
)
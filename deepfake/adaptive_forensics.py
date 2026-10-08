from __future__ import annotations

import cv2
import numpy as np


def _safe_float(
    value,
    default: float = 0.0
) -> float:
    """
    Safely convert a value to float.

    Handles None, invalid strings, NaN and infinity.
    """

    try:
        if value is None:
            return float(default)

        result = float(value)

        if not np.isfinite(result):
            return float(default)

        return result

    except (
        TypeError,
        ValueError
    ):
        return float(default)


def _quality_factors(
    base_features: dict
) -> dict:
    """
    Estimate image-quality factors from the
    available forensic features.
    """

    sharpness = _safe_float(
        base_features.get(
            "sharpness"
        )
    )

    contrast = _safe_float(
        base_features.get(
            "contrast"
        )
    )

    noise = _safe_float(
        base_features.get(
            "noise_residual_std"
        )
    )

    # Normalize sharpness approximately to 0-1.
    sharpness_quality = min(
        1.0,
        max(
            0.0,
            sharpness / 100.0
        )
    )

    # Normalize contrast approximately to 0-1.
    contrast_quality = min(
        1.0,
        max(
            0.0,
            contrast / 80.0
        )
    )

    # Moderate residual noise is not automatically bad.
    # This is only a quality heuristic.
    noise_quality = min(
        1.0,
        max(
            0.0,
            noise / 10.0
        )
    )

    overall_quality = (
        0.45 * sharpness_quality
        +
        0.35 * contrast_quality
        +
        0.20 * noise_quality
    )

    overall_quality = min(
        1.0,
        max(
            0.0,
            overall_quality
        )
    )

    return {
        "sharpness_quality":
            round(
                sharpness_quality,
                4
            ),

        "contrast_quality":
            round(
                contrast_quality,
                4
            ),

        "noise_quality":
            round(
                noise_quality,
                4
            ),

        "overall_quality":
            round(
                overall_quality,
                4
            )
    }


def extract_dct_features(
    gray: np.ndarray
) -> dict:
    """
    Extract simple frequency-domain features
    using the 2-D Discrete Cosine Transform.
    """

    gray = np.asarray(
        gray,
        dtype=np.float32
    )

    if gray.ndim != 2:
        gray = cv2.cvtColor(
            gray.astype(np.uint8),
            cv2.COLOR_BGR2GRAY
        ).astype(
            np.float32
        )

    if gray.size == 0:
        return {
            "dct_low_energy": 0.0,
            "dct_mid_energy": 0.0,
            "dct_high_energy": 0.0,
            "dct_high_ratio": 0.0
        }

    # Normalize image before DCT.
    gray = gray / 255.0

    dct = cv2.dct(
        gray
    )

    energy = np.square(
        np.abs(dct)
    )

    height, width = energy.shape

    # Divide the DCT coefficients into
    # low, middle and high-frequency regions.
    low_end = max(
        1,
        min(
            height,
            width
        ) // 8
    )

    mid_end = max(
        low_end + 1,
        min(
            height,
            width
        ) // 3
    )

    low_region = energy[
        :low_end,
        :low_end
    ]

    mid_region = energy[
        low_end:mid_end,
        low_end:mid_end
    ]

    high_region = energy[
        mid_end:,
        mid_end:
    ]

    low_energy = _safe_float(
        np.mean(low_region)
    )

    mid_energy = _safe_float(
        np.mean(mid_region)
    )

    high_energy = _safe_float(
        np.mean(high_region)
    )

    total_energy = (
        low_energy
        +
        mid_energy
        +
        high_energy
    )

    if total_energy > 0:
        high_ratio = (
            high_energy
            /
            total_energy
        )
    else:
        high_ratio = 0.0

    return {
        "dct_low_energy":
            low_energy,

        "dct_mid_energy":
            mid_energy,

        "dct_high_energy":
            high_energy,

        "dct_high_ratio":
            _safe_float(
                high_ratio
            )
    }


def adaptive_forensic_features(
    image: np.ndarray,
    base_features: dict
) -> dict:
    """
    Return quality-aware adaptive forensic features.

    If face_sharpness is unavailable because no face
    was detected, the normal image sharpness is used
    as a fallback.
    """

    image = np.asarray(
        image
    )

    if image.size == 0:
        return {
            "dct_low_energy": 0.0,
            "dct_mid_energy": 0.0,
            "dct_high_energy": 0.0,
            "dct_high_ratio": 0.0,
            "frequency_weight": 0.45,
            "spatial_weight": 0.55,
            "spatial_signal": 0.0,
            "frequency_signal": 0.0,
            "adaptive_forensic_signal": 0.0,
            "overall_quality": 0.0
        }

    # --------------------------------------------------------
    # Convert image to grayscale.
    # --------------------------------------------------------

    gray = (
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
        if image.ndim == 3
        else image
    )

    # --------------------------------------------------------
    # Calculate image-quality factors.
    # --------------------------------------------------------

    quality = _quality_factors(
        base_features
    )

    # --------------------------------------------------------
    # Extract DCT frequency features.
    # --------------------------------------------------------

    dct = extract_dct_features(
        gray
    )

    # --------------------------------------------------------
    # Poor image quality gives more weight
    # to frequency-domain evidence.
    # --------------------------------------------------------

    frequency_weight = round(
        0.45
        +
        0.35
        *
        (
            1.0
            -
            quality[
                "overall_quality"
            ]
        ),
        4
    )

    spatial_weight = round(
        1.0
        -
        frequency_weight,
        4
    )

    # --------------------------------------------------------
    # FACE SHARPNESS
    #
    # Important:
    #
    # face_sharpness can be None when no face exists.
    # In that situation, fall back to normal sharpness.
    # --------------------------------------------------------

    face_sharpness = (
        base_features.get(
            "face_sharpness"
        )
    )

    if face_sharpness is None:

        face_sharpness = (
            base_features.get(
                "sharpness",
                0.0
            )
        )

    spatial_signal = _safe_float(
        face_sharpness
    )

    # --------------------------------------------------------
    # Frequency-domain signal.
    # --------------------------------------------------------

    frequency_signal = _safe_float(
        dct.get(
            "dct_high_ratio",
            0.0
        )
    )

    # --------------------------------------------------------
    # Combined adaptive forensic signal.
    # --------------------------------------------------------

    combined_signal = (
        spatial_weight
        *
        spatial_signal
        +
        frequency_weight
        *
        frequency_signal
    )

    return {
        **dct,

        "frequency_weight":
            frequency_weight,

        "spatial_weight":
            spatial_weight,

        "spatial_signal":
            spatial_signal,

        "frequency_signal":
            frequency_signal,

        "adaptive_forensic_signal":
            _safe_float(
                combined_signal
            ),

        "overall_quality":
            _safe_float(
                quality.get(
                    "overall_quality",
                    0.0
                )
            )
    }
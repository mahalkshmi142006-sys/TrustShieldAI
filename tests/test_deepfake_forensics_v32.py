import cv2
import numpy as np

from deepfake.forensics import (
    extract_forensic_features,
    analyze_media
)


def test_spatial_and_frequency_features_are_present():

    image = np.zeros(
        (256, 256, 3),
        dtype=np.uint8
    )

    cv2.rectangle(
        image,
        (40, 40),
        (210, 210),
        (255, 255, 255),
        -1
    )

    features = (
        extract_forensic_features(
            image
        )
    )

    required = {

        "sharpness",
        "contrast",
        "edge_density",
        "noise_residual_std",

        "fft_low_energy",
        "fft_mid_energy",
        "fft_high_energy",
        "fft_high_ratio",

        "face_count"
    }

    assert required.issubset(
        features
    )

    for key in required:

        assert np.isfinite(
            float(
                features[key]
            )
        )


def test_image_analysis_reports_stage_3_2(
    tmp_path
):

    path = (
        tmp_path
        /
        "sample.png"
    )

    image = np.random.default_rng(
        7
    ).integers(
        0,
        256,
        (300, 300, 3),
        dtype=np.uint8
    )

    assert cv2.imwrite(
        str(path),
        image
    )

    result = analyze_media(
        path
    )

    assert result[
        "type"
    ] == "image"

    assert (
        "fft_high_ratio"
        in result
    )

    assert (
        "forensic_indicators"
        in result
    )

    assert (
        "Stage 3.2"
        in result["method"]
    )
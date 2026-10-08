from pathlib import Path

import cv2
import numpy as np

from .adaptive_forensics import adaptive_forensic_features


# ============================================================
# TRUSTSHIELD AI - DEEPFAKE FORENSICS
# Stage 3.2 + Stage 3.3
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}

MAX_BYTES = 100 * 1024 * 1024


# ============================================================
# BASIC IMAGE UTILITIES
# ============================================================

def _gray(image):
    """Convert image to grayscale."""

    if image is None:
        raise ValueError(
            "Could not read media frame."
        )

    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


# ============================================================
# FREQUENCY-DOMAIN FEATURES
# ============================================================

def _frequency_features(gray):
    """
    Extract FFT-based frequency-domain features.

    These features are forensic signals and should not
    be treated as proof of manipulation by themselves.
    """

    image = cv2.resize(
        gray,
        (256, 256),
        interpolation=cv2.INTER_AREA
    ).astype(
        np.float32
    )

    image -= image.mean()

    window = cv2.createHanningWindow(
        (256, 256),
        cv2.CV_32F
    )

    fft = np.fft.fft2(
        image * window
    )

    shifted = np.fft.fftshift(
        fft
    )

    magnitude = np.log1p(
        np.abs(shifted)
    )

    yy, xx = np.indices(
        magnitude.shape
    )

    radius = np.sqrt(
        (xx - 127.5) ** 2
        +
        (yy - 127.5) ** 2
    )

    total_energy = (
        float(
            magnitude.sum()
        )
        +
        1e-8
    )

    features = {}

    frequency_bands = [
        ("low", 0, 20),
        ("mid", 20, 60),
        ("high", 60, 128)
    ]

    for name, low, high in frequency_bands:

        mask = (
            (radius >= low)
            &
            (radius < high)
        )

        energy = magnitude[
            mask
        ].sum()

        features[
            f"fft_{name}_energy"
        ] = float(
            energy / total_energy
        )

    high_frequency = magnitude[
        radius >= 60
    ]

    features[
        "fft_high_mean"
    ] = float(
        high_frequency.mean()
    )

    features[
        "fft_high_std"
    ] = float(
        high_frequency.std()
    )

    features[
        "fft_high_ratio"
    ] = float(
        high_frequency.sum()
        /
        total_energy
    )

    return features


# ============================================================
# SPATIAL FEATURES
# ============================================================

def _spatial_features(gray):
    """
    Extract spatial image-quality and forensic features.
    """

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    noise_residual = (
        gray.astype(
            np.float32
        )
        -
        blurred.astype(
            np.float32
        )
    )

    return {

        "sharpness":
            float(
                laplacian.var()
            ),

        "brightness":
            float(
                gray.mean()
            ),

        "contrast":
            float(
                gray.std()
            ),

        "edge_density":
            float(
                edges.mean()
                /
                255.0
            ),

        "noise_residual_std":
            float(
                noise_residual.std()
            ),

        "saturation_mean":
            0.0
    }


# ============================================================
# FACE FEATURES
# ============================================================

def _face_features(
    image,
    gray
):
    """
    Detect faces and calculate forensic
    characteristics inside detected face regions.
    """

    classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades
        +
        "haarcascade_frontalface_default.xml"
    )

    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    result = {

        "face_count":
            int(
                len(faces)
            ),

        "face_area_ratio":
            0.0,

        "face_sharpness":
            None,

        "face_fft_high_ratio":
            None
    }

    if len(faces) == 0:
        return result

    height, width = (
        gray.shape[:2]
    )

    areas = []
    sharpness_values = []
    frequency_values = []

    for (
        x,
        y,
        face_width,
        face_height
    ) in faces:

        crop = gray[
            y:y + face_height,
            x:x + face_width
        ]

        if crop.size == 0:
            continue

        areas.append(
            (
                face_width
                *
                face_height
            )
            /
            (
                width
                *
                height
            )
        )

        sharpness_values.append(
            _spatial_features(
                crop
            )[
                "sharpness"
            ]
        )

        frequency_values.append(
            _frequency_features(
                crop
            )[
                "fft_high_ratio"
            ]
        )

    if areas:

        result[
            "face_area_ratio"
        ] = float(
            np.mean(
                areas
            )
        )

        result[
            "face_sharpness"
        ] = float(
            np.mean(
                sharpness_values
            )
        )

        result[
            "face_fft_high_ratio"
        ] = float(
            np.mean(
                frequency_values
            )
        )

    return result


# ============================================================
# COMPLETE FEATURE EXTRACTION
# ============================================================

def extract_forensic_features(
    image
):
    """
    Extract the complete Stage 3.2
    forensic feature set.
    """

    gray = _gray(
        image
    )

    if gray.size == 0:

        raise ValueError(
            "Empty image/frame."
        )

    # Prevent extremely large images from consuming
    # excessive memory during FFT processing.
    if max(
        gray.shape
    ) > 1280:

        scale = (
            1280
            /
            max(
                gray.shape
            )
        )

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        image = cv2.resize(
            image,
            (
                gray.shape[1],
                gray.shape[0]
            ),
            interpolation=cv2.INTER_AREA
        )

    features = {}

    # Stage 3.2 spatial features
    features.update(
        _spatial_features(
            gray
        )
    )

    # Stage 3.2 frequency features
    features.update(
        _frequency_features(
            gray
        )
    )

    # Stage 3.2 face features
    features.update(
        _face_features(
            image,
            gray
        )
    )

    # Color/saturation feature
    if len(
        image.shape
    ) == 3:

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        features[
            "saturation_mean"
        ] = float(
            hsv[:, :, 1].mean()
        )

    return features


# ============================================================
# MEDIA VALIDATION
# ============================================================

def _validate_media(
    path
):

    path = Path(
        path
    )

    if not path.exists():

        raise ValueError(
            "Media file does not exist."
        )

    if not path.is_file():

        raise ValueError(
            "Media path is not a file."
        )

    if path.stat().st_size > MAX_BYTES:

        raise ValueError(
            "Media file is larger than "
            "the 100 MB safety limit."
        )

    return path


# ============================================================
# BASELINE FORENSIC RISK
# ============================================================

def _forensic_screening(
    features
):

    indicators = []

    if features.get(
        "face_count",
        0
    ) == 0:

        indicators.append(
            "No face detected"
        )

    if features.get(
        "sharpness",
        0
    ) < 35:

        indicators.append(
            "Low image sharpness"
        )

    if features.get(
        "contrast",
        0
    ) < 25:

        indicators.append(
            "Low contrast"
        )

    if features.get(
        "noise_residual_std",
        0
    ) < 1.5:

        indicators.append(
            "Low residual noise"
        )

    if features.get(
        "fft_high_ratio",
        0
    ) < 0.12:

        indicators.append(
            "Low high-frequency energy"
        )

    face_frequency = features.get(
        "face_fft_high_ratio"
    )

    if (
        face_frequency is not None
        and
        face_frequency < 0.10
    ):

        indicators.append(
            "Low high-frequency energy "
            "in face region"
        )

    risk = (
        30
        +
        8 * len(
            indicators
        )
    )

    risk = min(
        95,
        max(
            5,
            risk
        )
    )

    return (
        risk,
        indicators
    )


# ============================================================
# TRAINED MODEL
# ============================================================

def _model_prediction(
    features
):

    try:

        from .model import predict

        return predict(
            features
        )

    except Exception:

        return None


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def _analyze_image(
    image
):

    # --------------------------------------------------------
    # Stage 3.2
    # --------------------------------------------------------

    features = extract_forensic_features(
        image
    )

    # --------------------------------------------------------
    # Adaptive Stage 3.2 analysis
    # --------------------------------------------------------

    adaptive_result = (
        adaptive_forensic_features(
            image,
            features
        )
    )

    # --------------------------------------------------------
    # Baseline forensic screening
    # --------------------------------------------------------

    risk, indicators = (
        _forensic_screening(
            features
        )
    )

    # --------------------------------------------------------
    # Stage 3.3 trained classifier
    # --------------------------------------------------------

    prediction = _model_prediction(
        features
    )

    # --------------------------------------------------------
    # Base result
    # --------------------------------------------------------

    result = {

        "type":
            "image",

        "deepfake_risk":
            int(
                risk
            ),

        "authenticity_score":
            100 - int(
                risk
            ),

        **features,

        **adaptive_result,

        "forensic_indicators":
            indicators,

        "classifier_enabled":
            False,

        "method":
            (
                "Stage 3.2 spatial + frequency "
                "forensic screening"
            )
    }

    # --------------------------------------------------------
    # Stage 3.3 classifier result
    # --------------------------------------------------------

    if prediction is not None:

        fake_probability = float(
            prediction[
                "fake_probability"
            ]
        )

        authentic_probability = float(
            prediction[
                "authentic_probability"
            ]
        )

        result.update({

            "classifier_enabled":
                True,

            "classifier_fake_probability":
                fake_probability,

            "classifier_authentic_probability":
                authentic_probability,

            "classifier":
                prediction[
                    "model"
                ],

            "trained_dataset_size":
                prediction[
                    "trained_dataset_size"
                ],

            "deepfake_risk":
                round(
                    fake_probability
                    *
                    100
                ),

            "authenticity_score":
                round(
                    authentic_probability
                    *
                    100
                ),

            # IMPORTANT:
            # Keep Stage 3.2 in the method name
            # because Stage 3.3 uses the Stage 3.2
            # spatial/frequency forensic features.
            "method":
                (
                    "Stage 3.2 spatial + frequency "
                    "forensic feature extraction + "
                    "Stage 3.3 Random Forest classifier"
                )
        })

    else:

        result[
            "classifier_status"
        ] = (
            "No trained classifier found; "
            "forensic screening only."
        )

    return result


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def _analyze_video(
    capture
):

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
        or 0
    )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
        or 0
    )

    if total_frames:

        sample_count = min(
            30,
            max(
                5,
                total_frames
            )
        )

        indices = np.linspace(
            0,
            total_frames - 1,
            sample_count
        ).astype(
            int
        )

    else:

        sample_count = 30

        indices = (
            np.arange(
                sample_count
            )
            *
            5
        )

    frame_features = []
    temporal_differences = []

    previous_gray = None

    for index in indices:

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = (
            capture.read()
        )

        if not success:
            continue

        features = (
            extract_forensic_features(
                frame
            )
        )

        frame_features.append(
            features
        )

        current_gray = _gray(
            frame
        )

        if previous_gray is not None:

            previous_resized = cv2.resize(
                previous_gray,
                (160, 160)
            )

            current_resized = cv2.resize(
                current_gray,
                (160, 160)
            )

            difference = cv2.absdiff(
                previous_resized,
                current_resized
            )

            temporal_differences.append(
                float(
                    np.mean(
                        difference
                    )
                    /
                    255.0
                )
            )

        previous_gray = current_gray

    if not frame_features:

        raise ValueError(
            "No readable video frames."
        )

    averaged_keys = [

        "sharpness",
        "brightness",
        "contrast",
        "edge_density",
        "noise_residual_std",
        "fft_low_energy",
        "fft_mid_energy",
        "fft_high_energy",
        "fft_high_ratio"
    ]

    averages = {

        key:
            float(
                np.mean(
                    [
                        f[key]
                        for f in frame_features
                    ]
                )
            )

        for key in averaged_keys
    }

    face_counts = [
        f[
            "face_count"
        ]
        for f in frame_features
    ]

    representative_features = {
        **averages,

        "face_count":
            int(
                round(
                    np.mean(
                        face_counts
                    )
                )
            )
    }

    risk, indicators = (
        _forensic_screening(
            representative_features
        )
    )

    if temporal_differences:

        temporal_difference = float(
            np.mean(
                temporal_differences
            )
        )

        averages[
            "temporal_frame_difference"
        ] = temporal_difference

        if temporal_difference < 0.01:

            indicators.append(
                "Very low frame-to-frame change"
            )

            risk = min(
                95,
                risk + 5
            )

    return {

        "type":
            "video",

        "deepfake_risk":
            int(
                risk
            ),

        "authenticity_score":
            100 - int(
                risk
            ),

        "sampled_frames":
            len(
                frame_features
            ),

        "fps":
            round(
                fps,
                3
            ),

        "total_frames":
            total_frames,

        "faces_detected_across_samples":
            int(
                sum(
                    count > 0
                    for count in face_counts
                )
            ),

        **{
            key:
                round(
                    value,
                    6
                )

            for key, value
            in averages.items()
        },

        "forensic_indicators":
            indicators,

        "classifier_enabled":
            False,

        "method":
            (
                "Stage 3.2 spatial + frequency + "
                "temporal forensic screening; "
                "not a trained deepfake classifier"
            )
    }


# ============================================================
# PUBLIC MEDIA ANALYSIS FUNCTION
# ============================================================

def analyze_media(
    path
):

    path = _validate_media(
        path
    )

    extension = (
        path.suffix.lower()
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if extension in IMAGE_EXTENSIONS:

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise ValueError(
                "Could not read image."
            )

        return _analyze_image(
            image
        )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    if extension in VIDEO_EXTENSIONS:

        capture = cv2.VideoCapture(
            str(path)
        )

        if not capture.isOpened():

            raise ValueError(
                "Could not open video."
            )

        try:

            return _analyze_video(
                capture
            )

        finally:

            capture.release()

    # --------------------------------------------------------
    # UNSUPPORTED FORMAT
    # --------------------------------------------------------

    raise ValueError(
        "Unsupported media type. "
        "Supported images: JPG, JPEG, PNG, BMP, WEBP. "
        "Supported videos: MP4, AVI, MOV, MKV, WEBM."
    )
"""TrustShieldAI v4 - cross-modal video/audio grounding prototype.

Checks video motion around the mouth against audio speech activity.
This is a forensic consistency signal, not a claim of full lip-reading
or voice identity.
Audio extraction uses ffmpeg when available.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np


def _audio_activity(video_path: Path) -> dict:
    """Extract basic audio activity information using FFmpeg."""

    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        return {
            "available": False,
            "reason": "ffmpeg not installed",
        }

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"

        command = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ]

        try:
            process = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
        except Exception as exc:
            return {
                "available": False,
                "reason": f"audio extraction failed: {exc}",
            }

        if process.returncode != 0 or not wav_path.exists():
            return {
                "available": False,
                "reason": "audio extraction failed",
            }

        try:
            with wave.open(str(wav_path), "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                rate = wav.getframerate()
        except Exception as exc:
            return {
                "available": False,
                "reason": f"could not read extracted audio: {exc}",
            }

        audio = np.frombuffer(
            frames,
            dtype=np.int16,
        ).astype(np.float32)

        if audio.size == 0:
            return {
                "available": False,
                "reason": "empty audio",
            }

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(audio)
                )
            )
            / 32768.0
        )

        return {
            "available": True,
            "rms": round(rms, 6),
            "sample_rate": rate,
        }


def _mouth_motion(
    video_path: Path,
    sample_count: int = 24,
) -> dict:
    """Estimate mouth-region motion across sampled video frames."""

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        return {
            "available": False,
            "reason": "video could not be opened",
        }

    total = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
        or 0
    )

    if total:
        count = min(
            sample_count,
            max(1, total),
        )

        indices = np.linspace(
            0,
            total - 1,
            count,
        ).astype(int)

    else:
        indices = np.arange(
            sample_count
        )

    previous = None
    motions = []

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    try:
        for index in indices:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index),
            )

            success, frame = capture.read()

            if not success:
                continue

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            faces = face_cascade.detectMultiScale(
                gray,
                1.1,
                4,
            )

            if len(faces) == 0:
                continue

            x, y, w, h = max(
                faces,
                key=lambda box: box[2] * box[3],
            )

            mouth = gray[
                y + int(h * 0.55):
                y + int(h * 0.95),
                x + int(w * 0.20):
                x + int(w * 0.80),
            ]

            if mouth.size == 0:
                continue

            mouth = cv2.resize(
                mouth,
                (64, 32),
            )

            if previous is not None:
                motion = float(
                    np.mean(
                        cv2.absdiff(
                            previous,
                            mouth,
                        )
                    )
                    / 255.0
                )

                motions.append(
                    motion
                )

            previous = mouth

    finally:
        capture.release()

    if not motions:
        return {
            "available": False,
            "reason": "no usable face/mouth samples",
        }

    return {
        "available": True,
        "mean_mouth_motion": round(
            float(np.mean(motions)),
            6,
        ),
        "samples": len(motions),
    }


def analyze_video_cross_modal(
    video_path,
) -> dict:
    """Analyze audio/mouth-motion consistency in a video."""

    path = Path(video_path)

    audio = _audio_activity(path)
    mouth = _mouth_motion(path)

    result = {
        "cross_modal_grounding": True,
        "audio_available": audio.get(
            "available",
            False,
        ),
        "mouth_motion_available": mouth.get(
            "available",
            False,
        ),
        "cross_modal_method":
            "audio speech activity + facial "
            "mouth-motion consistency",
    }

    if (
        not audio.get("available")
        or not mouth.get("available")
    ):
        result["cross_modal_status"] = "PARTIAL"

        result["cross_modal_note"] = (
            audio.get("reason")
            or mouth.get("reason")
        )

        return result

    rms = audio["rms"]

    mouth_motion = mouth[
        "mean_mouth_motion"
    ]

    mismatch = bool(
        rms > 0.03
        and mouth_motion < 0.01
    )

    result.update(
        {
            "cross_modal_status":
                "MISMATCH"
                if mismatch
                else "CONSISTENT",

            "audio_rms":
                rms,

            "mouth_motion":
                mouth_motion,

            "audio_lip_mismatch":
                mismatch,

            "cross_modal_risk":
                70
                if mismatch
                else 0,
        }
    )

    return result
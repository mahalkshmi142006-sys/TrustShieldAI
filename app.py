from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from pathlib import Path

import sys
import json
import time

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


if str(ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT)
    )


# ============================================================
# TRUSTSHIELD AI MODULES
# ============================================================

from hallucination.engine import (
    analyze_answer
)

from deepfake.forensics import (
    analyze_media
)

from identity.verifier import (
    verify_faces
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(

    __name__,

    template_folder="templates",

    static_folder="static"
)


app.config[
    "MAX_CONTENT_LENGTH"
] = (
    100
    *
    1024
    *
    1024
)


# ============================================================
# PATHS
# ============================================================

EVIDENCE_PATH = (
    ROOT
    /
    "data"
    /
    "evidence.json"
)


UPLOADS_PATH = (
    ROOT
    /
    "uploads"
)


UPLOADS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EVIDENCE COUNT
# ============================================================

def evidence_count():

    try:

        data = json.loads(

            EVIDENCE_PATH.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(
            data,
            list
        ):

            return len(data)

        if isinstance(
            data,
            dict
        ):

            if "evidence" in data:

                return len(
                    data["evidence"]
                )

            return len(data)

        return 0

    except Exception:

        return 0


# ============================================================
# LANDING PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def landing():

    return render_template(
        "landing.html"
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================

@app.route(
    "/dashboard",
    methods=["GET"]
)
def dashboard():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH API
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "online",

        "service":
            "TrustShieldAI",

        "version":
            "3.3",

        "evidence_count":
            evidence_count(),

        "modules": {

            "llm_hallucination":
                True,

            "deepfake_verification":
                True,

            "evidence_verification":
                True,

            "identity_verification":
                True,

            "deepfake_classifier":
                True
        },

        "timestamp":
            time.time()
    })


# ============================================================
# HALLUCINATION EVALUATION
# ============================================================

@app.route(
    "/api/hallucination-evaluation",
    methods=["GET"]
)
def hallucination_evaluation():

    eval_path = (
        ROOT
        /
        "data"
        /
        "hallucination_eval.json"
    )

    try:

        cases = json.loads(

            eval_path.read_text(
                encoding="utf-8"
            )
        )

        y_true = []
        y_pred = []

        for case in cases:

            result = analyze_answer(

                case["question"],

                case["response"],

                EVIDENCE_PATH
            )

            if result[
                "contradicted"
            ] > 0:

                prediction = (
                    "CONTRADICTED"
                )

            elif (
                result["supported"]
                ==
                result["total_claims"]
            ):

                prediction = (
                    "SUPPORTED"
                )

            else:

                prediction = (
                    "UNCERTAIN"
                )

            y_true.append(
                case["expected"]
            )

            y_pred.append(
                prediction
            )

        labels = [

            "SUPPORTED",
            "CONTRADICTED",
            "UNCERTAIN"
        ]

        precision, recall, f1, _ = (
            precision_recall_fscore_support(

                y_true,
                y_pred,

                labels=labels,

                zero_division=0
            )
        )

        matrix = confusion_matrix(

            y_true,
            y_pred,

            labels=labels
        ).tolist()

        return jsonify({

            "dataset_size":
                len(cases),

            "accuracy":
                round(
                    float(
                        accuracy_score(
                            y_true,
                            y_pred
                        )
                    ),
                    4
                ),

            "macro_precision":
                round(
                    float(
                        precision.mean()
                    ),
                    4
                ),

            "macro_recall":
                round(
                    float(
                        recall.mean()
                    ),
                    4
                ),

            "macro_f1":
                round(
                    float(
                        f1.mean()
                    ),
                    4
                ),

            "labels":
                labels,

            "confusion_matrix":
                matrix,

            "engine_version":
                "3.0-hybrid-claim-verifier"
        })

    except Exception as error:

        return jsonify({

            "error":
                "Evaluation failed.",

            "details":
                str(error)

        }), 500


# ============================================================
# LLM HALLUCINATION ANALYSIS
# ============================================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():

    data = request.get_json(
        silent=True
    )

    if not isinstance(
        data,
        dict
    ):

        data = {}


    question = str(

        data.get(
            "question",
            ""
        )
    ).strip()


    response = str(

        data.get(
            "response",
            ""
        )
    ).strip()


    if not question:

        return jsonify({

            "error":
                "Question is required."

        }), 400


    if not response:

        return jsonify({

            "error":
                "AI response is required."

        }), 400


    try:

        result = analyze_answer(

            question,

            response,

            EVIDENCE_PATH
        )

        return jsonify(
            result
        )

    except Exception as error:

        return jsonify({

            "error":
                "LLM analysis failed.",

            "details":
                str(error)

        }), 500


# ============================================================
# DEEPFAKE / MEDIA ANALYSIS
# ============================================================


@app.route(
    "/api/analyze-media",
    methods=["POST"]
)
def media():

    if "file" not in request.files:

        return jsonify({

            "error":
                "Upload an image or video file."

        }), 400


    uploaded_file = (
        request.files["file"]
    )


    if not uploaded_file.filename:

        return jsonify({

            "error":
                "No file selected."

        }), 400


    filename = Path(
        uploaded_file.filename
    ).name


    file_path = (
        UPLOADS_PATH
        /
        filename
    )


    try:

        # ----------------------------------------------------
        # SAVE UPLOADED MEDIA
        # ----------------------------------------------------

        uploaded_file.save(
            file_path
        )


        # ----------------------------------------------------
        # EXISTING TRUSTSHIELD ANALYSIS
        # ----------------------------------------------------

        result = analyze_media(
            file_path
        )


        if not isinstance(
            result,
            dict
        ):

            result = {
                "result": result
            }


        result[
            "filename"
        ] = filename


        # ----------------------------------------------------
        # STAGE 3.6 EXPLAINABLE CLASSIFIER
        # ----------------------------------------------------

        extension = (
            file_path.suffix.lower()
        )


        image_extensions = {

            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"

        }


        if extension in image_extensions:

            try:

                import joblib

                from deepfake.model import (
                    feature_vector
                )


                stage35_model_path = (
                    ROOT
                    /
                    "deepfake"
                    /
                    "deepfake_rf_stage3_5.joblib"
                )


                if stage35_model_path.exists():

                    bundle = joblib.load(
                        stage35_model_path
                    )


                    # ----------------------------------------
                    # BUILD FEATURE VECTOR
                    # ----------------------------------------

                    vector = feature_vector(
                        result
                    )


                    import numpy as np


                    X = np.asarray(

                        [vector],

                        dtype=np.float32
                    )


                    model = bundle[
                        "model"
                    ]


                    probabilities = (
                        model.predict_proba(
                            X
                        )[0]
                    )


                    fake_probability = float(
                        probabilities[1]
                    )


                    authentic_probability = float(
                        probabilities[0]
                    )


                    # ----------------------------------------
                    # EXPLAINABLE RESULT
                    # ----------------------------------------

                    stage35_risk = round(
                        fake_probability * 100
                    )


                    stage35_authenticity = round(
                        authentic_probability * 100
                    )


                    if stage35_risk >= 70:

                        verdict = (
                            "HIGH RISK OF DEEPFAKE"
                        )

                    elif stage35_risk >= 40:

                        verdict = (
                            "SUSPICIOUS / NEEDS REVIEW"
                        )

                    else:

                        verdict = (
                            "LIKELY AUTHENTIC"
                        )


                    explanations = []


                    if result.get(
                        "face_count",
                        0
                    ) > 0:

                        explanations.append(
                            "Face detected for "
                            "face-region analysis."
                        )

                    else:

                        explanations.append(
                            "No face detected."
                        )


                    if result.get(
                        "face_fft_high_ratio"
                    ) is not None:

                        explanations.append(
                            "Face-region frequency "
                            "characteristics were analyzed."
                        )


                    if result.get(
                        "sharpness",
                        0
                    ) < 35:

                        explanations.append(
                            "Low image sharpness "
                            "was detected."
                        )

                    else:

                        explanations.append(
                            "Image sharpness was within "
                            "the analyzed range."
                        )


                    if result.get(
                        "contrast",
                        0
                    ) < 25:

                        explanations.append(
                            "Low image contrast "
                            "was detected."
                        )


                    if result.get(
                        "noise_residual_std",
                        0
                    ) < 1.5:

                        explanations.append(
                            "Low residual-noise "
                            "characteristics were detected."
                        )


                    # ----------------------------------------
                    # TOP MODEL FEATURES
                    # ----------------------------------------

                    feature_importance = (
                        bundle.get(
                            "feature_names",
                            []
                        )
                    )


                    importances = (
                        model.feature_importances_
                    )


                    ranked_features = sorted(

                        zip(
                            feature_importance,
                            importances
                        ),

                        key=lambda item:
                            item[1],

                        reverse=True
                    )


                    top_features = [

                        {
                            "feature":
                                name,

                            "importance":
                                round(
                                    float(value),
                                    6
                                )
                        }

                        for name, value
                        in ranked_features[:10]
                    ]


                    # ----------------------------------------
                    # ADD STAGE 3.6 INFORMATION
                    # ----------------------------------------

                    result.update({

                        "stage":
                            "Stage 3.6",

                        "explainable_verification":
                            True,

                        "stage35_classifier":
                            "RandomForestClassifier",

                        "stage35_model":
                            "deepfake_rf_stage3_5.joblib",

                        "stage35_test_accuracy":
                            0.9167,

                        "stage35_test_precision":
                            0.9032,

                        "stage35_test_recall":
                            0.9333,

                        "stage35_test_f1":
                            0.9180,

                        "stage35_test_roc_auc":
                            0.9667,

                        "stage35_test_size":
                            60,

                        "stage35_fake_probability":
                            round(
                                fake_probability,
                                4
                            ),

                        "stage35_authentic_probability":
                            round(
                                authentic_probability,
                                4
                            ),

                        "stage35_deepfake_risk":
                            stage35_risk,

                        "stage35_authenticity_score":
                            stage35_authenticity,

                        "stage35_verdict":
                            verdict,

                        "stage35_explanation":
                            explanations,

                        "stage35_top_features":
                            top_features,

                        "stage35_note":
                            (
                                "Prediction generated using "
                                "the Stage 3.5 Random Forest "
                                "trained on spatial and "
                                "frequency-domain forensic "
                                "features. Performance values "
                                "refer to the independent "
                                "60-image held-out test set."
                            )

                    })


                    # Use the Stage 3.5 classifier result
                    # as the primary image risk score.

                    result[
                        "deepfake_risk"
                    ] = stage35_risk


                    result[
                        "authenticity_score"
                    ] = stage35_authenticity


                    result[
                        "classifier_enabled"
                    ] = True


                    result[
                        "method"
                    ] = (
                        "Stage 3.6 Explainable "
                        "Random Forest verification "
                        "using Stage 3.5 model"
                    )


                else:

                    result[
                        "explainable_verification"
                    ] = False

                    result[
                        "stage35_status"
                    ] = (
                        "Stage 3.5 model not found."
                    )


            except Exception as model_error:

                result[
                    "explainable_verification"
                ] = False

                result[
                    "stage35_status"
                ] = (
                    "Stage 3.5 model could not "
                    "be loaded."
                )

                result[
                    "stage35_error"
                ] = str(
                    model_error
                )


        else:

            # ------------------------------------------------
            # VIDEO
            # ------------------------------------------------

            result[
                "explainable_verification"
            ] = False

            result[
                "stage35_status"
            ] = (
                "Stage 3.5 image classifier is "
                "not applied to video. Video uses "
                "the existing spatial, frequency "
                "and temporal forensic screening."
            )


        return jsonify(
            result
        )


    except Exception as error:

        return jsonify({

            "error":
                "Media analysis failed.",

            "details":
                str(error)

        }), 500


    finally:

        try:

            file_path.unlink()

        except OSError:

            pass


    if "file" not in request.files:

        return jsonify({

            "error":
                "Upload an image or video file."

        }), 400


    uploaded_file = (
        request.files["file"]
    )


    if not uploaded_file.filename:

        return jsonify({

            "error":
                "No file selected."

        }), 400


    filename = Path(
        uploaded_file.filename
    ).name


    file_path = (
        UPLOADS_PATH
        /
        filename
    )


    try:

        uploaded_file.save(
            file_path
        )

        result = analyze_media(
            file_path
        )

        if not isinstance(
            result,
            dict
        ):

            result = {
                "result": result
            }

        result[
            "filename"
        ] = filename

        return jsonify(
            result
        )

    except Exception as error:

        return jsonify({

            "error":
                "Media analysis failed.",

            "details":
                str(error)

        }), 500

    finally:

        try:

            file_path.unlink()

        except OSError:

            pass


# ============================================================
# IDENTITY VERIFICATION
# ============================================================

@app.route(
    "/api/verify-identity",
    methods=["POST"]
)
def identity():

    if (
        "reference"
        not in request.files
        or
        "probe"
        not in request.files
    ):

        return jsonify({

            "error":
                "Upload both reference and probe images."

        }), 400


    reference_file = (
        request.files["reference"]
    )

    probe_file = (
        request.files["probe"]
    )


    if (
        not reference_file.filename
        or
        not probe_file.filename
    ):

        return jsonify({

            "error":
                "Both images are required."

        }), 400


    reference_filename = (
        "reference_"
        +
        Path(
            reference_file.filename
        ).name
    )


    probe_filename = (
        "probe_"
        +
        Path(
            probe_file.filename
        ).name
    )


    reference_path = (
        UPLOADS_PATH
        /
        reference_filename
    )

    probe_path = (
        UPLOADS_PATH
        /
        probe_filename
    )


    try:

        reference_file.save(
            reference_path
        )

        probe_file.save(
            probe_path
        )

        result = verify_faces(

            reference_path,

            probe_path
        )

        return jsonify(
            result
        )

    except Exception as error:

        return jsonify({

            "error":
                "Identity verification failed.",

            "details":
                str(error)

        }), 500

    finally:

        for path in (
            reference_path,
            probe_path
        ):

            try:

                path.unlink()

            except OSError:

                pass


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "error":
            "TrustShieldAI page not found.",

        "available_pages": [

            "/",

            "/dashboard"

        ]

    }), 404


@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "error":
            "Uploaded file is too large. "
            "Maximum size is 100 MB."

    }), 413


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("                 TRUSTSHIELD AI")
    print("=" * 60)
    print()

    print(
        "Landing Page:"
    )

    print(
        "http://127.0.0.1:5000/"
    )

    print()

    print(
        "Dashboard:"
    )

    print(
        "http://127.0.0.1:5000/dashboard"
    )

    print()

    print(
        "Deepfake classifier:"
    )

    print(
        "Enabled when deepfake_rf.joblib exists."
    )

    print()

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False
    )
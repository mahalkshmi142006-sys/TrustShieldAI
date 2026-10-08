from deepfake.model import (
    FEATURE_NAMES,
    feature_vector
)


def test_feature_vector_schema():

    features = {

        name:
            index + 1

        for index, name
        in enumerate(
            FEATURE_NAMES
        )
    }

    vector = feature_vector(
        features
    )

    assert len(vector) == len(
        FEATURE_NAMES
    )

    assert vector[0] == 1.0

    assert vector[-1] == float(
        len(FEATURE_NAMES)
    )
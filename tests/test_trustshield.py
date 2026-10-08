from pathlib import Path

from hallucination.engine import analyze_answer


E = Path(__file__).parents[1] / "data" / "evidence.json"


def test_supported():
    result = analyze_answer(
        "What is the largest planet?",
        "Jupiter is the largest planet in the Solar System.",
        E
    )

    assert result["supported"] >= 1
    assert result["hallucination_risk"] < 50


def test_mismatch():
    result = analyze_answer(
        "What is the smallest planet?",
        "Jupiter is the largest planet in the Solar System.",
        E
    )

    assert result["question_answer_consistent"] is False


def test_water():
    result = analyze_answer(
        "What is the formula for water?",
        "Water has the chemical formula H2O.",
        E
    )

    assert result["supported"] >= 1


def test_saturn_jupiter_contradiction():
    result = analyze_answer(
        "What is the smallest planet in the Solar System?",
        "Jupiter is the largest planet in the Solar System.",
        E
    )

    assert result["question_answer_consistent"] is False
    assert result["hallucination_risk"] >= 70
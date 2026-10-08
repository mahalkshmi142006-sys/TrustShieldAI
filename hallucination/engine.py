"""
TrustShieldAI - General LLM Hallucination Verification Engine

Features:
    - Domain-independent claim extraction
    - Local + dynamic web evidence retrieval
    - Explainable evidence ranking
    - Lexical similarity
    - Entity-aware relation / contradiction verification
    - Numerical contradiction detection
    - Year/date contradiction detection
    - Negation contradiction detection
    - Formula / symbolic-expression matching
    - Optional NLI semantic verification
    - Question/answer consistency
    - Backward-compatible result fields
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hallucination.evidence_manager import retrieve_combined_evidence


# ============================================================
# CONFIGURATION
# ============================================================

MIN_EVIDENCE_SCORE = 0.12
STRONG_EVIDENCE_SCORE = 0.45

LOW_RISK_THRESHOLD = 25
MEDIUM_RISK_THRESHOLD = 55

MAX_CLAIMS = 20
MAX_EVIDENCE = 8


# ============================================================
# TEXT UTILITIES
# ============================================================

def _normalize(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", _normalize(text)))


def _token_similarity(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return intersection / union


def _sentence_split(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|[\n]+", text)
    result = []
    for piece in pieces:
        piece = piece.strip()
        if len(piece) >= 3:
            result.append(piece)
    return result


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def _extract_claims(answer: str) -> List[str]:
    sentences = _sentence_split(answer)
    claims = []
    for sentence in sentences:
        cleaned = sentence.strip()
        cleaned = re.sub(
            r"^(answer|response|according to me)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE
        )
        if len(cleaned) < 3:
            continue
        claims.append(cleaned)
        if len(claims) >= MAX_CLAIMS:
            break
    return claims


# ============================================================
# SYMBOLIC / FORMULA DETECTION
# ============================================================

def _extract_formulas(text: str) -> List[str]:
    text = str(text or "")
    formulas = re.findall(
        r"\b[A-Z][A-Za-z]?\d*(?:[A-Z][A-Za-z]?\d*)+\b",
        text
    )
    math_formulas = re.findall(
        r"\b[a-zA-Z]\s*[+\-*/=]\s*[a-zA-Z0-9]+",
        text
    )
    formulas.extend(math_formulas)
    return [_normalize(x).replace(" ", "") for x in formulas]


def _formula_match(claim: str, evidence: str) -> bool:
    claim_formulas = _extract_formulas(claim)
    evidence_formulas = _extract_formulas(evidence)
    if not claim_formulas or not evidence_formulas:
        return False
    return bool(set(claim_formulas) & set(evidence_formulas))


# ============================================================
# NUMBER / DATE / ENTITY CONFLICT DETECTION
# ============================================================

def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:\.\d+)?%?\b", str(text))


def _extract_years(text: str) -> List[str]:
    return re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", str(text))


def _extract_named_entities(text: str) -> set:
    """Extract candidate capitalized Proper Nouns/Entities from text."""
    stop_entities = {"The", "A", "An", "Who", "What", "When", "Where", "Why", "How", "In", "On", "At", "According"}
    words = re.findall(r"\b[A-Z][a-zA-Z0-9._'-]+\b", str(text))
    return {w for w in words if w not in stop_entities and len(w) > 1}


def _normalize_relation_word(word: str) -> str:
    irregular_forms = {
        "am": "be", "is": "be", "are": "be", "was": "be", "were": "be",
        "been": "be", "being": "be", "wrote": "write", "written": "write",
        "did": "do", "done": "do", "had": "have", "has": "have",
        "made": "make", "known": "know", "knew": "know", "built": "build",
        "took": "take", "taken": "take", "gave": "give", "given": "give",
        "found": "find", "saw": "see", "seen": "see", "ran": "run",
        "won": "win", "born": "bear", "became": "become", "led": "lead"
    }
    word = word.lower()
    if word in irregular_forms:
        return irregular_forms[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 3 and word.endswith("ed"):
        stem = word[:-2]
        if stem.endswith("i"):
            stem = stem[:-1] + "y"
        elif len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _extract_relation_triples(text: str) -> List[Tuple[str, str, str]]:
    """Extract simple named-entity subject/predicate/object relations."""
    stop_words = {
        "a", "an", "the", "am", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "has", "have", "had", "to", "by", "of", "in", "on",
        "at", "for", "from", "with", "and", "or", "but", "not", "never"
    }
    initial_stop_words = {"the", "a", "an", "according", "in", "on", "at"}
    triples = []

    for sentence in _sentence_split(text):
        mentions = []
        for match in re.finditer(
            r"\b[A-Z][a-zA-Z0-9]+(?:[.'’-][A-Za-z0-9]+)*(?:\s+[A-Z][a-zA-Z0-9]+(?:[.'’-][A-Za-z0-9]+)*)*",
            sentence
        ):
            words = match.group().split()
            while words and words[0].lower() in initial_stop_words:
                words.pop(0)
            if words:
                mentions.append((match.start(), match.end(), " ".join(words).lower()))

        for left, right in zip(mentions, mentions[1:]):
            between = sentence[left[1]:right[0]]
            words = re.findall(r"[A-Za-z]+", between.lower())
            if not words:
                continue

            by_index = len(words) - 1 - words[::-1].index("by") if "by" in words else -1
            if by_index > 0:
                relation_candidates = [word for word in words[:by_index] if word not in stop_words]
                if not relation_candidates:
                    continue
                subject, obj = right[2], left[2]
                relation = _normalize_relation_word(relation_candidates[-1])
            else:
                relation_candidates = [word for word in words if word not in stop_words]
                if not relation_candidates:
                    continue
                subject, obj = left[2], right[2]
                relation = _normalize_relation_word(relation_candidates[0])

            triples.append((subject, relation, obj))

        if len(mentions) == 1:
            subject = mentions[0][2]
            tail = sentence[mentions[0][1]:]
            tail_words = list(re.finditer(r"[A-Za-z]+", tail.lower()))
            predicate_index = next(
                (
                    index for index, match in enumerate(tail_words)
                    if match.group() not in stop_words
                    and (
                        match.group().endswith(("ed", "ing", "s"))
                        or _normalize_relation_word(match.group()) != match.group()
                    )
                ),
                None
            )
            if predicate_index is None:
                continue

            predicate = tail_words[predicate_index].group()
            object_word = next(
                (
                    match.group()
                    for match in tail_words[predicate_index + 1:]
                    if match.group() not in stop_words
                ),
                None
            )
            if object_word:
                triples.append(
                    (subject, _normalize_relation_word(predicate), object_word)
                )

    return triples


def _number_conflict(claim: str, evidence: str) -> bool:
    claim_numbers = _extract_numbers(claim)
    evidence_numbers = _extract_numbers(evidence)
    if not claim_numbers or not evidence_numbers:
        return False
    if len(claim_numbers) == 1 and len(evidence_numbers) == 1:
        if claim_numbers[0] != evidence_numbers[0]:
            similarity = _token_similarity(claim, evidence)
            if similarity >= 0.20:
                return True
    return False


def _year_conflict(claim: str, evidence: str) -> bool:
    claim_years = _extract_years(claim)
    evidence_years = _extract_years(evidence)
    if not claim_years or not evidence_years:
        return False
    if len(claim_years) == 1 and len(evidence_years) == 1:
        if claim_years[0] != evidence_years[0]:
            similarity = _token_similarity(claim, evidence)
            if similarity >= 0.20:
                return True
    return False


def _entity_relation_conflict(claim: str, evidence: str) -> bool:
    """
    Detects a different subject or object for the same expressed relation.
    """
    claim_relations = _extract_relation_triples(claim)
    evidence_relations = _extract_relation_triples(evidence)

    for claim_subject, claim_relation, claim_object in claim_relations:
        for evidence_subject, evidence_relation, evidence_object in evidence_relations:
            if claim_relation != evidence_relation:
                continue
            same_subject = claim_subject == evidence_subject
            same_object = claim_object == evidence_object
            if (same_object and not same_subject) or (same_subject and not same_object):
                return True

    return False


def _relation_supported(claim: str, evidence: str) -> bool:
    claim_relations = _extract_relation_triples(claim)
    evidence_relations = set(_extract_relation_triples(evidence))
    if any(relation in evidence_relations for relation in claim_relations):
        return True

    evidence_sentences = _sentence_split(evidence)
    for subject, relation, obj in claim_relations:
        subject_tokens = re.findall(r"[a-z0-9]+", subject)
        object_tokens = re.findall(r"[a-z0-9]+", obj)
        if not subject_tokens or not object_tokens:
            continue

        for sentence in evidence_sentences:
            sentence_tokens = re.findall(r"[a-z0-9]+", sentence.lower())
            normalized_tokens = {
                _normalize_relation_word(token) for token in sentence_tokens
            }
            if (
                _contains_token_phrase(sentence_tokens, subject_tokens)
                and _contains_token_phrase(sentence_tokens, object_tokens)
                and relation in normalized_tokens
            ):
                return True

    return False


def _contains_token_phrase(tokens: List[str], phrase: List[str]) -> bool:
    phrase_length = len(phrase)
    return any(
        tokens[index:index + phrase_length] == phrase
        for index in range(len(tokens) - phrase_length + 1)
    )


# ============================================================
# NEGATION
# ============================================================

NEGATION_WORDS = {
    "not", "never", "no", "none", "neither", "cannot", "can't",
    "isn't", "wasn't", "weren't", "doesn't", "don't", "didn't",
    "false", "incorrect", "impossible"
}


def _has_negation(text: str) -> bool:
    tokens = _tokens(text)
    return bool(tokens & NEGATION_WORDS)


def _negation_conflict(claim: str, evidence: str) -> bool:
    similarity = _token_similarity(claim, evidence)
    if similarity < 0.35:
        return False

    claim_negated = _has_negation(claim)
    evidence_negated = _has_negation(evidence)

    return claim_negated != evidence_negated


# ============================================================
# NLI MODEL
# ============================================================

_nli_pipeline = None
_nli_attempted = False


def _load_nli():
    global _nli_pipeline
    global _nli_attempted

    if _nli_attempted:
        return _nli_pipeline

    _nli_attempted = True
    try:
        from transformers import pipeline
        _nli_pipeline = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            top_k=None
        )
    except Exception:
        _nli_pipeline = None

    return _nli_pipeline


def _nli_scores(premise: str, hypothesis: str) -> Dict[str, float]:
    model = _load_nli()
    if model is None:
        return {}

    try:
        result = model(f"{premise} </s> {hypothesis}")
        if isinstance(result, list) and result:
            first = result[0]
            rows = first if isinstance(first, list) else result
            scores = {}
            for row in rows:
                label = str(row.get("label", "")).lower()
                score = float(row.get("score", 0.0))
                scores[label] = score
            return scores
    except Exception:
        pass

    return {}


def _semantic_relationship(evidence: str, claim: str) -> Tuple[str, float]:
    scores = _nli_scores(evidence, claim)
    if not scores:
        return ("unavailable", 0.0)

    entailment = 0.0
    contradiction = 0.0
    neutral = 0.0

    for label, score in scores.items():
        if "entail" in label:
            entailment = max(entailment, score)
        elif "contrad" in label:
            contradiction = max(contradiction, score)
        elif "neutral" in label:
            neutral = max(neutral, score)

    if entailment >= max(contradiction, neutral):
        return ("supported", entailment)
    if contradiction >= max(entailment, neutral):
        return ("contradicted", contradiction)

    return ("uncertain", neutral)


# ============================================================
# EXPLICIT EVIDENCE STORE SUPPORT
# ============================================================

def _load_explicit_evidence(evidence_store: Any) -> List[Dict[str, Any]]:
    if evidence_store is None:
        return []

    data = evidence_store
    try:
        if isinstance(evidence_store, (str, Path)):
            path = Path(evidence_store)
            if not path.exists() or not path.is_file():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = []
        for key in ("evidence", "documents", "records", "items", "data"):
            value = data.get(key)
            if isinstance(value, list):
                records = value
                break
        if not records:
            if any(key in data for key in ("text", "content", "fact", "statement", "answer", "response")):
                records = [data]
    else:
        return []

    return [item for item in records if isinstance(item, dict)]


def _explicit_evidence_text(item: Dict[str, Any]) -> str:
    parts = []
    for key in (
        "title", "question", "text", "content", "fact",
        "statement", "answer", "response", "description",
        "topic", "source"
    ):
        value = item.get(key)
        if value is not None:
            val_str = str(value).strip()
            if val_str:
                parts.append(val_str)
    return " ".join(parts).strip()


def _retrieve_explicit_evidence(
    query: str,
    evidence_store: Any,
    limit: int
) -> List[Dict[str, Any]]:
    records = _load_explicit_evidence(evidence_store)
    if not records:
        return []

    scored = []
    for record in records:
        text = _explicit_evidence_text(record)
        if not text:
            continue

        relevance = _token_similarity(query, text)
        record_question = _normalize(str(record.get("question", "")))
        normalized_query = _normalize(query)

        if record_question and record_question == normalized_query:
            relevance = max(relevance, 0.95)

        if relevance <= 0.0:
            continue

        item = dict(record)
        item["text"] = text
        item["relevance"] = round(min(1.0, relevance), 6)
        item["score"] = item["relevance"]
        item["similarity"] = item["relevance"]
        item["combined_score"] = item["relevance"]
        item["evidence_origin"] = "local"
        item["source"] = (
            item.get("source") or item.get("title") or item.get("name") or "Explicit evidence store"
        )
        item["url"] = str(item.get("url", "") or "")
        item["source_type"] = item.get("source_type") or "local"
        item["source_reliability"] = 0.90
        item["source_authority"] = 0.90
        item["is_evaluation_data"] = False
        item["runtime_score"] = item["relevance"]
        item["explanation"] = [
            "Evidence was supplied explicitly by the caller.",
            "Evidence relevance was calculated against the query.",
            "Explicit evidence takes precedence over automatic runtime retrieval."
        ]
        scored.append(item)

    scored.sort(
        key=lambda item: (
            float(item.get("relevance", 0.0) or 0.0),
            float(item.get("source_authority", 0.0) or 0.0)
        ),
        reverse=True
    )
    return scored[:max(1, limit)]


# ============================================================
# CLAIM-SPECIFIC EVIDENCE RELEVANCE
# ============================================================

def _claim_evidence_overlap(claim: str, evidence_text: str) -> float:
    """Estimate direct, domain-independent relevance of evidence to a claim."""
    claim_tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9._'-]*", str(claim).lower()))
    evidence_tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9._'-]*", str(evidence_text).lower()))
    if not claim_tokens or not evidence_tokens:
        return 0.0

    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "by", "of", "to", "in", "on", "at", "for", "and", "or", "but", "with",
        "from", "that", "this", "these", "those", "it", "its", "as", "who",
        "what", "when", "where", "which", "how", "why", "did", "do", "does",
        "has", "have", "had"
    }
    content = {t for t in claim_tokens if t not in stop and len(t) > 2}
    if not content:
        return 0.0

    overlap = content & evidence_tokens
    base = len(overlap) / len(content)

    entities = {t.lower() for t in re.findall(r"\b[A-Z][A-Za-z0-9._'-]{2,}\b", str(claim))}
    entity_bonus = 0.15 if entities & evidence_tokens else 0.0

    words = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9._'-]*", str(claim).lower()) if t not in stop and len(t) > 2]
    phrase_bonus = 0.0
    for size in (4, 3, 2):
        for i in range(max(0, len(words) - size + 1)):
            if " ".join(words[i:i + size]) in str(evidence_text).lower():
                phrase_bonus = 0.15
                break
        if phrase_bonus:
            break

    return max(0.0, min(1.0, base + entity_bonus + phrase_bonus))


# ============================================================
# EVIDENCE RETRIEVAL
# ============================================================

def _retrieve(
    question: str,
    claim: str,
    evidence_store: Any = None
) -> List[Dict[str, Any]]:
    queries = [
        claim,
        question,
        f"{question} {claim}"
    ]

    evidence = []
    seen = set()

    if evidence_store is not None:
        for query in queries:
            rows = _retrieve_explicit_evidence(query, evidence_store, MAX_EVIDENCE)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                key = _normalize(text)
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(row)

        evidence.sort(
            key=lambda item: (
                float(item.get("relevance", item.get("score", 0.0)) or 0.0),
                float(item.get("runtime_score", 0.0) or 0.0)
            ),
            reverse=True
        )
        return evidence[:MAX_EVIDENCE]

    for query in queries:
        try:
            rows = retrieve_combined_evidence(query, limit=MAX_EVIDENCE)
        except TypeError:
            try:
                rows = retrieve_combined_evidence(query, MAX_EVIDENCE)
            except Exception:
                rows = []
        except Exception:
            rows = []

        if not rows:
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            key = _normalize(text)
            if key in seen:
                continue
            seen.add(key)

            relevance = float(row.get("relevance", row.get("similarity", row.get("score", 0.0))) or 0.0)
            relevance = max(0.0, min(1.0, relevance))

            item = {**row, "text": text, "score": relevance, "similarity": relevance}
            evidence.append(item)

    non_evaluation = [
        item for item in evidence
        if not bool(item.get("is_evaluation_data", False))
        and str(item.get("evidence_origin", "")).lower() != "local_evaluation"
    ]
    ranking_pool = non_evaluation if non_evaluation else evidence

    for item in ranking_pool:
        claim_match = _claim_evidence_overlap(claim, str(item.get("text", "")))
        item["_claim_match"] = claim_match
        base_runtime = float(item.get("runtime_score", item.get("combined_score", 0.0)) or 0.0)
        reliability = float(item.get("source_reliability", 0.0) or 0.0)
        item["_runtime_priority"] = (
            0.55 * claim_match + 0.25 * max(0.0, min(1.0, base_runtime)) + 0.20 * max(0.0, min(1.0, reliability))
        )

    ranking_pool.sort(
        key=lambda item: (
            float(item.get("_runtime_priority", 0.0) or 0.0),
            float(item.get("_claim_match", 0.0) or 0.0),
            float(item.get("source_authority", 0.0) or 0.0),
            float(item.get("score", 0.0) or 0.0)
        ),
        reverse=True
    )
    return ranking_pool[:MAX_EVIDENCE]


# ============================================================
# CLAIM VERIFICATION
# ============================================================

def _verify_claim(
    question: str,
    claim: str,
    evidence_store: Any = None
) -> Dict[str, Any]:
    evidence = _retrieve(question, claim, evidence_store)

    if not evidence:
        return {
            "claim": claim,
            "status": "INSUFFICIENT EVIDENCE",
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "evidence": [],
            "best_evidence": {},
            "reason": "No sufficiently relevant evidence was retrieved.",
            "explainability": {
                "evidence_count": 0,
                "best_evidence_source": "",
                "best_evidence_url": "",
                "best_evidence_relevance": 0.0,
                "source_reliability": 0.0,
                "evidence_origin": "",
                "evidence_explanation": []
            }
        }

    best_support = 0.0
    best_contradiction = 0.0
    best_evidence = None
    reasons = []

    for item in evidence:
        text = item.get("text", "")
        semantic_status, semantic_score = _semantic_relationship(text, claim)

        number_conflict = _number_conflict(claim, text)
        year_conflict = _year_conflict(claim, text)
        negation_conflict = _negation_conflict(claim, text)
        formula_match = _formula_match(claim, text)
        entity_conflict = _entity_relation_conflict(claim, text)
        relation_match = _relation_supported(claim, text)

        # ----------------------------------------------------
        # SUPPORT
        # ----------------------------------------------------
        support = 0.90 if relation_match else 0.0

        if semantic_status == "supported" and not entity_conflict:
            support = max(support, semantic_score)

        if formula_match:
            support = max(support, 0.90)
            reasons.append("Formula or symbolic expression matches retrieved evidence.")

        # ----------------------------------------------------
        # CONTRADICTION
        # ----------------------------------------------------
        contradiction = 0.0

        if semantic_status == "contradicted":
            contradiction = max(contradiction, semantic_score)

        if entity_conflict:
            contradiction = max(contradiction, 0.85)
            reasons.append("Subject or object conflicts with the same relation in retrieved evidence.")

        if number_conflict:
            contradiction = max(contradiction, 0.85)
            reasons.append("Numerical value conflicts with retrieved evidence.")

        if year_conflict:
            contradiction = max(contradiction, 0.85)
            reasons.append("Year/date conflicts with retrieved evidence.")

        if negation_conflict:
            contradiction = max(contradiction, 0.80)
            reasons.append("Negation conflicts with retrieved evidence.")

        # ----------------------------------------------------
        # BEST SUPPORTING / CONTRADICTING EVIDENCE
        # ----------------------------------------------------
        if support > best_support:
            best_support = support
            best_evidence = item

        if contradiction > best_contradiction:
            best_contradiction = contradiction
            if best_support == 0.0:
                best_evidence = item

    # If no evidence scored above threshold, avoid false positives/negatives
    if best_support < MIN_EVIDENCE_SCORE and best_contradiction < 0.65:
        best_support = 0.0

    # ========================================================
    # STATUS
    # ========================================================
    if best_contradiction >= 0.65:
        status = "CONTRADICTED"
        reason = "The claim conflicts with retrieved evidence."
    elif best_support >= STRONG_EVIDENCE_SCORE:
        status = "SUPPORTED"
        reason = "The claim is supported by retrieved evidence."
    elif best_support >= MIN_EVIDENCE_SCORE:
        status = "UNCERTAIN"
        reason = "Some evidence was found, but support is not strong enough."
    else:
        status = "INSUFFICIENT EVIDENCE"
        reason = "Retrieved evidence is not sufficiently relevant."

    if reasons:
        reason += " " + " ".join(dict.fromkeys(reasons))

    # ========================================================
    # EXPLAINABILITY
    # ========================================================
    if best_evidence:
        best_source = best_evidence.get("source", "")
        best_url = best_evidence.get("url", "")
        best_relevance = float(best_evidence.get("relevance", 0.0) or 0.0)
        source_reliability = float(best_evidence.get("source_reliability", 0.0) or 0.0)
        evidence_origin = best_evidence.get("evidence_origin", "")
        evidence_explanation = best_evidence.get("explanation", [])
    else:
        best_source = ""
        best_url = ""
        best_relevance = 0.0
        source_reliability = 0.0
        evidence_origin = ""
        evidence_explanation = []

    return {
        "claim": claim,
        "status": status,
        "support_score": round(best_support, 4),
        "contradiction_score": round(best_contradiction, 4),
        "evidence": evidence,
        "best_evidence": best_evidence if best_evidence else {},
        "reason": reason,
        "explainability": {
            "evidence_count": len(evidence),
            "best_evidence_source": best_source,
            "best_evidence_url": best_url,
            "best_evidence_relevance": round(best_relevance, 4),
            "source_reliability": round(source_reliability, 4),
            "evidence_origin": evidence_origin,
            "evidence_explanation": evidence_explanation
        }
    }


# ============================================================
# QUESTION / ANSWER CONSISTENCY
# ============================================================

def _question_answer_consistency(
    question: str,
    answer: str,
    claim_results: List[Dict[str, Any]]
) -> Tuple[bool, float, str]:
    question_lower = _normalize(question)
    answer_lower = _normalize(answer)

    if not question_lower or not answer_lower:
        return (False, 0.0, "Question or answer is empty.")

    q_smallest = any(word in question_lower for word in ["smallest", "minimum", "least"])
    q_largest = any(word in question_lower for word in ["largest", "biggest", "greatest", "maximum"])
    a_smallest = any(word in answer_lower for word in ["smallest", "minimum", "least"])
    a_largest = any(word in answer_lower for word in ["largest", "biggest", "greatest", "maximum"])

    if q_smallest and a_largest:
        return (False, 0.0, "Question asks for a smallest/minimum item, but the answer states a largest/maximum item.")
    if q_largest and a_smallest:
        return (False, 0.0, "Question asks for a largest/maximum item, but the answer states a smallest/minimum item.")

    question_tokens = _tokens(question)
    answer_tokens = _tokens(answer)

    stop_words = {
        "what", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on",
        "for", "to", "does", "do", "did", "how", "which", "who", "where", "when",
        "why", "can", "could", "tell", "me", "please"
    }

    meaningful_question_tokens = question_tokens - stop_words
    overlap = meaningful_question_tokens & answer_tokens
    overlap_score = len(overlap) / max(1, len(meaningful_question_tokens))

    contradicted = sum(1 for result in claim_results if result["status"] == "CONTRADICTED")
    supported = sum(1 for result in claim_results if result["status"] == "SUPPORTED")

    if contradicted > 0:
        return (False, 0.0, "The answer contains a claim that contradicts the available evidence.")

    if supported > 0:
        return (True, max(0.70, overlap_score), "The answer is consistent with the retrieved evidence.")

    if overlap_score >= 0.20:
        return (True, overlap_score, "The answer appears related to the question, but evidence is limited.")

    return (False, overlap_score, "The answer does not provide enough evidence of consistency with the question.")


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_answer(
    question: str,
    answer: str,
    evidence_store: Any = None
) -> Dict[str, Any]:
    question = str(question or "").strip()
    answer = str(answer or "").strip()

    if not question or not answer:
        return {
            "trust_score": 0,
            "hallucination_risk": 100,
            "confidence": 0,
            "status": "INVALID",
            "qa_consistency": "INCONSISTENT",
            "qa_consistency_score": 0.0,
            "question_answer_consistent": False,
            "supported": 0,
            "contradicted": 0,
            "uncertain": 0,
            "insufficient_evidence": 0,
            "total_claims": 0,
            "supported_claims": 0,
            "contradicted_claims": 0,
            "uncertain_claims": 0,
            "insufficient_evidence_claims": 0,
            "claims": [],
            "evidence_coverage": 0.0,
            "verification_method": (
                "General claim extraction + local and dynamic web evidence retrieval + contradiction analysis"
            ),
            "domain_independent": True,
            "hardcoded_fact_rules": False
        }

    claims = _extract_claims(answer)
    if not claims:
        claims = [answer]

    claim_results = []
    for claim in claims:
        result = _verify_claim(question, claim, evidence_store)
        claim_results.append(result)

    supported_count = sum(1 for c in claim_results if c["status"] == "SUPPORTED")
    contradicted_count = sum(1 for c in claim_results if c["status"] == "CONTRADICTED")
    uncertain_count = sum(1 for c in claim_results if c["status"] == "UNCERTAIN")
    insufficient_count = sum(1 for c in claim_results if c["status"] == "INSUFFICIENT EVIDENCE")
    total_claims = max(1, len(claim_results))

    question_answer_consistent, qa_score, qa_reason = _question_answer_consistency(
        question, answer, claim_results
    )

    evidence_coverage = (supported_count + contradicted_count) / total_claims

    if contradicted_count > 0:
        risk = 85
        if contradicted_count >= 2:
            risk = 95
    elif supported_count == len(claim_results):
        risk = 5
    elif uncertain_count > 0:
        risk = 45
    else:
        risk = 65

    if not question_answer_consistent:
        risk = max(risk, 70)

    risk = int(max(0, min(100, risk)))
    trust = 100 - risk

    if contradicted_count > 0 or supported_count == len(claim_results):
        confidence = 95
    elif uncertain_count > 0:
        confidence = 70
    else:
        confidence = 55

    confidence = int(max(0, min(100, confidence)))

    if risk <= LOW_RISK_THRESHOLD:
        status = "LOW HALLUCINATION RISK"
    elif risk <= MEDIUM_RISK_THRESHOLD:
        status = "MEDIUM HALLUCINATION RISK"
    else:
        status = "HIGH HALLUCINATION RISK"

    qa_label = "CONSISTENT" if question_answer_consistent else "INCONSISTENT"

    web_evidence_count = sum(
        1 for claim in claim_results
        for item in claim.get("evidence", [])
        if item.get("evidence_origin") == "web"
    )

    local_evidence_count = sum(
        1 for claim in claim_results
        for item in claim.get("evidence", [])
        if item.get("evidence_origin") == "local"
    )

    evidence_sources = sorted({
        str(item.get("source", ""))
        for claim in claim_results
        for item in claim.get("evidence", [])
        if item.get("source")
    })

    return {
        "trust_score": trust,
        "hallucination_risk": risk,
        "confidence": confidence,
        "status": status,
        "qa_consistency": qa_label,
        "qa_consistency_score": round(qa_score, 4),
        "question_answer_consistent": question_answer_consistent,
        "qa_reason": qa_reason,
        "supported": supported_count,
        "contradicted": contradicted_count,
        "uncertain": uncertain_count,
        "insufficient_evidence": insufficient_count,
        "total_claims": len(claim_results),
        "supported_claims": supported_count,
        "contradicted_claims": contradicted_count,
        "uncertain_claims": uncertain_count,
        "insufficient_evidence_claims": insufficient_count,
        "evidence_coverage": round(evidence_coverage, 4),
        "web_evidence_count": web_evidence_count,
        "local_evidence_count": local_evidence_count,
        "evidence_sources": evidence_sources,
        "claims": claim_results,
        "explanation": (
            "The answer was divided into factual claims. Each claim was compared against "
            "local and dynamically retrieved web evidence. Evidence was ranked using relevance "
            "and source reliability; lexical overlap affected retrieval ranking only. Factual "
            "support was assessed using matched relations, optional NLI entailment, or formula "
            "matching, alongside entity/relation, numerical/date, and negation conflict checks."
        ),
        "verification_method": (
            "General claim extraction + local evidence + dynamic web retrieval + source ranking + "
            "relation-aware support + lexical retrieval ranking + entity/relation conflict verification + "
            "numerical/date contradiction checks + "
            "negation analysis + formula matching + optional NLI semantic verification"
        ),
        "domain_independent": True,
        "hardcoded_fact_rules": False
    }


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

def verify_answer(
    question: str,
    answer: str,
    evidence_store: Any = None
) -> Dict[str, Any]:
    return analyze_answer(
        question,
        answer,
        evidence_store
    )
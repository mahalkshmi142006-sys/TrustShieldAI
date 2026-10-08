"""
TrustShieldAI - Evidence Manager

Combines local evidence and dynamically retrieved web evidence.

Design:
- Explicit evidence supplied by the caller is preserved.
- Runtime retrieval prefers reliable web evidence.
- Evaluation datasets are down-weighted for runtime ranking.
- Local evidence remains available as a fallback.
- Existing engine/test interfaces remain compatible.
"""

from __future__ import annotations

from typing import Any, Dict, List

from hallucination.evidence_store import retrieve_evidence
from hallucination.web_retriever import retrieve_web_evidence


DEFAULT_LIMIT = 8

# Evaluation data is useful for benchmarking, but should not dominate
# normal runtime evidence ranking.
EVALUATION_RELIABILITY_CAP = 0.20

AUTHORITATIVE_DOMAINS = {
    "gov",
    "gov.in",
    "nic.in",
    "edu",
    "ac.in",
    "who.int",
    "un.org",
    "nasa.gov",
    "nih.gov",
    "cdc.gov",
    "nature.com",
    "science.org",
    "britannica.com",
}

HIGH_AUTHORITY_SOURCE_TYPES = {
    "government",
    "official",
    "academic",
    "research",
    "scientific",
    "encyclopedia",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_domain(domain: str) -> str:
    if not domain:
        return ""

    domain = str(domain).lower().strip()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def _is_evaluation_data(item: Dict[str, Any]) -> bool:
    """
    Detect evaluation/benchmark evidence.

    This does NOT remove the evidence. It only labels it so runtime
    ranking can treat it appropriately.
    """
    if item.get("is_evaluation_data") is True:
        return True

    source = str(item.get("source", "")).lower()
    file_name = str(item.get("file", "")).lower()
    source_type = str(item.get("source_type", "")).lower()
    source_label = str(item.get("source_label", "")).lower()

    combined = " ".join(
        [
            source,
            file_name,
            source_type,
            source_label,
        ]
    )

    evaluation_terms = (
        "hallucination_eval",
        "evaluation_dataset",
        "evaluation",
        "benchmark",
    )

    return any(term in combined for term in evaluation_terms)


def _is_web_evidence(item: Dict[str, Any]) -> bool:
    origin = str(
        item.get("evidence_origin", "")
    ).lower()

    source_type = str(
        item.get("source_type", "")
    ).lower()

    if origin == "web":
        return True

    if source_type in {
        "web",
        "encyclopedia",
        "government",
        "official",
        "academic",
        "research",
        "scientific",
    }:
        return True

    return bool(item.get("url"))


def _relevance_score(item: Dict[str, Any]) -> float:
    """
    Get the original relevance/similarity score.
    """
    for key in (
        "relevance",
        "similarity",
        "search_score",
    ):
        if key in item:
            value = _safe_float(
                item.get(key),
                0.0,
            )

            if value > 0:
                return min(value, 1.0)

    return 0.0


def _authority_score(item: Dict[str, Any]) -> float:
    """
    Estimate source authority.

    This is source quality only. It does not determine whether
    the claim itself is true.
    """
    domain = _normalize_domain(
        str(item.get("domain", ""))
    )

    source_type = str(
        item.get("source_type", "")
    ).lower()

    if domain:
        for authoritative_domain in AUTHORITATIVE_DOMAINS:
            if (
                domain == authoritative_domain
                or domain.endswith(
                    "." + authoritative_domain
                )
            ):
                return 1.0

    if source_type in HIGH_AUTHORITY_SOURCE_TYPES:
        return 0.90

    if source_type == "encyclopedia":
        return 0.80

    if _is_web_evidence(item):
        return 0.65

    return 0.50


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def _normalize_evidence(
    item: Dict[str, Any],
    origin: str,
) -> Dict[str, Any]:
    """
    Normalize local/web evidence into one common structure.
    """
    result = dict(item)

    result["evidence_origin"] = origin

    result["text"] = str(
        result.get("text", "")
    )

    result["source"] = (
        result.get("source")
        or result.get("title")
        or "unknown"
    )

    result["source_label"] = (
        result.get("source_label")
        or result.get("title")
        or result.get("source")
        or "unknown"
    )

    result["url"] = str(
        result.get("url", "")
        or ""
    )

    result["domain"] = _normalize_domain(
        str(result.get("domain", ""))
    )

    result["source_type"] = (
        result.get("source_type")
        or "unknown"
    )

    result["relevance"] = round(
        _relevance_score(result),
        6,
    )

    result["source_authority"] = round(
        _authority_score(result),
        4,
    )

    evaluation = _is_evaluation_data(result)

    result["is_evaluation_data"] = evaluation

    # Preserve existing source reliability when supplied.
    reliability = _safe_float(
        result.get("source_reliability"),
        result["source_authority"],
    )

    if evaluation:
        reliability = min(
            reliability,
            EVALUATION_RELIABILITY_CAP,
        )

    result["source_reliability"] = round(
        max(0.0, min(reliability, 1.0)),
        4,
    )

    return result


# ---------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------

def _runtime_score(
    item: Dict[str, Any],
) -> float:
    """
    Runtime ranking score.

    IMPORTANT:
    This score ranks evidence. It does not itself decide whether
    a claim is true or false.
    """
    relevance = _relevance_score(item)

    authority = _safe_float(
        item.get("source_authority"),
        0.5,
    )

    reliability = _safe_float(
        item.get("source_reliability"),
        0.5,
    )

    score = (
        0.55 * relevance
        + 0.25 * authority
        + 0.20 * reliability
    )

    # Prefer dynamically retrieved web evidence.
    if _is_web_evidence(item):
        score += 0.10

    # Evaluation evidence is retained, but its runtime ranking is reduced.
    if item.get("is_evaluation_data"):
        score *= 0.35

    return round(
        max(0.0, min(score, 1.0)),
        6,
    )


def _add_explanation(
    item: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add explainability information.
    """
    relevance = _relevance_score(item)
    authority = _safe_float(
        item.get("source_authority"),
        0.5,
    )

    explanation = []

    if relevance >= 0.70:
        explanation.append(
            "Strong textual relevance with the claim/query."
        )
    elif relevance >= 0.35:
        explanation.append(
            "Moderate textual relevance with the claim/query."
        )
    else:
        explanation.append(
            "Weak lexical overlap with the claim/query."
        )

    if authority >= 0.90:
        explanation.append(
            "High-authority source."
        )
    elif authority >= 0.75:
        explanation.append(
            "Relatively strong source authority."
        )
    elif authority >= 0.60:
        explanation.append(
            "Moderate source authority."
        )
    else:
        explanation.append(
            "Source authority is limited."
        )

    if _is_web_evidence(item):
        explanation.append(
            "Dynamically retrieved web evidence."
        )
    else:
        explanation.append(
            "Local evidence retained as supporting evidence."
        )

    if item.get("is_evaluation_data"):
        explanation.append(
            "Evaluation data is retained for testing and down-weighted for runtime ranking."
        )

    item["explanation"] = explanation

    return item


# ---------------------------------------------------------------------
# Local retrieval
# ---------------------------------------------------------------------

def _retrieve_local(
    query: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Retrieve local evidence.
    """
    try:
        results = retrieve_evidence(
            query,
            limit,
        )
    except TypeError:
        results = retrieve_evidence(query)
    except Exception:
        return []

    if not results:
        return []

    if isinstance(results, dict):
        results = [results]

    normalized = []

    for item in results:
        if not isinstance(item, dict):
            continue

        normalized.append(
            _normalize_evidence(
                item,
                "local",
            )
        )

    return normalized


# ---------------------------------------------------------------------
# Web retrieval
# ---------------------------------------------------------------------

def _retrieve_web(
    query: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Retrieve dynamic web evidence.
    """
    try:
        results = retrieve_web_evidence(
            query,
            limit,
        )
    except TypeError:
        results = retrieve_web_evidence(query)
    except Exception:
        return []

    if not results:
        return []

    if isinstance(results, dict):
        results = [results]

    normalized = []

    for item in results:
        if not isinstance(item, dict):
            continue

        normalized.append(
            _normalize_evidence(
                item,
                "web",
            )
        )

    return normalized


# ---------------------------------------------------------------------
# Combined retrieval
# ---------------------------------------------------------------------

def retrieve_combined_evidence(
    query: str,
    limit: int = DEFAULT_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Retrieve local + web evidence and rank it for runtime use.

    Important compatibility rule:
    This function does NOT delete local evidence and does NOT treat
    evaluation data as false. It only ranks the evidence differently.
    """
    if not query or not str(query).strip():
        return []

    query = str(query).strip()

    retrieval_limit = max(
        limit * 2,
        8,
    )

    local_results = _retrieve_local(
        query,
        retrieval_limit,
    )

    web_results = _retrieve_web(
        query,
        retrieval_limit,
    )

    combined = local_results + web_results

    if not combined:
        return []

    # Remove exact duplicate text + URL combinations.
    unique = []
    seen = set()

    for item in combined:
        text = str(
            item.get("text", "")
        ).strip().lower()

        url = str(
            item.get("url", "")
        ).strip().lower()

        key = (
            text,
            url,
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    ranked = []

    for item in unique:
        item["runtime_score"] = _runtime_score(
            item
        )

        # Keep combined_score for engine compatibility.
        item["combined_score"] = item[
            "runtime_score"
        ]

        item = _add_explanation(item)

        ranked.append(item)

    ranked.sort(
        key=lambda x: (
            x.get("runtime_score", 0.0),
            x.get("source_authority", 0.0),
            x.get("relevance", 0.0),
        ),
        reverse=True,
    )

    return ranked[:limit]


# ---------------------------------------------------------------------
# Evidence summary
# ---------------------------------------------------------------------

def get_evidence_summary(
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Return a compact evidence summary.
    """
    if not evidence:
        return {
            "total": 0,
            "local": 0,
            "web": 0,
            "evaluation": 0,
            "runtime_web": 0,
            "authoritative_web": 0,
            "sources": 0,
            "best_source": "",
            "best_source_type": "",
        }

    local_count = 0
    web_count = 0
    evaluation_count = 0
    authoritative_web_count = 0

    sources = set()

    for item in evidence:
        if _is_web_evidence(item):
            web_count += 1

            if _safe_float(
                item.get("source_authority"),
                0.0,
            ) >= 0.90:
                authoritative_web_count += 1
        else:
            local_count += 1

        if item.get("is_evaluation_data"):
            evaluation_count += 1

        source = (
            item.get("source")
            or item.get("title")
            or item.get("domain")
        )

        if source:
            sources.add(str(source))

    best = evidence[0]

    return {
        "total": len(evidence),
        "local": local_count,
        "web": web_count,
        "evaluation": evaluation_count,
        "runtime_web": web_count,
        "authoritative_web": authoritative_web_count,
        "sources": len(sources),
        "best_source": (
            best.get("source")
            or best.get("title")
            or ""
        ),
        "best_source_type": (
            best.get("source_type")
            or ""
        ),
    }


# ---------------------------------------------------------------------
# Runtime evidence selection
# ---------------------------------------------------------------------

def get_runtime_evidence(
    evidence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Select evidence appropriate for runtime verification.

    Priority:
        1. Web evidence
        2. Local non-evaluation evidence
        3. Evaluation evidence as last resort

    This function is separate from retrieve_combined_evidence so that
    existing engine/test behavior is not broken.
    """
    if not evidence:
        return []

    web = [
        item
        for item in evidence
        if _is_web_evidence(item)
        and not item.get("is_evaluation_data")
    ]

    if web:
        return sorted(
            web,
            key=lambda x: x.get(
                "runtime_score",
                0.0,
            ),
            reverse=True,
        )

    local_non_eval = [
        item
        for item in evidence
        if not item.get("is_evaluation_data")
    ]

    if local_non_eval:
        return sorted(
            local_non_eval,
            key=lambda x: x.get(
                "runtime_score",
                0.0,
            ),
            reverse=True,
        )

    # Testing fallback.
    return sorted(
        evidence,
        key=lambda x: x.get(
            "runtime_score",
            0.0,
        ),
        reverse=True,
    )
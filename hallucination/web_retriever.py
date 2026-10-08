"""
TrustShieldAI - Dynamic Web Evidence Retriever

Purpose:
    Retrieve external evidence for LLM hallucination verification.

Design:
    - Uses Wikipedia's public REST API.
    - Does not depend on a fixed local fact database.
    - Returns source metadata.
    - Uses simple relevance scoring.
    - Fails safely when the network is unavailable.
    - Does NOT decide whether a claim is true or false.

The verification engine will perform that task later.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import quote

import requests


# ============================================================
# CONFIGURATION
# ============================================================

WIKIPEDIA_API = (
    "https://en.wikipedia.org/w/api.php"
)

WIKIPEDIA_SUMMARY = (
    "https://en.wikipedia.org/api/rest_v1/page/summary/"
)

DEFAULT_TIMEOUT = 10

DEFAULT_TOP_K = 5

USER_AGENT = (
    "TrustShieldAI/1.0 "
    "(LLM hallucination verification project)"
)


# ============================================================
# TEXT UTILITIES
# ============================================================

def _normalize(
    text: Any
) -> str:
    """
    Normalize text for comparison.
    """

    if text is None:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def _tokens(
    text: str
) -> set:
    """
    Extract simple alphanumeric tokens.
    """

    return set(
        re.findall(
            r"[a-zA-Z0-9]+",
            _normalize(text).lower()
        )
    )


def _similarity(
    query: str,
    text: str
) -> float:
    """
    Calculate Jaccard token similarity.
    """

    query_tokens = _tokens(
        query
    )

    text_tokens = _tokens(
        text
    )

    if not query_tokens or not text_tokens:
        return 0.0

    intersection = len(
        query_tokens
        &
        text_tokens
    )

    union = len(
        query_tokens
        |
        text_tokens
    )

    if union == 0:
        return 0.0

    return (
        intersection
        /
        union
    )


# ============================================================
# HTTP REQUEST
# ============================================================

def _request(
    url: str,
    params: Dict[str, Any] | None = None
) -> Dict[str, Any] | None:
    """
    Perform a safe HTTP GET request.

    Returns:
        Parsed JSON dictionary, or None if the request fails.
    """

    headers = {
        "User-Agent":
            USER_AGENT
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=DEFAULT_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(
            data,
            dict
        ):
            return data

    except (
        requests.RequestException,
        ValueError
    ):
        return None

    return None


# ============================================================
# WIKIPEDIA SEARCH
# ============================================================

def _search_wikipedia(
    query: str,
    limit: int = DEFAULT_TOP_K
) -> List[Dict[str, Any]]:
    """
    Search Wikipedia for pages relevant to a query.
    """

    query = _normalize(
        query
    )

    if not query:
        return []

    params = {

        "action":
            "query",

        "list":
            "search",

        "srsearch":
            query,

        "format":
            "json",

        "utf8":
            1,

        "srlimit":
            max(
                1,
                min(
                    int(limit),
                    10
                )
            )
    }

    data = _request(
        WIKIPEDIA_API,
        params
    )

    if not data:
        return []

    query_data = data.get(
        "query"
    )

    if not isinstance(
        query_data,
        dict
    ):
        return []

    search_results = query_data.get(
        "search",
        []
    )

    if not isinstance(
        search_results,
        list
    ):
        return []

    results = []

    for item in search_results:

        if not isinstance(
            item,
            dict
        ):
            continue

        title = _normalize(
            item.get(
                "title",
                ""
            )
        )

        snippet = _normalize(
            item.get(
                "snippet",
                ""
            )
        )

        if not title:
            continue

        # Wikipedia search snippets can contain HTML tags.
        snippet = re.sub(
            r"<[^>]+>",
            "",
            snippet
        )

        page_score = _similarity(
            query,
            title + " " + snippet
        )

        results.append({

            "title":
                title,

            "snippet":
                snippet,

            "search_score":
                round(
                    page_score,
                    6
                ),

            "source":
                "Wikipedia",

            "domain":
                "wikipedia.org",

            "source_type":
                "encyclopedia"
        })

    return results


# ============================================================
# WIKIPEDIA PAGE SUMMARY
# ============================================================

def _get_wikipedia_summary(
    title: str
) -> Dict[str, Any] | None:
    """
    Retrieve the summary of a Wikipedia page.
    """

    title = _normalize(
        title
    )

    if not title:
        return None

    encoded_title = quote(
        title.replace(
            " ",
            "_"
        )
    )

    url = (
        WIKIPEDIA_SUMMARY
        +
        encoded_title
    )

    data = _request(
        url
    )

    if not data:
        return None

    extract = _normalize(
        data.get(
            "extract",
            ""
        )
    )

    if not extract:
        return None

    page_url = ""

    content_urls = data.get(
        "content_urls"
    )

    if isinstance(
        content_urls,
        dict
    ):

        desktop = content_urls.get(
            "desktop"
        )

        if isinstance(
            desktop,
            dict
        ):

            page_url = _normalize(
                desktop.get(
                    "page",
                    ""
                )
            )

    if not page_url:

        page_url = (
            "https://en.wikipedia.org/wiki/"
            +
            quote(
                title.replace(
                    " ",
                    "_"
                )
            )
        )

    return {

        "title":
            _normalize(
                data.get(
                    "title",
                    title
                )
            ),

        "text":
            extract,

        "url":
            page_url,

        "source":
            "Wikipedia",

        "domain":
            "wikipedia.org",

        "source_type":
            "encyclopedia"
    }


# ============================================================
# PUBLIC WIKIPEDIA RETRIEVER
# ============================================================

def retrieve_wikipedia(
    query: str,
    top_k: int = DEFAULT_TOP_K
) -> List[Dict[str, Any]]:
    """
    Retrieve evidence from Wikipedia.

    The returned evidence contains:
        - text
        - title
        - URL
        - source
        - domain
        - source type
        - relevance score
    """

    query = _normalize(
        query
    )

    if not query:
        return []

    search_results = _search_wikipedia(
        query,
        limit=top_k
    )

    results = []

    seen_titles = set()

    for item in search_results:

        title = item.get(
            "title",
            ""
        )

        normalized_title = (
            title.lower()
        )

        if (
            not normalized_title
            or
            normalized_title in seen_titles
        ):
            continue

        seen_titles.add(
            normalized_title
        )

        summary = (
            _get_wikipedia_summary(
                title
            )
        )

        if summary is None:
            continue

        text = summary.get(
            "text",
            ""
        )

        relevance = _similarity(
            query,
            title + " " + text
        )

        result = {

            "text":
                text,

            "title":
                summary.get(
                    "title",
                    title
                ),

            "url":
                summary.get(
                    "url",
                    ""
                ),

            "source":
                "Wikipedia",

            "domain":
                "wikipedia.org",

            "source_type":
                "encyclopedia",

            "relevance":
                round(
                    relevance,
                    6
                ),

            "search_score":
                item.get(
                    "search_score",
                    0.0
                )
        }

        results.append(
            result
        )

    results.sort(
        key=lambda item: (
            float(
                item.get(
                    "relevance",
                    0.0
                )
            ),
            float(
                item.get(
                    "search_score",
                    0.0
                )
            )
        ),
        reverse=True
    )

    return results[
        :max(
            1,
            int(top_k)
        )
    ]


# ============================================================
# GENERAL PUBLIC RETRIEVER
# ============================================================

def retrieve_web_evidence(
    query: str,
    top_k: int = DEFAULT_TOP_K
) -> List[Dict[str, Any]]:
    """
    Retrieve external evidence.

    Current implementation:
        Wikipedia

    Future implementations can add:
        - search APIs
        - government sources
        - scientific databases
        - trusted news sources
        - official documentation
    """

    return retrieve_wikipedia(
        query,
        top_k
    )


# ============================================================
# SIMPLE CONNECTION TEST
# ============================================================

def web_retrieval_available() -> bool:
    """
    Check whether the configured web source is reachable.
    """

    try:

        results = retrieve_wikipedia(
            "Earth",
            top_k=1
        )

        return bool(
            results
        )

    except Exception:
        return False
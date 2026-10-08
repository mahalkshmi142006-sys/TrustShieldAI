from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".csv",
}


def _normalize(text: Any) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _tokens(text: str) -> set:
    return set(
        re.findall(
            r"[a-zA-Z0-9]+",
            _normalize(text).lower()
        )
    )


def _similarity(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)

    if not ta or not tb:
        return 0.0

    union = len(ta | tb)

    if union == 0:
        return 0.0

    return len(ta & tb) / union


def _read_text_file(path: Path) -> List[Dict[str, Any]]:
    results = []

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        text = _normalize(text)

        if text:
            results.append({
                "text": text,
                "source": str(
                    path.relative_to(PROJECT_ROOT)
                ),
                "file": path.name
            })

    except Exception:
        pass

    return results


def _read_csv_file(path: Path) -> List[Dict[str, Any]]:
    results = []

    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                parts = []

                for key, value in row.items():

                    if value is None:
                        continue

                    value = _normalize(value)

                    if value:
                        parts.append(
                            f"{key}: {value}"
                        )

                text = " | ".join(parts)

                if text:
                    results.append({
                        "text": text,
                        "source": str(
                            path.relative_to(PROJECT_ROOT)
                        ),
                        "file": path.name
                    })

    except Exception:
        pass

    return results


def _extract_json_text(value: Any) -> List[str]:
    results = []

    if isinstance(value, str):

        value = _normalize(value)

        if value:
            results.append(value)

    elif isinstance(value, dict):

        for key, item in value.items():

            key_text = _normalize(key)

            if isinstance(item, (dict, list)):
                results.extend(
                    _extract_json_text(item)
                )

            else:

                item_text = _normalize(item)

                if item_text:
                    results.append(
                        f"{key_text}: {item_text}"
                    )

    elif isinstance(value, list):

        for item in value:
            results.extend(
                _extract_json_text(item)
            )

    return results


def _read_json_file(path: Path) -> List[Dict[str, Any]]:
    results = []

    try:

        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            data = json.load(file)

        texts = _extract_json_text(data)

        for text in texts:

            if len(text) >= 3:
                results.append({
                    "text": text,
                    "source": str(
                        path.relative_to(PROJECT_ROOT)
                    ),
                    "file": path.name
                })

    except Exception:
        pass

    return results


def _discover_files() -> List[Path]:
    files = []
    seen = set()

    if not DATA_DIR.exists():
        return files

    try:

        for path in DATA_DIR.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            resolved = str(
                path.resolve()
            ).lower()

            if resolved in seen:
                continue

            seen.add(resolved)
            files.append(path)

    except Exception:
        pass

    return files


def _load_corpus() -> List[Dict[str, Any]]:
    corpus = []

    for path in _discover_files():

        suffix = path.suffix.lower()

        if suffix in {".txt", ".md"}:
            corpus.extend(
                _read_text_file(path)
            )

        elif suffix == ".csv":
            corpus.extend(
                _read_csv_file(path)
            )

        elif suffix == ".json":
            corpus.extend(
                _read_json_file(path)
            )

    return corpus


_CORPUS = None


def _get_corpus() -> List[Dict[str, Any]]:
    global _CORPUS

    if _CORPUS is None:
        _CORPUS = _load_corpus()

    return _CORPUS


def reload_evidence() -> int:
    global _CORPUS

    _CORPUS = _load_corpus()

    return len(_CORPUS)


def retrieve_evidence(
    query: str,
    top_k: int = 8
) -> List[Dict[str, Any]]:

    query = _normalize(query)

    if not query:
        return []

    corpus = _get_corpus()

    if not corpus:
        return []

    results = []

    for item in corpus:

        text = _normalize(
            item.get("text", "")
        )

        if not text:
            continue

        score = _similarity(
            query,
            text
        )

        if score <= 0:
            continue

        results.append({
            **item,
            "score": round(
                float(score),
                6
            ),
            "similarity": round(
                float(score),
                6
            )
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:max(1, int(top_k))]


def get_evidence_stats() -> Dict[str, Any]:

    corpus = _get_corpus()

    files = sorted({
        item.get("file", "")
        for item in corpus
        if item.get("file")
    })

    return {
        "entries": len(corpus),
        "files": len(files),
        "sources": files,
        "data_directory": str(DATA_DIR)
    }


def clear_cache() -> None:
    global _CORPUS
    _CORPUS = None
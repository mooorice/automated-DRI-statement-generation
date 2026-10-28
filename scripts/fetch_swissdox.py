#!/usr/bin/env python3
"""Submit, monitor, and download Swissdox queries from keywords.json."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from dotenv import load_dotenv
import os

API_BASE_URL = "https://swissdox.linguistik.uzh.ch/api"
DEFAULT_COLUMNS = [
    "id",
    "pubtime",
    "medium_code",
    "medium_name",
    "rubric",
    "regional",
    "doctype",
    "doctype_description",
    "language",
    "char_count",
    "dateline",
    "head",
    "subhead",
    "content_id",
    "content",
]
TERMINAL_STATUSES = {"FINISHED", "FAILED", "ERROR", "CANCELLED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swissdox API helper")
    parser.add_argument(
        "--keywords-file",
        default="projects/TA-Swiss/keywords.json",
        help=(
            "Path to keyword JSON. Supported: "
            "(1) flat list, "
            "(2) {lang: [keywords]}, "
            "(3) {'ai': {...}, 'education': {...}, 'work': {...}} for "
            "(AI AND education) OR (AI AND work)."
        ),
    )
    parser.add_argument("--from-date", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--to-date", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--languages",
        default="en,de,fr,it",
        help="Comma-separated languages to query and pull keywords from.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Swissdox query name (defaults to ta_swiss_<timestamp>).",
    )
    parser.add_argument(
        "--comment",
        default="AutoConcourse Swissdox retrieval from keywords.json",
        help="Optional Swissdox query comment.",
    )
    parser.add_argument(
        "--expiration-date",
        default=None,
        help="Optional expiration date YYYY-MM-DD for Swissdox storage.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=1_000_000,
        help="Swissdox maxResults.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Syntax check only. Swissdox will not execute retrieval.",
    )
    parser.add_argument(
        "--query-id",
        default=None,
        help="Existing query id. If set, skip submission and inspect this query.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll status until terminal state.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=20,
        help="Status polling interval in seconds.",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=180,
        help="Timeout for --wait.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download when query is finished.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for downloaded .tsv.xz (default: projects/<project_name>/data).",
    )
    return parser.parse_args()


def load_credentials() -> dict[str, str]:
    load_dotenv()
    api_key = os.getenv("liri_key")
    api_secret = os.getenv("liri_secret")
    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing credentials. Set both liri_key and liri_secret in .env "
            "(or environment variables)."
        )
    return {"X-API-Key": api_key, "X-API-Secret": api_secret}


def _dedupe_terms(terms: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for kw in terms:
        token = str(kw).strip()
        if token and token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def _select_terms(payload: list | dict, languages: list[str]) -> list[str]:
    if isinstance(payload, list):
        return _dedupe_terms(payload)
    if isinstance(payload, dict):
        selected: list[str] = []
        for language in languages:
            selected.extend(payload.get(language, []))
        return _dedupe_terms(selected)
    raise ValueError("Expected list or {lang: [terms]} map.")


def load_keywords(path: Path, languages: list[str]) -> dict:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    # New grouped format: build (AI AND education) OR (AI AND work)
    if isinstance(payload, dict) and all(k in payload for k in ("ai", "education", "work")):
        ai_terms = _select_terms(payload["ai"], languages)
        education_terms = _select_terms(payload["education"], languages)
        work_terms = _select_terms(payload["work"], languages)
        if not ai_terms or not education_terms or not work_terms:
            raise ValueError(
                "Grouped keyword format requires non-empty ai, education, and work "
                f"terms for languages {languages}."
            )
        return {
            "mode": "and_or",
            "ai": ai_terms,
            "education": education_terms,
            "work": work_terms,
        }

    # Legacy formats: flat OR keyword query
    terms = _select_terms(payload, languages)
    if not terms:
        raise ValueError("No keywords selected. Check --languages and keywords file.")
    return {"mode": "or", "keywords": terms}


def build_query_yaml(
    *,
    keyword_spec: dict,
    languages: list[str],
    from_date: str,
    to_date: str,
    max_results: int,
) -> str:
    if keyword_spec["mode"] == "and_or":
        content = {
            "OR": [
                {"AND": [{"OR": keyword_spec["ai"]}, {"OR": keyword_spec["education"]}]},
                {"AND": [{"OR": keyword_spec["ai"]}, {"OR": keyword_spec["work"]}]},
            ]
        }
    else:
        content = {"OR": keyword_spec["keywords"]}

    query_obj = {
        "query": {
            "dates": [{"from": from_date, "to": to_date}],
            "languages": languages,
            "content": content,
        },
        "result": {
            "format": "TSV",
            "maxResults": max_results,
            "columns": DEFAULT_COLUMNS,
        },
        "version": 1.2,
    }
    return yaml.safe_dump(query_obj, sort_keys=False, allow_unicode=True)


def submit_query(
    headers: dict[str, str],
    *,
    query_yaml: str,
    name: str,
    comment: str,
    expiration_date: str | None,
    test: bool,
) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/query",
        headers=headers,
        data={
            "query": query_yaml,
            "test": "1" if test else "0",
            "name": name,
            "comment": comment,
            "expirationDate": expiration_date or "",
        },
        timeout=120,
    )
    if not response.ok:
        details = response.text
        try:
            details = response.json()
        except Exception:
            pass
        raise requests.HTTPError(
            f"Swissdox query submission failed ({response.status_code}): {details}",
            response=response,
        )
    return response.json()


def normalize_status_payload(payload: dict | list, query_id: str | None = None) -> dict:
    """Normalize Swissdox status payloads to a single dict row.

    Some endpoints return a dict, others a list (sometimes with one entry).
    """
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        if not payload:
            raise RuntimeError("Empty status response from Swissdox API.")
        if query_id:
            match = next((row for row in payload if str(row.get("id")) == str(query_id)), None)
            if match is not None:
                return match
        return payload[0]
    raise RuntimeError(f"Unexpected status payload type: {type(payload).__name__}")


def get_status_by_id(headers: dict[str, str], query_id: str) -> dict:
    response = requests.get(f"{API_BASE_URL}/status/{query_id}", headers=headers, timeout=60)
    response.raise_for_status()
    return normalize_status_payload(response.json(), query_id=query_id)


def get_status_by_name(headers: dict[str, str], name: str) -> dict:
    response = requests.get(f"{API_BASE_URL}/status", headers=headers, timeout=60)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        rows = [normalize_status_payload(rows)]
    match = next((row for row in rows if row.get("name") == name), None)
    if match is None:
        raise RuntimeError(f"Query with name '{name}' not found in /status.")
    return match


def poll_until_terminal(
    headers: dict[str, str],
    *,
    query_id: str | None,
    query_name: str | None,
    poll_interval: int,
    timeout_minutes: int,
) -> dict:
    started = time.monotonic()
    timeout_seconds = timeout_minutes * 60 if timeout_minutes > 0 else None

    while True:
        info = (
            get_status_by_id(headers, query_id)
            if query_id
            else get_status_by_name(headers, query_name or "")
        )
        status = str(info.get("status", "UNKNOWN"))
        status_upper = status.upper()
        query_identifier = info.get("id") or query_id or "unknown-id"
        print(f"[{datetime.now().isoformat(timespec='seconds')}] id={query_identifier} status={status}")

        if status_upper in TERMINAL_STATUSES:
            return info

        elapsed_seconds = time.monotonic() - started
        if timeout_seconds is not None and elapsed_seconds > timeout_seconds:
            raise TimeoutError(
                f"Query did not reach terminal status within {timeout_minutes} minutes."
            )

        time.sleep(poll_interval)


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "swissdox_query"


def infer_download_filename(info: dict, fallback_name: str) -> str:
    download_url = info.get("downloadUrl")
    if download_url:
        parsed = urlparse(download_url)
        filename = Path(parsed.path).name
        if filename:
            return filename
    return f"{sanitize_name(fallback_name)}.tsv.xz"


def download_result(headers: dict[str, str], download_url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(download_url, headers=headers, timeout=600)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    kb = len(response.content) / 1024
    print(f"Downloaded {kb:.2f} KB -> {out_path}")


def default_output_dir_from_keywords(keywords_file: str) -> Path:
    """Infer projects/<project_name>/data from keywords path when possible."""
    kw_path = Path(keywords_file)
    try:
        idx = kw_path.parts.index("projects")
        # projects/<project_name>/...
        project_root = Path(*kw_path.parts[: idx + 2])
        if len(kw_path.parts) > idx + 1:
            return project_root / "data"
    except ValueError:
        pass
    # Safe fallback if keywords file is not under projects/
    return Path("data/00_downloads")


def main() -> None:
    args = parse_args()
    headers = load_credentials()
    languages = [part.strip() for part in args.languages.split(",") if part.strip()]
    if not languages:
        raise ValueError("No languages specified.")

    query_name = args.name or f"ta_swiss_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    if args.query_id:
        status_info = get_status_by_id(headers, args.query_id)
    else:
        keyword_spec = load_keywords(Path(args.keywords_file), languages)
        if keyword_spec["mode"] == "and_or":
            print(
                f"Loaded grouped keywords from {args.keywords_file} "
                f"(ai={len(keyword_spec['ai'])}, "
                f"education={len(keyword_spec['education'])}, "
                f"work={len(keyword_spec['work'])}) "
                f"for languages {languages}."
            )
        else:
            print(f"Loaded {len(keyword_spec['keywords'])} keywords from {args.keywords_file} for languages {languages}.")
        query_yaml = build_query_yaml(
            keyword_spec=keyword_spec,
            languages=languages,
            from_date=args.from_date,
            to_date=args.to_date,
            max_results=args.max_results,
        )
        try:
            submit_resp = submit_query(
                headers,
                query_yaml=query_yaml,
                name=query_name,
                comment=args.comment,
                expiration_date=args.expiration_date,
                test=args.test,
            )
            print(f"Submitted query '{query_name}' (test={args.test}).")
            print(f"Swissdox response: {submit_resp}")
            query_id = submit_resp.get("id") or submit_resp.get("queryId")
            status_info = get_status_by_id(headers, str(query_id)) if query_id else get_status_by_name(headers, query_name)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 406 and args.name:
                print(f"{exc}")
                print(
                    "Swissdox returned 406; trying to continue with existing query "
                    f"name '{query_name}'."
                )
                status_info = get_status_by_name(headers, query_name)
            else:
                raise

    print(
        "Current status:",
        {
            "id": status_info.get("id"),
            "name": status_info.get("name"),
            "status": status_info.get("status"),
            "downloadUrl": status_info.get("downloadUrl"),
        },
    )

    if args.wait and str(status_info.get("status", "")).upper() not in TERMINAL_STATUSES:
        status_info = poll_until_terminal(
            headers,
            query_id=status_info.get("id"),
            query_name=status_info.get("name") or query_name,
            poll_interval=args.poll_interval,
            timeout_minutes=args.timeout_minutes,
        )
        print(
            "Terminal status:",
            {
                "id": status_info.get("id"),
                "name": status_info.get("name"),
                "status": status_info.get("status"),
                "downloadUrl": status_info.get("downloadUrl"),
            },
        )

    if args.download:
        status = str(status_info.get("status"))
        if status.upper() != "FINISHED":
            raise RuntimeError(f"Cannot download because status is '{status}' (needs FINISHED).")
        download_url = status_info.get("downloadUrl")
        if not download_url:
            raise RuntimeError("No downloadUrl found in status response.")
        filename = infer_download_filename(status_info, query_name)
        output_dir = Path(args.output_dir) if args.output_dir else default_output_dir_from_keywords(args.keywords_file)
        out_path = output_dir / filename
        download_result(headers, download_url, out_path)


if __name__ == "__main__":
    main()

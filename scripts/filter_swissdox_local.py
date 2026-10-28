#!/usr/bin/env python3
"""Filter a local Swissdox TSV(.xz) before paragraph splitting.

Supports keyword formats:
1) flat list
2) {lang: [keywords]}
3) {ai: {...}, education: {...}, work: {...}} -> (AI AND education) OR (AI AND work)
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter local Swissdox dataset by keyword logic")
    parser.add_argument("--input", required=True, help="Path to local Swissdox .tsv or .tsv.xz file")
    parser.add_argument(
        "--keywords-file",
        default="projects/TA-Swiss/keywords.json",
        help=(
            "Path to keyword JSON. Supported: "
            "(1) flat list, (2) {lang: [keywords]}, "
            "(3) {ai, education, work} for (AI AND education) OR (AI AND work)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .tsv path (default: <input>.filtered.tsv).",
    )
    parser.add_argument(
        "--languages",
        default="en,de,fr,it",
        help="Comma-separated languages to use for keyword selection and optional row filter.",
    )
    parser.add_argument(
        "--language-column",
        default="language",
        help="Column to use for row-level language filtering.",
    )
    parser.add_argument(
        "--filter-languages",
        action="store_true",
        help="If set, keep only rows whose language is in --languages.",
    )
    parser.add_argument(
        "--text-columns",
        default="head,subhead,content",
        help="Comma-separated columns used for matching.",
    )
    parser.add_argument(
        "--source-id-column",
        default="content_id",
        help="Source id column for dedupe (if present).",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate source ids while filtering.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Rows per processing chunk.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ASCII", "ignore").decode("utf-8")
    return text.lower()


def dedupe_terms(terms: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for kw in terms:
        token = str(kw).strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def select_terms(payload: list | dict, languages: list[str]) -> list[str]:
    if isinstance(payload, list):
        return dedupe_terms(payload)
    if isinstance(payload, dict):
        terms: list[str] = []
        for lang in languages:
            terms.extend(payload.get(lang, []))
        return dedupe_terms(terms)
    raise ValueError("Expected list or {lang: [terms]} keyword shape.")


def load_keyword_spec(path: Path, languages: list[str]) -> dict:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and all(k in payload for k in ("ai", "education", "work")):
        ai_terms = select_terms(payload["ai"], languages)
        edu_terms = select_terms(payload["education"], languages)
        work_terms = select_terms(payload["work"], languages)
        if not ai_terms or not edu_terms or not work_terms:
            raise ValueError("Grouped keyword format requires non-empty ai/education/work terms.")
        return {"mode": "and_or", "ai": ai_terms, "education": edu_terms, "work": work_terms}

    terms = select_terms(payload, languages)
    if not terms:
        raise ValueError("No keywords selected.")
    return {"mode": "or", "keywords": terms}


def make_regex(terms: list[str]) -> str:
    normalized = [normalize_text(t) for t in terms if str(t).strip()]
    escaped = [re.escape(t) for t in normalized if t]
    if not escaped:
        return r"$^"
    return "|".join(escaped)


def row_mask(searchable: pd.Series, spec: dict) -> pd.Series:
    if spec["mode"] == "or":
        regex = make_regex(spec["keywords"])
        return searchable.str.contains(regex, regex=True, na=False)

    ai_regex = make_regex(spec["ai"])
    edu_regex = make_regex(spec["education"])
    work_regex = make_regex(spec["work"])

    ai_match = searchable.str.contains(ai_regex, regex=True, na=False)
    edu_match = searchable.str.contains(edu_regex, regex=True, na=False)
    work_match = searchable.str.contains(work_regex, regex=True, na=False)
    return ai_match & (edu_match | work_match)


def default_output_path(input_path: Path) -> Path:
    if input_path.name.endswith(".tsv.xz"):
        stem = input_path.name[:-7]
    else:
        stem = input_path.stem
    return input_path.with_name(f"{stem}.filtered.tsv")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    text_columns = [x.strip() for x in args.text_columns.split(",") if x.strip()]
    spec = load_keyword_spec(Path(args.keywords_file), languages)

    if spec["mode"] == "and_or":
        print(
            f"Loaded grouped keywords: ai={len(spec['ai'])}, "
            f"education={len(spec['education'])}, work={len(spec['work'])}"
        )
    else:
        print(f"Loaded OR keywords: n={len(spec['keywords'])}")

    total_in = 0
    total_out = 0
    seen_source_ids: set[str] = set()
    wrote_header = False

    chunks = pd.read_csv(input_path, sep="\t", chunksize=args.chunksize)
    for chunk in chunks:
        total_in += len(chunk)

        if args.filter_languages and args.language_column in chunk.columns:
            chunk = chunk[chunk[args.language_column].isin(languages)]

        existing_text_cols = [c for c in text_columns if c in chunk.columns]
        if not existing_text_cols:
            raise ValueError(f"None of text columns found in input: {text_columns}")

        searchable = (
            chunk[existing_text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .apply(normalize_text)
        )
        mask = row_mask(searchable, spec)
        filtered = chunk[mask].copy()

        if args.dedupe and args.source_id_column in filtered.columns:
            keep_rows = []
            for source_id in filtered[args.source_id_column].astype(str):
                if source_id in seen_source_ids:
                    keep_rows.append(False)
                else:
                    seen_source_ids.add(source_id)
                    keep_rows.append(True)
            filtered = filtered[pd.Series(keep_rows, index=filtered.index)]

        total_out += len(filtered)
        filtered.to_csv(output_path, sep="\t", index=False, mode="a", header=not wrote_header)
        wrote_header = True

        print(f"Processed {total_in} rows -> kept {total_out}")

    print(f"Done. Saved filtered dataset to: {output_path}")
    print(f"Final keep ratio: {total_out}/{total_in} ({(100.0 * total_out / max(total_in, 1)):.2f}%)")


if __name__ == "__main__":
    main()

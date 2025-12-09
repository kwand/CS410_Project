from __future__ import annotations

import json
import mmap
import click
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .sentiment import SentimentAnalyzer, InContextLearningOllamaAnalyzer


def _count_input_records(in_path: Path) -> int:
    # Fast line count using mmap; counts newline-delimited records.
    with in_path.open("rb") as fin, mmap.mmap(fin.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        if mm.size() == 0:
            return 0
        data = mm[:]
        count = data.count(b"\n")
        # If the file does not end with a newline, account for the final line.
        if data[-1:] != b"\n":
            count += 1
        return count


def _has_parse_failed(record: dict[str, Any], analyzer: SentimentAnalyzer) -> bool:
    # Only applicable to InContextLearningOllamaAnalyzer
    if not isinstance(analyzer, InContextLearningOllamaAnalyzer):
        return False

    sentiment = record.get("sentiment")
    if isinstance(sentiment, dict):
        sentiment_raw = sentiment.get("raw")
        if isinstance(sentiment_raw, dict) and sentiment_raw.get("error") == "parse_failed":
            return True

    return False

def _comment_key(record: dict[str, Any]) -> tuple[str, str] | None:
    video_id = record.get("video_id")
    comment_id = record.get("comment_id")
    if video_id is None or comment_id is None:
        return None
    return str(video_id), str(comment_id)


def _load_processed_records(out_path: Path, analyzer: SentimentAnalyzer) -> set[tuple[str, str]]:
    processed: set[tuple[str, str]] = set()
    with out_path.open("r", encoding="utf-8") as existing:
        for line in existing:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = _comment_key(record)
            if key and not _has_parse_failed(record, analyzer):
                processed.add(key)
    return processed


def annotate_comments(
    input_path: str | Path,
    output_path: str | Path,
    analyzer: SentimentAnalyzer,
    overwrite: bool = False,
) -> Path:
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    append_mode = False
    if overwrite and out_path.exists():
        click.echo(f"Overwriting existing sentiment-scored comments file {out_path}...")
    elif not overwrite and out_path.exists():
        click.echo(f"Found existing sentiment-scored comments file {out_path}. Appending to it...")
        append_mode = True
    else:
        click.echo(f"Creating new sentiment-scored comments file {out_path}...")

    mode = "a" if append_mode else "w"
    processed_keys = _load_processed_records(out_path, analyzer) if append_mode else set()
    total_records = _count_input_records(in_path)

    skipped_records = 0
    new_processed_records = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(mode, encoding="utf-8") as fout:
        for line in tqdm(
            fin,
            desc=f"Annotating comments with sentiment scores using {analyzer.backend_name}...",
            total=total_records,
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            key = _comment_key(record)

            if key and key in processed_keys:
                skipped_records += 1
                continue

            text = record.get("text", "")
            sentiment = analyzer.analyze(text)
            record["sentiment"] = sentiment
            record["sentiment_score"] = sentiment.get("normalized_score")
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            new_processed_records += 1

    click.echo(f"Skipped {skipped_records} comments that were already processed.")
    click.echo(f"Processed {new_processed_records} new comments.")

    return out_path

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from tqdm import tqdm


@dataclass
class CompetitorEntry:
    raw: Dict[str, Any]
    name_key: str


def load_official_scores(path: str | Path) -> Dict[str, CompetitorEntry]:
    """
    Load official scores JSON and build a lookup of COMPETITOR NAME (FIRST LAST, upper) -> entry.
    Expects each entry to have "name" in the format "Last, First".
    """
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup: Dict[str, CompetitorEntry] = {}
    for entry in entries:
        name = entry.get("name", "")
        if "," not in name:
            continue
        last, first = [part.strip() for part in name.split(",", 1)]
        key = _normalize_name_key(f"{first} {last}")
        lookup[key] = CompetitorEntry(raw=entry, name_key=key)
    return lookup


def parse_title(title: str, title_pattern: str, round_to_stage: Dict[str, str]) -> Tuple[str, str]:
    """
    Extract (name_key, stage_key) from a video title using a configurable regex and round mapping.
    """
    cleaned_title = title.strip()
    patterns = [
        title_pattern,
        r"^(.+?)\s+[–-]\s+([a-z]+)\s+round",  # fallback: lenient match, Unicode dash or hyphen
    ]
    last_error: Exception | None = None
    for pat in patterns:
        try:
            pattern = re.compile(pat, re.IGNORECASE)
        except re.error as exc:
            last_error = exc
            continue
        match = pattern.search(cleaned_title)
        if not match:
            continue
        name_raw, round_raw = match.group(1), match.group(2)
        name_key = _normalize_name_key(name_raw)
        stage_key = round_to_stage.get(round_raw.lower())
        if not stage_key:
            raise ValueError(f"Unknown round token in title: {round_raw!r}")
        return name_key, stage_key
    raise ValueError(f"Title does not match expected pattern(s): {title!r}. Last error: {last_error}")


def map_video_to_competitor(
    title: str, competitors: Dict[str, CompetitorEntry], title_pattern: str, round_to_stage: Dict[str, str]
) -> Tuple[CompetitorEntry, str]:
    """
    Deterministically map a video title to a competitor entry and stage key.
    Raises if the name/stage cannot be resolved.
    """
    name_key, stage_key = parse_title(title, title_pattern, round_to_stage)
    competitor = competitors.get(name_key)
    if not competitor:
        raise KeyError(f"No competitor found for {name_key!r} parsed from title {title!r}")
    return competitor, stage_key


def filter_comments(
    input_path: str | Path,
    output_path: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    ignore_replies: bool = True,
) -> Path:
    """
    Filter scraped comments by date window and optionally drop replies.
    Dates are inclusive and expected in YYYY-MM-DD.
    """
    start_dt = _parse_date(start_date) if start_date else None
    end_dt = _parse_date(end_date) if end_date else None

    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Filtering comments"):
            if not line.strip():
                continue
            record = json.loads(line)
            if ignore_replies and record.get("parent_id"):
                continue
            pub = record.get("published_at", "")
            if not _within_range(pub, start_dt, end_dt):
                continue
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def _parse_date(date_str: str) -> date:
    return datetime.fromisoformat(date_str).date()


def _within_range(published_at: str, start_dt: Optional[date], end_dt: Optional[date]) -> bool:
    if not (start_dt or end_dt):
        return True
    # published_at typically ISO 8601, often ending with Z
    try:
        pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00")).date()
    except Exception:
        return False
    if start_dt and pub_date < start_dt:
        return False
    if end_dt and pub_date > end_dt:
        return False
    return True


def _normalize_name_key(name_raw: str) -> str:
    """
    Normalize competitor names extracted from titles:
    - collapse whitespace
    - normalize dash variants to hyphen
    - uppercase for consistent lookups
    """
    cleaned = name_raw.replace("–", "-").replace("—", "-")
    return " ".join(cleaned.split()).upper()

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import click
from tqdm import tqdm

from data_utils.preprocess import load_official_scores, parse_title

from .config import AppConfig
from .youtube_client import Comment, YouTubeClient, VideoInfo


def match_videos(
    client: YouTubeClient,
    channel_id: str,
    cfg: AppConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    debug_log_path: Optional[Path] = None,
    debug_all_log_path: Optional[Path] = None,
) -> List[VideoInfo]:
    """Return videos whose titles match the configured pattern and optional date window."""
    pattern = re.compile(cfg.competition.channel["keyword_pattern"], re.IGNORECASE)
    matched: List[VideoInfo] = []
    unmatched: List[VideoInfo] = []
    all_videos: List[Tuple[VideoInfo, bool]] = []
    for video in client.iter_channel_videos(
        channel_id,
        max_results=cfg.scrape.max_videos,
        start_date=start_date,
        end_date=end_date,
    ):
        if pattern.search(video.title):
            matched.append(video)
            all_videos.append((video, True))
        elif debug_log_path:
            unmatched.append(video)
            all_videos.append((video, False))
        elif debug_all_log_path:
            all_videos.append((video, False))

    if debug_log_path:
        _write_unmatched_log(unmatched, debug_log_path, pattern)
    if debug_all_log_path:
        _write_all_videos_log(all_videos, debug_all_log_path, pattern)
    return matched


@dataclass
class ScrapeSummary:
    output_path: Path
    new_videos: int
    total_videos: int


def scrape_comments(cfg: AppConfig, output_path: Path) -> ScrapeSummary:
    """Scrape matched videos and write comments to JSONL (resume-aware)."""
    api_key = cfg.api.resolve_key()
    client = YouTubeClient(api_key)

    existing_index = _load_existing_video_index(output_path)
    expected_performances, name_lookup = _expected_performances(cfg)
    expected_count = len(expected_performances)
    if expected_count and len(existing_index) >= expected_count:
        click.echo(
            f"Found existing comments for {len(existing_index)} videos at {output_path} "
            f"which meets or exceeds expected performances ({expected_count}); skipping scrape."
        )
        _summarize_coverage(cfg, existing_index, expected_performances, name_lookup)
        return ScrapeSummary(
            output_path=output_path, new_videos=0, total_videos=len(existing_index)
        )

    channel_url = cfg.competition.channel.get("url")
    if not channel_url:
        raise ValueError("channel.url must be provided in competition settings.")
    channel_id = client.resolve_channel_id(channel_url)

    debug_log_path = None
    debug_all_log_path = None
    if cfg.scrape.debug_log_unmatched:
        debug_log_path = (
            Path(cfg.scrape.debug_log_path)
            if cfg.scrape.debug_log_path
            else output_path.parent / "unmatched_videos.log"
        )
    if cfg.scrape.debug_log_all_videos:
        debug_all_log_path = (
            Path(cfg.scrape.debug_log_all_path)
            if cfg.scrape.debug_log_all_path
            else output_path.parent / "all_channel_videos.log"
        )

    videos = match_videos(
        client,
        channel_id,
        cfg,
        start_date=cfg.competition.date_filter_start,
        end_date=cfg.competition.date_filter_end,
        debug_log_path=debug_log_path,
        debug_all_log_path=debug_all_log_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    videos_to_scrape = [v for v in videos if v.id not in existing_index]
    resumed = bool(existing_index)
    if resumed:
        click.echo(
            f"Found existing comments for {len(existing_index)} videos at {output_path}. "
            f"Scraping {len(videos_to_scrape)} missing videos."
        )
    else:
        click.echo(f"Scraping {len(videos)} videos...")

    mode = "a" if resumed else "w"
    needs_leading_newline = resumed and output_path.exists() and output_path.stat().st_size > 0

    new_videos = 0
    with output_path.open(mode, encoding="utf-8") as f:
        first_write = True
        for video in tqdm(videos_to_scrape, desc="Scraping videos"):
            comments = client.fetch_comments(
                video.id,
                video_title=video.title,
                max_comments=cfg.scrape.max_comments,
                include_replies=cfg.scrape.include_replies,
            )
            if not comments:
                continue
            if first_write and needs_leading_newline:
                f.write("\n")
            first_write = False
            for c in comments:
                f.write(json.dumps(_comment_to_dict(c), ensure_ascii=False) + "\n")
            existing_index[video.id] = video.title
            new_videos += 1

    _summarize_coverage(cfg, existing_index)

    return ScrapeSummary(output_path=output_path, new_videos=new_videos, total_videos=len(existing_index))


def _comment_to_dict(comment: Comment) -> dict:
    return {
        "video_id": comment.video_id,
        "video_title": comment.video_title,
        "comment_id": comment.comment_id,
        "text": comment.text,
        "author": comment.author,
        "like_count": comment.like_count,
        "published_at": comment.published_at,
        "updated_at": comment.updated_at,
        "parent_id": comment.parent_id,
    }


def _load_existing_video_index(path: Path) -> Dict[str, str]:
    """Return mapping of video_id -> video_title from an existing comments file, if present."""
    if not path.exists():
        return {}
    index: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = record.get("video_id")
            title = record.get("video_title")
            if vid and title and vid not in index:
                index[vid] = title
    return index


def _expected_performances(cfg: AppConfig) -> Tuple[Set[Tuple[str, str]], Dict[Tuple[str, str], str]]:
    """Build the set of expected (canonical_name, stage) pairs based on official scores."""
    scores = load_official_scores(cfg.competition.official_scores_path)
    valid_stages = set(cfg.competition.round_to_stage.values())
    expected: Set[Tuple[str, str]] = set()
    name_lookup: Dict[Tuple[str, str], str] = {}
    for entry in scores.values():
        display_name = entry.raw.get("name", entry.name_key.title())
        canon = _canonicalize_name(entry.name_key)
        for stage_key in valid_stages:
            stage_data = entry.raw.get(stage_key)
            if not stage_data:
                continue
            expected.add((canon, stage_key))
            name_lookup.setdefault((canon, stage_key), display_name)
    return expected, name_lookup


def _summarize_coverage(
    cfg: AppConfig,
    video_index: Dict[str, str],
    expected_performances: Optional[Set[Tuple[str, str]]] = None,
    name_lookup: Optional[Dict[Tuple[str, str], str]] = None,
) -> None:
    """Print coverage summary vs official scores."""
    if expected_performances is None or name_lookup is None:
        expected_performances, name_lookup = _expected_performances(cfg)
    scraped_performances: Set[Tuple[str, str]] = set()
    unparsable: List[Tuple[str, str]] = []

    for vid, title in video_index.items():
        try:
            name_key, stage_key = parse_title(
                title, cfg.competition.title_pattern, cfg.competition.round_to_stage
            )
        except Exception as exc:
            unparsable.append((title, str(exc)))
            continue
        canon = _canonicalize_name(name_key)
        scraped_performances.add((canon, stage_key))

    missing = expected_performances - scraped_performances
    click.echo(f"Gathered comments for {len(video_index)} videos.")
    if expected_performances:
        click.echo(f"Coverage: {len(scraped_performances)}/{len(expected_performances)} expected performances.")
    if missing:
        click.echo(f"Missing performances ({len(missing)}):")
        for name_key, stage_key in sorted(missing, key=lambda pair: (pair[1], pair[0])):
            display_name = name_lookup.get((name_key, stage_key), name_key.title())
            click.echo(f"- {display_name} ({stage_key})")
    else:
        click.echo(f"All expected performances found based on official scores.")

    if unparsable:
        click.echo(
            f"Warning: {len(unparsable)} video titles could not be parsed with the configured pattern; "
            "coverage may be incomplete."
        )


def _canonicalize_name(name: str) -> str:
    """
    Casefold, strip diacritics, drop parenthetical nicknames, and remove non-alphanumeric chars
    for tolerant matching between official names and video titles.
    """
    base = re.sub(r"\([^)]*\)", "", name)
    normalized = unicodedata.normalize("NFKD", base)
    cleaned = "".join(ch for ch in normalized if ch.isalnum())
    return cleaned.casefold()


def _write_unmatched_log(unmatched: List[VideoInfo], path: Path, pattern: re.Pattern) -> None:
    """Write unmatched videos to a log file for debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as log:
        log.write(f"# Unmatched videos for pattern {pattern.pattern!r}\n")
        log.write("# published_at\tvideo_id\ttitle\n")
        for video in unmatched:
            log.write(f"{video.published_at}\t{video.id}\t{video.title}\n")


def _write_all_videos_log(all_videos: List[Tuple[VideoInfo, bool]], path: Path, pattern: re.Pattern) -> None:
    """Write all fetched videos with match status to a log file for debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as log:
        log.write(f"# All channel videos for pattern {pattern.pattern!r}\n")
        log.write("# matched\tpublished_at\tvideo_id\ttitle\n")
        for video, matched in all_videos:
            log.write(f"{str(matched).lower()}\t{video.published_at}\t{video.id}\t{video.title}\n")

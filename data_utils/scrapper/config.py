from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class APIConfig:
    key_env: str = "YOUTUBE_API_KEY"
    key: str = ""

    def resolve_key(self) -> str:
        """Return the API key using env var fallback."""
        from os import getenv

        env_val = getenv(self.key_env or "")
        if env_val:
            return env_val
        if self.key:
            return self.key
        raise ValueError(
            "YouTube API key missing. Set env variable "
            f"{self.key_env!r} or populate api.key in config."
        )


@dataclass
class ScrapeConfig:
    max_videos: int = -1
    max_comments: int = -1
    include_replies: bool = True
    debug_log_unmatched: bool = False
    debug_log_path: Optional[str] = None
    debug_log_all_videos: bool = False
    debug_log_all_path: Optional[str] = None


@dataclass
class PathsConfig:
    data_dir: str = "data"

    def data_path(self) -> Path:
        return Path(self.data_dir)


@dataclass
class CompetitionConfig:
    name: str
    channel: Dict[str, str]
    title_pattern: str
    round_to_stage: Dict[str, str]
    official_scores_path: str = "raw_data/official_scores.json"
    date_filter_start: Optional[str] = None  # ISO date YYYY-MM-DD
    date_filter_end: Optional[str] = None  # ISO date YYYY-MM-DD
    ignore_replies_for_processing: bool = True
    average_score_key: Optional[str] = None
    len_norm_mode: str = "bm25"


@dataclass
class AppConfig:
    api: APIConfig
    scrape: ScrapeConfig
    paths: PathsConfig
    competition: CompetitionConfig


def load_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text()) or {}

    api = APIConfig(**data.get("api", {}))
    scrape = ScrapeConfig(**data.get("scrape", {}))
    paths = PathsConfig(**data.get("paths", {}))
    competition = CompetitionConfig(**data.get("competition", {}))
    return AppConfig(api=api, scrape=scrape, paths=paths, competition=competition)

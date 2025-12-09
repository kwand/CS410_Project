from __future__ import annotations

from pathlib import Path
import shutil
import re

import click

from reporting.metrics import generate_reports, load_scored_comments_df
from data_utils.scrapper.config import load_config
from data_utils.scrapper.scraper import scrape_comments
from data_utils.preprocess import filter_comments
from sentiment.pipeline import annotate_comments
from sentiment.sentiment import create_analyzer

DEFAULT_DATA_DIR = "data"
DEFAULT_RESULTS_DIR = "outputs"
DEFAULT_CONFIG_PATH = "config/chopin2025_config.yaml"

RAW_SUBDIR = "scrapped_data"
PROCESSED_SUBDIR = "processed_data"
SENTIMENT_SUBDIR = "sentiment"
REPORTS_SUBDIR = "reports"

RAW_COMMENTS_FILE = "comments_raw.jsonl"
PROCESSED_COMMENTS_FILE = "comments_processed.jsonl"
SCORED_COMMENTS_FILE = "comments_scored.jsonl"

@click.group(help="YouTube comment processing and sentiment CLI for competition analysis.")
def cli() -> None:
    pass


@cli.command(name="process_data", help="Scrape comments (incl. replies) and preprocess/filter.")
@click.option(
    "--config",
    "config_path",
    default=DEFAULT_CONFIG_PATH,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to YAML config for scraping and competition settings.",
)
@click.option(
    "--data-dir",
    default=DEFAULT_DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Override base data directory (defaults to ./data).",
)
def process_data(config_path: str, data_dir: str | None) -> None:
    cfg = load_config(config_path)
    paths = _resolve_paths(cfg, data_dir)
    raw_dir, processed_dir = paths["raw_dir"], paths["processed_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = paths["raw_comments"]
    processed_path = paths["processed_comments"]

    scraped = False
    click.echo(
        "Raw comments already exist; verifying coverage and resuming scrape for missing videos..."
        if raw_path.exists()
        else "Scraping comments..."
    )
    scrape_result = scrape_comments(cfg, raw_path)
    scraped = scrape_result.new_videos > 0

    if scraped:
        click.echo(f"Scraped comments to {raw_path}")

    if processed_path.exists() and not scraped:
        click.echo(f"Processed comments already exist at {processed_path}, skipping filtering.")
    else:
        filter_comments(
            input_path=raw_path,
            output_path=processed_path,
            start_date=cfg.competition.date_filter_start,
            end_date=cfg.competition.date_filter_end,
            ignore_replies=cfg.competition.ignore_replies_for_processing,
        )
        click.echo(f"Processed comments to {processed_path}")



@cli.command(
    name="analyze_sentiment",
    help="Annotate processed comments with sentiment scores (normalized to [0,1]).",
)
@click.option(
    "--config",
    "config_path",
    default=DEFAULT_CONFIG_PATH,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to YAML config.",
)
@click.option(
    "--data-dir",
    default=DEFAULT_DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, readable=True),
    help="Base data directory containing processed data (defaults to ./data).",
)
@click.option(
    "--results-dir",
    default=DEFAULT_RESULTS_DIR,
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    show_default=True,
    help="Parent directory under which outputs will be grouped by dataset and analyzer.",
)
@click.option(
    "--analyzer",
    "analyzer_name",
    default="vader",
    show_default=True,
    help="Sentiment backend: vader | transformers (alias: hf)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrite the existing sentiment-scored file instead of resuming/appending.",
)
def analyze_sentiment(
    config_path: str, data_dir: str | None, results_dir: str, analyzer_name: str, overwrite: bool
) -> None:
    cfg = load_config(config_path)
    paths = _resolve_paths(cfg, data_dir)
    processed_path = paths["processed_comments"]
    if not processed_path.exists():
        raise click.UsageError(f"Processed comments not found at {processed_path}")

    comp_slug = paths["comp_slug"]
    results_parent = Path(results_dir or DEFAULT_RESULTS_DIR)
    results_root = results_parent / f"{comp_slug}_{analyzer_name}"
    sentiment_dir = results_root / SENTIMENT_SUBDIR
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    output_path = sentiment_dir / SCORED_COMMENTS_FILE

    analyzer = create_analyzer(analyzer_name)
    annotate_comments(processed_path, output_path, analyzer, overwrite=overwrite)
    click.echo(f"Finishing writing sentiment-scored comments to {output_path}")


@cli.command(
    name="generate_report",
    help="Generate ranking and correlation reports from scored comments.",
)
@click.option(
    "--config",
    "config_path",
    default=DEFAULT_CONFIG_PATH,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to YAML config.",
)
@click.option(
    "--sentiment-path",
    default=None,
    type=click.Path(file_okay=False, dir_okay=False, readable=True),
    help="Path to sentiment-scored comments file. If omitted, you will be prompted to choose one.",
)
@click.option(
    "--results-base",
    default=DEFAULT_RESULTS_DIR,
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, readable=True, writable=True),
    help="Parent directory containing sentiment runs named <dataset>_<method>.",
)
def generate_report(config_path: str, sentiment_path: str | None, results_base: str) -> None:
    cfg = load_config(config_path)
    comp_slug = _slugify(cfg.competition.name)
    sentiment_file = (
        Path(sentiment_path)
        if sentiment_path
        else _prompt_for_sentiment_path(Path(results_base or DEFAULT_RESULTS_DIR))
    )
    if not sentiment_file.exists():
        raise click.UsageError(f"Sentiment-scored comments not found at {sentiment_file}")

    scored_df = load_scored_comments_df(sentiment_file)
    results_root = _infer_results_root(sentiment_file)
    out_reports_dir = results_root.parent / f"{results_root.name}" / REPORTS_SUBDIR
    if out_reports_dir.exists():
        click.echo(f"Removing existing reports directory at {out_reports_dir}")
        shutil.rmtree(out_reports_dir)
    generate_reports(
        scored_df,
        cfg.competition,
        out_reports_dir,
        name_suffix=comp_slug,
        len_norm_mode=getattr(cfg.competition, "len_norm_mode", None),
    )
    click.echo(f"Finished writing reports to {out_reports_dir}")


def _slugify(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def _resolve_paths(cfg, data_dir_override: str | None) -> dict:
    comp_slug = _slugify(cfg.competition.name)
    base_data_dir = Path(data_dir_override or cfg.paths.data_dir or DEFAULT_DATA_DIR)
    data_root = base_data_dir / comp_slug
    return {
        "comp_slug": comp_slug,
        "data_root": data_root,
        "raw_dir": data_root / RAW_SUBDIR,
        "processed_dir": data_root / PROCESSED_SUBDIR,
        "raw_comments": data_root / RAW_SUBDIR / RAW_COMMENTS_FILE,
        "processed_comments": data_root / PROCESSED_SUBDIR / PROCESSED_COMMENTS_FILE,
    }


def _prompt_for_sentiment_path(results_base: Path) -> Path:
    if not results_base.exists():
        raise click.UsageError(f"Results base directory not found at {results_base}")
    candidates = sorted(results_base.glob(f"*/{SENTIMENT_SUBDIR}/{SCORED_COMMENTS_FILE}"))
    if not candidates:
        raise click.UsageError(f"No sentiment outputs found under {results_base}")

    click.echo("Select a sentiment output:")
    for idx, cand in enumerate(candidates, start=1):
        click.echo(f"{idx}) {cand}")
    choice = click.prompt(
        "Enter the number of the sentiment file to use",
        type=click.IntRange(1, len(candidates)),
    )
    return candidates[choice - 1]


def _infer_results_root(sentiment_file: Path) -> Path:
    return (
        sentiment_file.parent.parent
        if sentiment_file.parent.name == SENTIMENT_SUBDIR
        else sentiment_file.parent
    )


if __name__ == "__main__":
    cli()

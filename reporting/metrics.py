from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils.preprocess import load_official_scores, parse_title
from data_utils.scrapper.scraper import _canonicalize_name


DEFAULT_AVERAGE_SCORE_KEY = "mean"


def _stage_order(competition_cfg) -> List[str]:
    """Return ordered list of stage keys defined by round_to_stage."""
    return list(dict.fromkeys(competition_cfg.round_to_stage.values()))


def _count_future_advancements(competitor: dict, current_stage: str, stage_order: List[str]) -> int:
    """Count later stages the competitor reaches after the current stage."""
    if not isinstance(competitor, dict) or current_stage not in stage_order:
        return 0
    idx = stage_order.index(current_stage)
    future = stage_order[idx + 1 :]
    count = 0
    for stage in future:
        if competitor.get(stage) is not None:
            count += 1
    return count


def _get_average_score_key(competition_cfg) -> str:
    """Return the configured official average score key with a safe default."""
    key = getattr(competition_cfg, "average_score_key", None)
    return key or DEFAULT_AVERAGE_SCORE_KEY


def load_scored_comments_df(path: str | Path) -> pd.DataFrame:
    """Load sentiment-scored comments JSONL into a DataFrame."""
    return pd.read_json(path, lines=True)


def aggregate_video_scores(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores per video with like weighting on every comment.

    Steps:
      - Drop rows without sentiment_score or video_id.
      - Weight each comment by max(1, like_count).
      - For each video, take the weighted average across all comments
    """
    df = scored_df.copy()
    df = df.dropna(subset=["sentiment_score", "video_id"])
    if df.empty:
        return pd.DataFrame(
            columns=["video_id", "video_title", "our_score", "comment_count", "comment_count_inc_likes"]
        )

    df["like_count_raw"] = df.get("like_count", 0).fillna(0).astype(float)
    df["weight"] = df["like_count_raw"].clip(lower=1)
    df["sentiment_score"] = df["sentiment_score"].astype(float)
    df["weighted_score"] = df["sentiment_score"] * df["weight"]

    grouped = df.groupby(["video_id", "video_title"])
    weight_sum = grouped["weight"].sum()
    weighted_score_sum = grouped["weighted_score"].sum()
    weighted_avg = weighted_score_sum / weight_sum
    comment_count = grouped.size()
    comment_count_inc_likes = grouped["like_count_raw"].sum()

    video_scores = (
        pd.DataFrame(
            {
                "our_score": weighted_avg,
                "comment_count": comment_count,
                "comment_count_inc_likes": comment_count_inc_likes,
            }
        )
        .reset_index()
    )
    return video_scores


def build_competitor_video_scores(
    video_scores: pd.DataFrame,
    competition_cfg,
    official_scores_path: str | Path,
    average_score_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Attach competitor and official score data to per-video aggregates 
    Official averages are pulled from the configured `average_score_key` (defaults to "mean").
    """
    if video_scores.empty:
        raise ValueError("No video scores available to map to competitors.")

    avg_key = average_score_key or _get_average_score_key(competition_cfg)
    stage_order = _stage_order(competition_cfg)
    competitors = load_official_scores(official_scores_path)
    competitor_lookup_canon = {_canonicalize_name(entry.name_key): entry.raw for entry in competitors.values()}

    rows: List[Dict[str, Any]] = []
    unmatched: List[tuple[str, str]] = []
    for _, row in video_scores.iterrows():
        title = row.get("video_title", "")
        try:
            name_key, stage_key = parse_title(
                title, competition_cfg.title_pattern, competition_cfg.round_to_stage
            )
        except Exception as exc:
            unmatched.append((str(title), f"parse_title error: {exc}"))
            continue

        canon = _canonicalize_name(name_key)
        competitor = competitor_lookup_canon.get(canon)
        if not competitor:
            unmatched.append((str(title), f"no official match for {canon}"))
            continue

        stage_data = competitor.get(stage_key)
        if not isinstance(stage_data, dict):
            unmatched.append((str(title), f"missing stage data for {canon}:{stage_key}"))
            continue

        rows.append(
            {
                "video_id": row.get("video_id"),
                "video_title": title,
                "competitor_canon": canon,
                "competitor_name": competitor.get("name"),
                "stage": stage_key,
                "our_score": row.get("our_score"),
                "official_mean": stage_data.get(avg_key),
                "jury_scores": stage_data.get("scores", {}),
                "comment_count": row.get("comment_count"),
                "comment_count_inc_likes": row.get("comment_count_inc_likes"),
                "advanced": bool(stage_data.get("result")),
                "future_advancements": _count_future_advancements(competitor, stage_key, stage_order),
            }
        )

    if unmatched:
        click.echo(f"[ranking] Skipped {len(unmatched)} videos due to name/stage mismatches.")
        for title, reason in unmatched:
            click.echo(f"[ranking] - {title} -> {reason}")

    return pd.DataFrame(rows)


def compute_correlation_summary(
    rows_df: pd.DataFrame, average_col: str = "official_mean", score_col: str = "our_score"
) -> Dict[str, Any]:
    """
    Compute Pearson correlations between our scores and official average/jury scores.
    """
    summary: Dict[str, Any] = {"mean": None, "jury": {}}

    if rows_df.empty:
        raise ValueError("rows_df is empty")

    if score_col not in rows_df.columns:
        raise ValueError(f"Score column {score_col} not found in rows_df")

    valid = rows_df.dropna(subset=[score_col, average_col])
    if len(valid) >= 2:
        corr_val = valid[[score_col, average_col]].corr().iloc[0, 1]
        summary["mean"] = float(corr_val)

    for jury in _collect_jurors(rows_df):
        pairs = rows_df[[score_col]].copy()
        pairs["jury_score"] = rows_df["jury_scores"].apply(
            lambda d: (d or {}).get(jury) if isinstance(d, dict) else None
        )
        pairs = pairs.dropna()
        if len(pairs) < 2:
            continue
        corr_val = pairs[[score_col, "jury_score"]].corr().iloc[0, 1]
        summary["jury"][jury] = float(corr_val)

    return summary


def save_correlation_plots(
    rows_df: pd.DataFrame,
    output_dir: str | Path,
    average_col: str = "official_mean",
    score_col: str = "our_score",
    suffix: str = "",
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if score_col not in rows_df.columns:
        raise ValueError(f"Score column {score_col} not found in rows_df")

    valid = rows_df.dropna(subset=[score_col, average_col])
    if len(valid) < 2:
        raise ValueError(f"Not enough valid data to create correlation plot for {score_col} and {average_col}")

    plt.figure()
    plt.scatter(valid[average_col], valid[score_col], alpha=0.6)
    plt.xlabel("Official mean")
    plt.ylabel("Our score")
    plt.title("Our scores vs Official mean")
    plt.tight_layout()
    filename = f"correlation_mean{suffix}.png" if suffix else "correlation_mean.png"
    plt.savefig(out_dir / filename)
    plt.close()


def save_score_distribution(
    rows_df: pd.DataFrame,
    scored_comments_df: pd.DataFrame,
    output_dir: str | Path,
    score_col: str = "our_score",
    suffix: str = "",
) -> Dict[str, Path]:
    """
    Output two plots:
      1) Distribution of per-video scores using `score_col` (with official jury scores overlay)
      2) Distribution of raw comment sentiment scores (not aggregated per video).
      3) If `score_col` is not "our_score", also plot unnormalized per-video scores.

    Returns a dict with paths for generated plots.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if score_col not in rows_df.columns:
        raise ValueError(f"Score column {score_col} not found in rows_df")
    if "sentiment_score" not in scored_comments_df.columns:
        raise ValueError("sentiment_score column not found in scored_comments_df")

    video_path = out_dir / f"score_distribution{suffix}.png"
    comment_path = out_dir / f"comment_score_distribution{suffix}.png"
    raw_video_path = out_dir / f"score_distribution_raw{suffix}.png"

    def _extract_scores(series: pd.Series) -> np.ndarray:
        return pd.to_numeric(series, errors="coerce").dropna().astype(float).to_numpy()

    def _collect_jury_norm(df: pd.DataFrame) -> np.ndarray:
        values: List[float] = []
        for scores in df.get("jury_scores", []):
            if isinstance(scores, dict):
                for val in scores.values():
                    try:
                        values.append(float(val))
                    except (TypeError, ValueError):
                        continue
        jury_arr = np.array(values, dtype=float)
        return (jury_arr - 1.0) / 24.0 if jury_arr.size else np.array([])

    def _plot_hist(scores: np.ndarray, label: str, alpha: float = 0.5, bins: int = 50) -> float:
        if scores.size == 0:
            return 0.0
        weights = np.ones(len(scores), dtype=float) / len(scores)
        hist_vals, _, _ = plt.hist(
            scores,
            bins=bins,
            range=(0, 1),
            weights=weights,
            alpha=alpha,
            label=label,
        )
        return float(hist_vals.max()) if len(hist_vals) else 0.0

    def _plot_with_jury(scores: np.ndarray, jury_norm: np.ndarray, path: Path, title: str, label: str) -> None:
        plt.figure()
        y_max = 0.0
        if jury_norm.size:
            unique, counts = np.unique(jury_norm, return_counts=True)
            freq = counts / counts.sum()
            plt.bar(unique, freq, width=0.025, alpha=0.6, label="Jury", color="green")
            if freq.size:
                y_max = max(y_max, float(freq.max()))
        y_max = max(y_max, _plot_hist(scores, label=label))
        plt.xlabel("Score (0-1)")
        plt.ylabel("Frequency")
        plt.title(title)
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(handles, labels)
        plt.xlim(0, 1)
        if y_max > 0:
            plt.ylim(0, y_max)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _plot_single_hist(scores: np.ndarray, path: Path, title: str, label: str) -> None:
        plt.figure()
        y_max = _plot_hist(scores, label=label, alpha=0.6)
        plt.xlabel("Score (0-1)")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.xlim(0, 1)
        if scores.size:
            plt.legend()
            if y_max > 0:
                plt.ylim(0, y_max)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    jury_norm = _collect_jury_norm(rows_df)
    our_scores = _extract_scores(rows_df[score_col])
    _plot_with_jury(
        our_scores, jury_norm, video_path, title="Score Distribution: Jury vs Sentiment", label=f"Our scores ({score_col})"
    )

    if score_col != "our_score":
        raw_scores = _extract_scores(rows_df["our_score"])
        _plot_with_jury(
            raw_scores,
            jury_norm,
            raw_video_path,
            title="Score Distribution: Jury vs Sentiment (Unnormalized)",
            label="Our scores (our_score)",
        )

    comment_scores = _extract_scores(scored_comments_df["sentiment_score"])
    _plot_single_hist(
        comment_scores,
        comment_path,
        title="Comment-Level Sentiment Score Distribution",
        label="Comment scores",
    )

    paths: Dict[str, Path] = {"video_distribution": video_path, "comment_distribution": comment_path}
    if score_col != "our_score":
        paths["video_distribution_raw"] = raw_video_path

    return paths


def _collect_jurors(rows_df: pd.DataFrame) -> List[str]:
    jurors = set()
    for scores in rows_df["jury_scores"]:
        if isinstance(scores, dict):
            jurors.update(scores.keys())
    return list(jurors)


def compute_precision_recall_at_k(rows_df: pd.DataFrame, k: int) -> Dict[str, Any]:
    """
    Precision@k and recall using `pred_advance` as the predicted set (top-k).

    precision = (# advanced in top k) / k
    recall    = (# advanced predicted) / (# advanced overall)
    """
    if rows_df.empty or "advanced" not in rows_df or k <= 0:
        raise ValueError("rows_df is empty or k is not positive")

    advanced_mask = rows_df["advanced"].fillna(False).astype(bool)
    predicted_mask = rows_df["pred_advance"].astype(bool)
    top_k_mask = rows_df["rank"] <= k

    true_positives_at_k = int((advanced_mask & top_k_mask).sum())
    true_positives = int((advanced_mask & predicted_mask).sum())
    actual_positives = int(advanced_mask.sum())

    precision = float(true_positives_at_k) / k if k else None
    recall = float(true_positives) / actual_positives if actual_positives else None

    return {
        "precision_at_k": precision,
        "recall": recall,
        "actual_positives": actual_positives,
        "true_positives_at_k": true_positives_at_k,
        "true_positives": true_positives,
        "k": k,
    }


def compute_ndcg(stage_df: pd.DataFrame, stage_name: str, final_stage: Optional[str]) -> Optional[float]:
    """
    Normalized DCG.

    - Final stage: gain is descending rank among advanced competitors by official_mean (non-advanced = 0).
    (For example, if 10 people placed for final awards, the gain of the first placed competitor is 10, the second is 9, etc.)
    - Other stages: gain is the number of future advancements.
    """
    if stage_df.empty or "rank" not in stage_df or "official_mean" not in stage_df:
        return None

    is_final = final_stage is not None and stage_name == final_stage

    if is_final:
        if "advanced" not in stage_df:
            return None
        advanced_mask = stage_df["advanced"].fillna(False).astype(bool)
        total_adv = int(advanced_mask.sum())
        gains = pd.Series(0.0, index=stage_df.index)
        if total_adv > 0:
            ordered = stage_df.loc[advanced_mask].sort_values("official_mean", ascending=False, na_position="last")
            for pos, idx in enumerate(ordered.index, start=1):
                gains.at[idx] = float(total_adv - pos + 1)
        gain_series = gains
    else:
        if "future_advancements" not in stage_df:
            return None
        gain_series = pd.to_numeric(stage_df["future_advancements"], errors="coerce").fillna(0)

    ranked = stage_df.sort_values("rank")
    gains_pred = gain_series.loc[ranked.index].to_numpy(dtype=float)
    dcg = float(sum(g / np.log2(idx + 1) for idx, g in enumerate(gains_pred, start=1)))

    ideal = stage_df.sort_values("official_mean", ascending=False, na_position="last")
    gains_ideal = gain_series.loc[ideal.index].to_numpy(dtype=float)
    idcg = float(sum(g / np.log2(idx + 1) for idx, g in enumerate(gains_ideal, start=1)))

    if idcg <= 0:
        return None
    return dcg / idcg


def _rank_competitors(rows_df: pd.DataFrame, score_col: str = "our_score") -> pd.DataFrame:
    """Sort competitors within each stage and compute rank/predicted advancement."""
    if rows_df.empty:
        return rows_df.copy()

    score_field = score_col if score_col in rows_df.columns else "our_score"
    ranked = rows_df.sort_values(["stage", score_field], ascending=[True, False]).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.groupby("stage").cumcount() + 1)

    # Compute predicted advancement using the cutoff rank.
    ranked["pred_advance"] = False
    for _, stage_df in ranked.groupby("stage"):
        rank_cutoff = int(stage_df["advanced"].sum()) if "advanced" in stage_df else 0
        if rank_cutoff <= 0:
            continue
        cutoff_mask = stage_df["rank"] <= rank_cutoff
        ranked.loc[stage_df.index, "pred_advance"] = cutoff_mask
    return ranked


def _write_ranking_outputs(
    rows_df: pd.DataFrame,
    reports_dir: Path,
    suffix: str,
    score_col: str = "our_score",
    final_stage: Optional[str] = None,
) -> Dict[str, Path]:
    """Output ranking CSVs (overall + per stage) and metrics."""
    ranking_cols = [
        "rank",
        "competitor_name",
        "stage",
        score_col,
        "our_score",
        "official_mean",
        "official_rank",
        "future_advancements",
        "comment_count",
        "comment_count_inc_likes",
        "advanced",
        "pred_advance",
    ]
    # de-duplicate "our_score"while preserving order, if score_col is left as default
    ranking_cols = list(dict.fromkeys(ranking_cols)) 

    ranking_df = rows_df.copy()
    if score_col not in ranking_df.columns:
        ranking_df[score_col] = np.nan
    if "future_advancements" not in ranking_df.columns:
        ranking_df["future_advancements"] = np.nan
    if "comment_count" not in ranking_df.columns:
        ranking_df["comment_count"] = np.nan
    if "comment_count_inc_likes" not in ranking_df.columns:
        ranking_df["comment_count_inc_likes"] = np.nan
    ranking_df["official_rank"] = pd.Series(pd.NA, index=ranking_df.index, dtype="Int64")
    for _, stage_df in ranking_df.groupby("stage"):
        ordered = stage_df.sort_values("official_mean", ascending=False, na_position="last")
        ranks = pd.Series(range(1, len(ordered) + 1), index=ordered.index, dtype="Int64")
        ranking_df.loc[ordered.index, "official_rank"] = ranks

    ranking_csv = reports_dir / f"ranking{suffix}.csv"
    ranking_df[ranking_cols].to_csv(ranking_csv, index=False)

    stage_files: Dict[str, Path] = {}
    for stage, stage_df in ranking_df.groupby("stage"):
        stage_file = reports_dir / f"ranking_{_slugify(str(stage))}{suffix}.csv"
        stage_df[ranking_cols].to_csv(stage_file, index=False)
        stage_files[str(stage)] = stage_file

    stage_metrics = {}
    for stage, stage_df in ranking_df.groupby("stage"):
        k_stage = int(stage_df["advanced"].sum()) if "advanced" in stage_df else 0
        stage_metric = compute_precision_recall_at_k(stage_df, k=k_stage // 2)
        stage_metric["ndcg"] = compute_ndcg(stage_df, stage, final_stage)
        stage_metrics[stage] = stage_metric

    metrics = {"per_stage": stage_metrics}
    metrics_path = reports_dir / f"ranking_metrics{suffix}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "ranking_csv": ranking_csv,
        "ranking_stage_csv": stage_files,
        "ranking_metrics": metrics_path,
    }

def bm25_len_norm(score, dl, avgdl, b=0.1, eps=1e-9, is_symmetric: bool = True):
    """
    BM25-style length normalization.

    b=0.1: apply very weak normalization by default
    is_symmetric: whether to apply symmetric normalization (i.e. shorter documents are equally penalized with longer documents)
    """
    r = dl / (avgdl + eps)
    if is_symmetric:
        r = max(r, 1.0 / (r + eps))  # == max(dl/avgdl, avgdl/dl)

    B = (1 - b) + b * r
    return score / (B + eps)


def _apply_length_norm(
    rows_df: pd.DataFrame,
    enabled: bool = True,
    mode: str = "shrinkage",
    output_col: str = "our_score_normalized",
) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Optionally normalize our_score using video comment counts (including likes), writing results to `output_col`.

    mode:
      - "bm25": bm25_len_norm with dl=comment_count_inc_likes and avgdl=mean of positives.
      - "shrinkage": shrink scores toward global mean using median comment_count_inc_likes as prior strength.
    """
    # Fast exit if disabled, missing data, or required columns absent.
    if (
        not enabled
        or rows_df.empty
        or "comment_count_inc_likes" not in rows_df
        or "our_score" not in rows_df
    ):
        return rows_df, None

    counts = pd.to_numeric(rows_df["comment_count_inc_likes"], errors="coerce")
    scores = pd.to_numeric(rows_df["our_score"], errors="coerce")
    valid_counts = counts > 0

    if mode == "bm25":
        # Normalize with BM25-style length scaling using average dl.
        avgdl = counts[valid_counts].mean()
        if pd.isna(avgdl) or avgdl <= 0:
            return rows_df, None

        norm_scores = scores.copy()
        norm_scores.loc[valid_counts] = norm_scores.loc[valid_counts].combine(
            counts.loc[valid_counts], lambda s, dl: bm25_len_norm(s, dl, avgdl)
        )
        normalized = rows_df.copy()
        normalized[output_col] = norm_scores
        return normalized, output_col

    if mode == "shrinkage":
        # Shrink toward global score mean with prior strength = median count.
        if counts[valid_counts].empty:
            return rows_df, None

        m_prior = float(counts[valid_counts].median())
        score_mask = valid_counts & scores.notna()
        weighted_sum = (scores[score_mask] * counts[score_mask]).sum()
        total_weight = counts[score_mask].sum()
        if pd.isna(m_prior) or m_prior <= 0 or pd.isna(total_weight) or total_weight <= 0:
            return rows_df, None
        mu0 = weighted_sum / total_weight

        norm_scores = scores.copy()
        norm_scores.loc[score_mask] = norm_scores.loc[score_mask].combine(
            counts.loc[score_mask],
            lambda s, n: (n * s + m_prior * mu0) / (n + m_prior),
        )
        normalized = rows_df.copy()
        normalized[output_col] = norm_scores
        return normalized, output_col

    return rows_df, None


def generate_reports(
    scored_df: pd.DataFrame,
    competition_cfg,
    results_dir: str | Path,
    reports: Optional[List[str]] = None,
    name_suffix: str = "",
    apply_len_norm: bool = True,
    len_norm_mode: str = "shrinkage",
) -> Dict[str, Path]:
    """
    Generate ranking and correlation reports from scored comments.

    reports: optional list of report types to generate (any of: "ranking", "correlation", "distribution").
             If None, generate all.
    name_suffix: optional string appended to output filenames (e.g., competition slug).
    apply_len_norm: whether to apply normalization to our_score using the comment count (incl. likes) per video.
                    When enabled, the normalized scores are stored in `our_score_normalized` and used for ranking/correlation.
    len_norm_mode: normalization strategy; supported values: "bm25", "shrinkage".
    """
    reports_dir = Path(results_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Aggregating sentiment scores per video...")
    video_scores = aggregate_video_scores(scored_df)
    rows_df = build_competitor_video_scores(
        video_scores,
        competition_cfg,
        competition_cfg.official_scores_path,
        average_score_key=_get_average_score_key(competition_cfg),
    )

    if apply_len_norm:
        click.echo(f"Applying normalization using {len_norm_mode} mode to sentiment scores...")
    rows_df, norm_col = _apply_length_norm(
        rows_df, enabled=apply_len_norm, mode=len_norm_mode, output_col="our_score_normalized"
    )

    score_col = norm_col or "our_score"
    _report_stage_coverage(rows_df, competition_cfg)
    rows_df = _rank_competitors(rows_df, score_col=score_col)
    stage_order = _stage_order(competition_cfg)
    final_stage = stage_order[-1] if stage_order else None

    output_paths: Dict[str, Path] = {}
    requested = set(reports or ["ranking", "correlation", "distribution"])

    suffix = f"_{name_suffix}" if name_suffix else ""

    if "ranking" in requested:
        click.echo(f"Saving rankings and metrics outputs (precision, recall, nDCG)...")
        output_paths.update(
            _write_ranking_outputs(rows_df, reports_dir, suffix, score_col=score_col, final_stage=final_stage)
        )

    if "correlation" in requested:
        click.echo(f"Computing correlation summary against official jury scores...")
        correlation_summary = compute_correlation_summary(
            rows_df, average_col="official_mean", score_col=score_col
        )
        correlation_path = reports_dir / f"correlation{suffix}.json"
        correlation_path.write_text(json.dumps(correlation_summary, indent=2), encoding="utf-8")
        save_correlation_plots(
            rows_df, reports_dir, average_col="official_mean", score_col=score_col, suffix=suffix
        )
        output_paths["correlation"] = correlation_path

    if "distribution" in requested:
        click.echo(f"Saving score distribution plots...")
        distribution_paths = save_score_distribution(
            rows_df, scored_df, reports_dir, score_col=score_col, suffix=suffix
        )
        output_paths["distribution"] = distribution_paths["video_distribution"]
        output_paths["comment_distribution"] = distribution_paths["comment_distribution"]

    return output_paths


def _slugify(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def _canonicalize_display_name(name: str) -> str:
    """Canonicalize a display name that may be in 'Last, First' form."""
    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        name = f"{first} {last}"
    return _canonicalize_name(name)


def _collect_expected_by_stage(competition_cfg) -> Dict[str, set[str]]:
    """Return canonical competitor names keyed by stage from official scores."""
    competitors = load_official_scores(competition_cfg.official_scores_path)
    stages = set(competition_cfg.round_to_stage.values())
    expected: Dict[str, set[str]] = {stage: set() for stage in stages}
    for comp in competitors.values():
        raw = comp.raw or {}
        for stage_key in stages:
            if raw.get(stage_key) is not None:
                expected[stage_key].add(_canonicalize_name(comp.name_key))
    return {stage: names for stage, names in expected.items() if names}


def _collect_actual_by_stage(rows_df: pd.DataFrame) -> Dict[str, set[str]]:
    """Return canonical competitor names keyed by stage from the sentiment output."""
    actual: Dict[str, set[str]] = {}
    for stage, canon, display in rows_df[["stage", "competitor_canon", "competitor_name"]].itertuples(index=False):
        if not isinstance(stage, str):
            continue
        key: Optional[str] = canon if isinstance(canon, str) else None
        if not key and isinstance(display, str):
            key = _canonicalize_display_name(display)
        if key:
            actual.setdefault(stage, set()).add(key)
    return actual


def _report_stage_coverage(rows_df: pd.DataFrame, competition_cfg) -> None:
    """Compare sentiment rows vs official roster and echo any gaps/extra entries."""
    expected_by_stage = _collect_expected_by_stage(competition_cfg)
    actual_by_stage = _collect_actual_by_stage(rows_df)

    coverage_messages: List[str] = []
    for stage, expected in expected_by_stage.items():
        actual = actual_by_stage.get(stage, set())
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            coverage_messages.append(
                f"[coverage] {stage}: expected {len(expected)} competitors, found {len(actual)} in sentiment output."
            )
            if missing:
                coverage_messages.append(f"[coverage] missing ({len(missing)}): {', '.join(missing)}")
            if extra:
                coverage_messages.append(f"[coverage] extra ({len(extra)} not in official roster): {', '.join(extra)}")

    unexpected_stages = sorted(set(actual_by_stage.keys()) - set(expected_by_stage.keys()))
    for stage in unexpected_stages:
        extras = sorted(actual_by_stage.get(stage, set()))
        if extras:
            coverage_messages.append(
                f"[coverage] {stage}: {len(extras)} competitors present but stage not in official roster."
            )

    if coverage_messages:
        for msg in coverage_messages:
            click.echo(msg)

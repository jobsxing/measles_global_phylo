from __future__ import annotations

import math
from pathlib import Path
import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT
VI = ROOT
OUT = ROOT
TABLES = ROOT
FIGURES = ROOT
WEEKS_PER_YEAR = 365.2425 / 7.0

STATE_ORDER = {
    13: ["CHN", "EEU", "JAK", "NAF", "NAM", "NEU", "OCE", "SAF", "SAM", "SEU", "SOA", "WEA", "WEU"],
    6: ["AFR", "AMR", "EMR", "EUR", "SEAR", "WPR"],
}

STATE_LABEL = {
    "CHN": "China",
    "EEU": "Eastern Europe",
    "JAK": "Japan/Korea",
    "NAF": "North Africa",
    "NAM": "North America",
    "NEU": "Northern Europe",
    "OCE": "Oceania",
    "SAF": "Sub-Saharan Africa",
    "SAM": "South America",
    "SEU": "Southern Europe",
    "SOA": "South Asia",
    "WEA": "West Asia",
    "WEU": "Western Europe",
    "AFR": "AFR",
    "AMR": "AMR",
    "EMR": "EMR",
    "EUR": "EUR",
    "SEAR": "SEAR",
    "WPR": "WPR",
}

VI_STATE_MAP = {
    "China": "CHN",
    "Eastern_Europe": "EEU",
    "Japan/Korea": "JAK",
    "North_Africa": "NAF",
    "North_America": "NAM",
    "North_Europe": "NEU",
    "Oceania": "OCE",
    "Sub-Saharan_Africa": "SAF",
    "South_America": "SAM",
    "South_Europe": "SEU",
    "South_Asia": "SOA",
    "West_Asia": "WEA",
    "Western_Europe": "WEU",
}


def hdi(values: np.ndarray, prob: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if np.all(values == values[0]):
        return float(values[0]), float(values[0])
    lo, hi = az.hdi(values, hdi_prob=prob)
    return float(lo), float(hi)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    lo50, hi50 = hdi(values, 0.50)
    lo95, hi95 = hdi(values, 0.95)
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "hdi50_low": lo50,
        "hdi50_high": hi50,
        "hdi95_low": lo95,
        "hdi95_high": hi95,
        "n_samples": int(np.isfinite(values).sum()),
    }


def read_history(path: Path, dataset: str, vi: bool = False) -> pd.DataFrame:
    history = pd.read_csv(path).rename(columns={"startLocation": "source", "endLocation": "sink"})
    if vi:
        history["source"] = history["source"].map(VI_STATE_MAP)
        history["sink"] = history["sink"].map(VI_STATE_MAP)
    required = ["source", "sink", "time", "treeId"]
    if history[required].isna().any().any():
        raise ValueError(f"Missing or unmapped history values in {path}")
    observed = set(history["source"]) | set(history["sink"])
    unexpected = observed - set(STATE_ORDER[13])
    if unexpected:
        raise ValueError(f"Unexpected history states in {path}: {sorted(unexpected)}")
    history["dataset"] = dataset
    history["treeId"] = history["treeId"].astype(str)
    return history


def state_and_weekly_summaries(
    history: pd.DataFrame,
    dataset: str,
    start: float,
    end: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = STATE_ORDER[13]
    x = history[(history["time"] >= start) & (history["time"] < end)].copy()
    tree_ids = history["treeId"].drop_duplicates().tolist()
    total = x.groupby("treeId").size().reindex(tree_ids, fill_value=0).to_numpy(float)
    outgoing = (
        x.groupby(["treeId", "source"]).size().unstack(fill_value=0)
        .reindex(index=tree_ids, fill_value=0).reindex(columns=states, fill_value=0)
        .to_numpy(float)
    )
    incoming = (
        x.groupby(["treeId", "sink"]).size().unstack(fill_value=0)
        .reindex(index=tree_ids, fill_value=0).reindex(columns=states, fill_value=0)
        .to_numpy(float)
    )
    source_share = np.divide(
        outgoing,
        total[:, None],
        out=np.zeros_like(outgoing),
        where=total[:, None] > 0,
    )
    sink_share = np.divide(
        incoming,
        total[:, None],
        out=np.zeros_like(incoming),
        where=total[:, None] > 0,
    )
    state_rows: list[dict[str, float | str]] = []
    for j, state in enumerate(states):
        row: dict[str, float | str] = {"dataset": dataset, "state": state, "label": STATE_LABEL[state]}
        for metric, values in [("source_share", source_share[:, j]), ("sink_share", sink_share[:, j])]:
            row.update({f"{metric}_{key}": value for key, value in summarize(values).items()})
        state_rows.append(row)

    n_weeks = int(math.ceil((end - start) * WEEKS_PER_YEAR))
    x["week_index"] = np.floor((x["time"] - start) * WEEKS_PER_YEAR).astype(int).clip(0, n_weeks - 1)
    weekly_counts = (
        x.groupby(["treeId", "week_index"]).size().unstack(fill_value=0)
        .reindex(index=tree_ids, fill_value=0).reindex(columns=np.arange(n_weeks), fill_value=0)
        .to_numpy(float)
    )
    weekly_share = np.divide(
        weekly_counts,
        total[:, None],
        out=np.zeros_like(weekly_counts),
        where=total[:, None] > 0,
    )
    rolling_weeks = 13
    kernel = np.ones(rolling_weeks, dtype=float)
    rolling_denominator = np.convolve(np.ones(n_weeks, dtype=float), kernel, mode="same")
    rolling_share = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same") / rolling_denominator,
        axis=1,
        arr=weekly_share,
    )
    week_rows: list[dict[str, float | str]] = []
    for j in range(n_weeks):
        center = start + (j + 0.5) / WEEKS_PER_YEAR
        week_rows.append(
            {
                "dataset": dataset,
                "week_index": j,
                "decimal_year": center,
                **{f"weekly_jump_share_{key}": value for key, value in summarize(weekly_share[:, j]).items()},
                **{
                    f"weekly_13w_rolling_share_{key}": value
                    for key, value in summarize(rolling_share[:, j]).items()
                },
            }
        )
    return pd.DataFrame(state_rows), pd.DataFrame(week_rows)



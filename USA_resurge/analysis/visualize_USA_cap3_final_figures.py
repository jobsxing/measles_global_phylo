
'Shared helpers retained from the original USA cap3 plotting code.'

import os

import re

from datetime import datetime

import baltic as bt

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.colors as clr

import matplotlib.lines as mlines

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np

import pandas as pd

import seaborn as sns

ROOT = os.path.dirname(os.path.abspath(__file__))

GEO = os.path.join(ROOT, "N450")

OUT = os.path.join(ROOT, "figures", "N450")

os.makedirs(OUT, exist_ok=True)

TREE = os.path.join(GEO, "geo_USA_cap3.HIPSTR.tree")

MJ = os.path.join(GEO, "mj_history_geo_USA_cap3.csv")

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei", "SimSun"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})

STATE_COLORS = {
    "USA": "#6a4c93",
    "Western_Europe": "#e76f51",
    "WesternEurope": "#e76f51",
    "South_Europe": "#2a9d8f",
    "SouthEurope": "#2a9d8f",
    "North_Europe": "#457b9d",
    "NorthEurope": "#457b9d",
    "Eastern_Europe": "#1d3557",
    "EasternEurope": "#1d3557",
    "North_America": "#6a4c93",
    "NorthAmerica": "#6a4c93",
    "South_America": "#8ab17d",
    "SouthAmerica": "#8ab17d",
    "China": "#8a171a",
    "Japan/Korea": "#f4a261",
    "JapanKorea": "#f4a261",
    "South_Asia": "#2d6a4f",
    "SouthAsia": "#2d6a4f",
    "West_Asia": "#277da1",
    "WestAsia": "#277da1",
    "North_Africa": "#f77f00",
    "NorthAfrica": "#f77f00",
    "Sub-Saharan_Africa": "#d62828",
    "SubSaharanAfrica": "#d62828",
    "Oceania": "#8d99ae",
}

def latest_tip_anchor(tree_path):
    text = open(tree_path, "r", encoding="utf-8", errors="ignore").read()
    found = re.findall(r"_(\d{4}-\d{2}(?:-\d{2})?)(?:_|:|\[)", text)
    dates = []
    for raw in found:
        if len(raw) == 7:
            raw = raw + "-15"
        try:
            dates.append(datetime.strptime(raw, "%Y-%m-%d"))
        except ValueError:
            pass
    return max(dates).strftime("%Y-%m-%d") if dates else "2026-06-01"

def load_tree():
    tree = bt.loadNexus(TREE, tip_regex=r"_(\d{4}-\d{2})(?:-|_)", date_fmt="%Y-%m", variableDate=False)
    tree.setAbsoluteTime(bt.decimalDate(latest_tip_anchor(TREE)))
    tree.traverse_tree()
    tree.sortBranches()
    return tree

def node_state(node):
    return getattr(node, "traits", {}).get("length")

def node_prob(node):
    value = getattr(node, "traits", {}).get("length.prob")
    try:
        return float(value) if value is not None else 1.0
    except Exception:
        return 1.0

def draw_year_bands(ax, alpha=0.06):
    x0, x1 = ax.get_xlim()
    start = int(np.floor(x0 / 5) * 5)
    for i, year in enumerate(range(start, int(np.ceil(x1)) + 5, 5)):
        if i % 2 == 0:
            ax.axvspan(year, min(year + 5, x1), color="k", alpha=alpha, zorder=0)

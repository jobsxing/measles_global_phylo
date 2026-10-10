"""annual3.pdf, extracted from saptiotemporal_new.ipynb cell 8 (one-based)."""
from pathlib import Path
from metapopulation_common import load_data, plot_cluster_bands

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
presence_df_C1, edges_df_C1, years_C1, rows_C1, df_jump_C1 = load_data()

plot_cluster_bands(
    pres_df = presence_df_C1,
    edges_df = edges_df_C1,
    years = years_C1,
    rows = rows_C1,
    CELL_W=5, CELL_H=1, YEAR_GAP=2.0, BUBBLE_PT=12,
    MAX_YEAR_SPAN=6, BOX_ASPECT=0.12,
    REGION_COLORS={
        "Europe":   "#2a9d8f",
        "Americas": "#004e98",
        "Endemic":  "#fdf0d5",
        "Asia":     "#c1121f",
        "Oceania":  "#a22522",
    },
    EVIDENCE_COLOR={"I":"#456f71","II":"#bfcfd8","III":"#fed9b7","IV":"#c8a55b","NA":"#11EF7C"},
    BF_EDGE_COLOR={"100-200":"#F2F2F2", ">200":"#000000"},
    YEAR_BAND_COLORS=("#e9ecef", "#6c757d"),
    LINEWIDTH_MIN=0.6, LINEWIDTH_MAX=3,
    USE_QUANTILES=False, MANUAL_THRESHOLDS=(1, 3, 6),
    CURVE_BASE=0.25, CURVE_GAIN=0.10,
    ARROW_HEAD=14, ARROW_ALPHA=1.0,
    OUTLINE_MIN=2.0, OUTLINE_MARGIN=1.0,
    title="Metapopulation model",
    save_path=OUT_DIR / "annual3.pdf",
    dpi=600
)

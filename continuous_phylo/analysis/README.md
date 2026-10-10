# Continuous phylogeography and global incidence

This directory contains two visualization workflows: a geographically annotated HIPSTR phylogeny and continuous-diffusion summaries aligned with global measles incidence. Related BEAST configurations are in `../xml/`.

## Code

| File | Purpose | Inputs | Outputs |
| --- | --- | --- | --- |
| `plot_HIPSTR_tree.py` | Read the HIPSTR tree with `baltic`, recover geographical state annotations and probabilities, and plot the dated phylogeny with regional colors. | `MeV_combine.discrete.HIPSTR.tree` | `MeV_combine.discrete.HIPSTR.phylogeny.pdf` and `.phylogeny.preview.png` |
| `velocity.ipynb` | Summarize posterior diffusion velocity and diffusion coefficient, overlay monthly global incidence, and scan lagged correlations over -24 to +24 months. | `geo.continuous.combined.resamp50000.log`; `case_status_pop_incidence_2011_2026_updated.csv` | Diffusion/incidence and cross-correlation figures, plus a correlation summary CSV |

`MeV_combine.discrete.combined.log` is a companion discrete-analysis output retained here; neither of these two visualization entry points reads it.

## Run

From the repository root:

```sh
cd continuous_phylo/analysis
python plot_HIPSTR_tree.py
jupyter notebook velocity.ipynb
```

Run the notebook cells in order with this directory as the notebook working directory. It reads its inputs relative to `.`. The Python script resolves inputs relative to its own location. Outputs are written into this analysis directory.

Dependencies are `baltic`, `numpy`, `pandas`, `matplotlib`, `scipy`, and Jupyter/IPython. The tree script needs `baltic`; the notebook does not.

## Notebook outputs and definitions

- `continuous_diffusion_velocity_coefficient_with_incidence.pdf` and `.preview.png`: posterior medians, 50% credible intervals, incidence overlays, and posterior trace insets.
- `continuous_diffusion_velocity_coefficient_cross_correlation.pdf` and `.preview.png`: lagged correlation curves for the two diffusion parameters.
- `continuous_diffusion_velocity_coefficient_cross_correlation_summary.csv`: the corresponding numerical summaries.

The notebook aggregates cases and population by month and computes incidence per 100,000 people. It uses `2026-05-28` as the time anchor, zero additional burn-in on the supplied combined log, and Spearman correlation by default. Its highlighted intervals are defined in `EPOCHS`. Notebook outputs and execution counts are cleared in the distributed notebook so that old figures and machine-specific traceback paths are not embedded.

# Measles genomic epidemiology: Cell manuscript revision

This repository collects major analysis code, figure inputs, and BEAST XML configurations supporting the Cell manuscript revision.

## Repository contents

| Directory | Scientific purpose 
| --- | --- 
| `meta-epi/` | Assemble the epidemiological cross-border transmission evidence visualization, including the combined evidence index and regional flow matrix.
| `epoch-DTA/` | Summarize discrete trait analysis (DTA) migration networks and net export across four periods, together with GLM predictor effects and vaccination-covariate sensitivity analyses. 
| `N450_WGS-sensitivity/` | Compare matched N450 and whole-genome (WGS) analyses, root-age-calibrated N450 analyses, and the main N450 dataset using temporal source/sink summaries and tanglegrams. 
| `continuous_phylo/` | Visualize the geographically annotated HIPSTR phylogeny, continuous diffusion velocity and diffusion coefficient, and their temporal relationship with global measles incidence.
| `Euro_resurge/` | Compare European and global transitions under two-state and four-state DTA sampling designs
| `phylo_metapopu/` | Visualize region-year metapopulation connections and four-period regional transition heatmaps from Markov-jump histories and Bayes factors.
| `USA_resurge/` | Visualize introductions into the USA, regional WGS migration, associations with case counts, and N450/WGS importation concordance.

## Layout and paths

Six projects use `analysis/` for code and its required inputs, and `xml/` for BEAST model configurations. `meta-epi/` retains its `code/` and `data/` layout. 

The documented figure workflows use the prepared posterior outputs and summary tables included in the analysis directories. The XML files record the model configurations; the figure bundles are not complete archives of every MCMC run. In particular, the USA tree workflow uses an included posterior-state cache instead of the full posterior tree collection.

Running the code creates the figure and table outputs described in each analysis README. Existing input tables and cached summaries should be retained. The USA bundle also contains three component PDFs in `intermediate_panels/` for the manually assembled WGS composite figure.

## Software

Python scripts use combinations of `numpy`, `pandas`, `matplotlib`, `scipy`, `seaborn`, `baltic`, and `arviz`. The notebook additionally uses Jupyter/IPython. R scripts use plotting, spatial, and phylogenetic packages listed in the relevant analysis README. Package lists describe imports in the provided code; they are not a pinned environment specification. Arial is used by several figures; font availability can affect the rendered appearance.

BEAST is needed to rerun the XML-based inference, but is not needed for the documented Python/R figure workflows using the supplied outputs.

## Epidemiological evidence: `meta-epi`

`meta-epi/code/plot_s27_epi_evidence.R` produces the combined Figure S27 evidence visualization from:

- `meta-epi/data/measles_importation_combined_index_v2.csv`: standardized cross-border evidence records used for maps and the chronological display.
- `meta-epi/data/s27_regional_flow_matrix_from_reference.csv`: regional flow values used for the regional matrix panel.

Run from the repository root:

```sh
Rscript meta-epi/code/plot_s27_epi_evidence.R
```

The script uses `readr`, `dplyr`, `tidyr`, `sf`, `rnaturalearth`, `ggplot2`, `patchwork`, and `scales`. It writes `Figure_S27_epi_evidence_combined_plus_jhu_cdc_v2_core.pdf` and the corresponding PNG to `meta-epi/figures/`.

## Interpretation

MJ denotes Markov jumps, BF denotes Bayes factor, and WGS denotes whole-genome sequence data. Route filters, temporal windows, averaging rules, and sampling designs differ between analyses. Use the definitions in the corresponding code and analysis README when interpreting a figure; a common filename prefix does not imply an identical statistical procedure.

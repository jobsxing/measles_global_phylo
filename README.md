# Global measles genomic epidemiology: In revision for Cell

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


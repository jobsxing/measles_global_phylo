# Global measles phylogenetic analyses

This repository contains the BEAST XML configurations used for global and regional measles phylogenetic, phylogeographic, and phylodynamic analyses. The collection includes continuous phylogeography, discrete-trait diffusion, generalized linear model (GLM) analyses, empirical-tree analyses, sensitivity analyses, and spatiotemporal-deme analyses.

## Repository structure

| Directory | XML files | Analysis scope |
| --- | ---: | --- |
| `1_global_continuous/` | 2 | Global empirical-tree inference and continuous phylogeography |
| `2-3_global_movement_time_inhomogeneous/` | 9 | Global time-inhomogeneous movement, combined and genotype-specific empirical trees, primary GLM analysis, and MCV1/MCV2 sensitivity analyses |
| `4_europe/` | 11 | European two-state datasets and four-state titration analyses, including empirical-tree and geographic models |
| `5_US_epidemic/` | 9 | United States epidemic analyses under alternative caps, with empirical-tree, geographic, and titration configurations |
| `6_eliminated_countries/` | 9 | Genotype B, D, and G/H spatiotemporal-deme analyses and updated combined/genotype-specific empirical-tree workflows |

There are 40 XML files in total. Directory names and XML filenames are preserved from the original analysis workspace so that analysis provenance remains traceable.

## Software

The XML root metadata indicate BEAST 1.x configurations: 39 files declare version `10.5.0`, and one file declares version `1.10.4`. Use a compatible BEAST installation and the package versions required by the individual XML files. BEAGLE is optional but recommended for computationally intensive sequence-likelihood analyses.

Check the available BEAGLE resources before starting a run:

```bash
java -jar /path/to/beast.jar -beagle_info
```

A typical GPU-enabled invocation is:

```bash
java -jar /path/to/beast.jar -beagle_gpu -beagle_cuda path/to/analysis.xml
```

Review the XML's output filenames and the target working directory before using `-overwrite`. Long analyses should use unique run prefixes and checkpoint files where appropriate.

## External empirical-tree inputs

Several phylogeographic XML files condition on posterior empirical-tree distributions through `empiricalTreeDistributionModel`. Because these posterior `.trees` files are large, they are not included in this repository. The BEAST XML configurations used to infer the corresponding empirical-tree distributions are provided in the relevant `empirical_tree` directories. Run those XML configurations (and combine/post-process replicate tree outputs as appropriate), or otherwise prepare matching posterior tree distributions, before running the dependent phylogeographic XML files.

Paths in BEAST XML files are resolved from the directory in which BEAST is launched. Preserve the intended working-directory layout or update the `fileName` attributes deliberately.

| XML | Required empirical-tree file |
| --- | --- |
| `1_global_continuous/02_continuous_phylogeography/continous.phylo.10.4.2.xml` | `empir.continuous.final.trees` |
| `2-3_global_movement_time_inhomogeneous/02_combined_GLM_primary/phylogeo.merge_vaccineindex_1.xml` | `../GLM_empiric_combine_final.trees` |
| `2-3_global_movement_time_inhomogeneous/03_sensitivity_withoutMCV1/phylogeo.merge_withoutMCV1_1.xml` | `../GLM_empiric_combine_final.trees` |
| `2-3_global_movement_time_inhomogeneous/04_sensitivity_withoutMCV2/phylogeo.merge_withoutMCV2_1.xml` | `../GLM_empiric_combine_final.trees` |
| `2-3_global_movement_time_inhomogeneous/06_genoB_geo_GLM/geo_genoB_comb1.xml` | `genoB_comb.final.resamp400000.trees` |
| `2-3_global_movement_time_inhomogeneous/08_genoD_geo_comb/geo_genoD_comb.xml` | `../genoD_comb_empir_final.trees` |
| `2-3_global_movement_time_inhomogeneous/09_genoD_geo_GLM/geo_GenoD_1.xml` | `../genoD_comb_empir_final.trees` |
| `4_europe/05_fourstate_titration1_geo/fourstate_titration1_seq_geo_1.xml` | `titration1_sequences.combined.trees` |
| `4_europe/07_fourstate_titration2_geo/fourstate_titration2_seq_geo_1.xml` | `titration2_sequences.combined.trees` |
| `4_europe/09_fourstate_titration3_geo/fourstate_titration3_seq_geo_1.xml` | `titration3_sequences.combined.trees` |
| `4_europe/11_fourstate_titration4_geo/fourstate_titration4_seq_geo_1.xml` | `titration4_sequences.combined.trees` |
| `5_US_epidemic/02_USA_cap1_geo/geo_USA_cap1_1.xml` | `USA_cap1_empir.combined.trees` |
| `5_US_epidemic/04_USA_cap2_geo/geo_USA_cap2_1.xml` | `USA_cap2_empir.combined.trees` |
| `5_US_epidemic/06_USA_cap3_geo/geo_USA_cap3_1.xml` | `USA_cap3_empir.combined.trees` |
| `5_US_epidemic/07_USA_cap5_geo_titration/geo_USA_cap5.xml` | `../empir_tre/USA_cap5_empirical_tree_distribution_PLACEHOLDER.trees` |
| `5_US_epidemic/08_USA_cap7_geo_titration/geo_USA_cap7.xml` | `../empir_tre/USA_cap7_empirical_tree_distribution_PLACEHOLDER.trees` |
| `5_US_epidemic/09_USA_cap10_geo_titration/geo_USA_cap10.xml` | `../empir_tre/USA_cap10_empirical_tree_distribution_PLACEHOLDER.trees` |
| `6_eliminated_countries/01_genoB_spatiotemporal_deme/spatio_cluster1_1.xml` | `../cluster1_claen_noOutlier_comb_resamp.trees` |
| `6_eliminated_countries/02_genoD_spatiotemporal_deme/spatio_cluster3_1.xml` | `../cluster3_clean_noOutlier_resamp.trees` |
| `6_eliminated_countries/03_genoGH_spatiotemporal_deme/spatio_cluster2_1.xml` | `../cluster2_claen_noOutlier_comb_resamp.trees` |
| `6_eliminated_countries/05_combined_spatiotemporal_deme_geo/combined_geo_empir_sap1.xml` | `combined_sap.emp.final.trees` |
| `6_eliminated_countries/07_genoB_spatiotemporal_deme_updated_geo/genoB_geo_empir_sap1.xml` | `genoB_sap.emp.final.trees` |
| `6_eliminated_countries/09_genoD_spatiotemporal_deme_updated_geo/genoD_geo_empir_sap1.xml` | `genoD_sap.emp.final.trees` |

The three United States cap-5, cap-7, and cap-10 XML files intentionally contain `PLACEHOLDER` empirical-tree filenames. Replace these placeholders with the final posterior tree distributions before execution.

## Output and convergence checks

BEAST output files such as `.log`, `.trees`, `.tree`, `.states`, and checkpoint files are intentionally excluded from this XML collection because they can be large and are generated results. The XML configurations needed to reproduce the empirical-tree inference and downstream analyses are included. For replicated runs, keep run identifiers consistent across all output filenames. Before interpreting results:

1. Remove an appropriate burn-in only after inspecting trace behavior.
2. Check effective sample sizes (ESS) in Tracer or an equivalent tool.
3. Confirm that combined runs use compatible models, logging intervals, and parameter columns.
4. Record the exact XML, BEAST version, package versions, run command, and random seed used for each analysis.

## XML validation status

Before upload, all 40 XML files were checked with an XML parser. Every file was well formed, began with the XML declaration (`<?xml`), and had no UTF-8 byte-order mark (BOM). This structural check does not confirm that external `.trees` inputs are present or that all taxa match those external tree distributions; those checks must be completed in the execution environment.

One source configuration requires model-level review before execution: `2-3_global_movement_time_inhomogeneous/08_genoD_geo_comb/geo_genoD_comb.xml` declares 12 geographic states, while `geography.frequencies` has dimension 13 and `geography.count` contains 169 (13 x 13) entries. The XML is preserved unchanged from the original analysis workspace so that this repository does not silently alter the model specification.

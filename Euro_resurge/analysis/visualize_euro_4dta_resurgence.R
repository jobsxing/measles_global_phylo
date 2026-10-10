suppressPackageStartupMessages({
  library(treeio)
  library(ape)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(lubridate)
  library(patchwork)
  library(gridExtra)
  library(grid)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
stopifnot(length(script_arg) == 1L)
analysis_dir <- normalizePath(dirname(sub("^--file=", "", script_arg)), winslash = "/", mustWork = TRUE)
base_dir <- file.path(analysis_dir, "4dta")
out_dir <- file.path(analysis_dir, "figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

state_info <- tibble::tibble(
  state = c("EUR_eli", "EUR_end", "GLO_eli", "GLO_end"),
  state_label = c(
    "Europe elimination region",
    "Europe endemic region",
    "Global elimination region",
    "Global endemic region"
  ),
  plot_label = c("Europe\neliminated", "Europe\nendemic", "Global\neliminated", "Global\nendemic"),
  table_label = c("EU\nelim", "EU\nend", "Global\nelim", "Global\nend"),
  color = c("#2d6a4f", "#134074", "#c9ada7", "#880d1e")
)

state_levels <- state_info$state
state_colors <- stats::setNames(state_info$color, state_info$state)
state_labels <- stats::setNames(state_info$state_label, state_info$state)

studies <- tibble::tibble(
  id = c("titration1", "titration1_2", "titration2", "titration3", "titration4"),
  study = c("Study1: 1:1:1:1", "Study2: 1:2:1:1", "Study3: 1:1:1:1.5",
            "Study4: 1:1:1:2", "Study5: 1:1:1:2.5"),
  study_short = c("T1", "T2", "T3", "T4", "T5"),
  violin_title = paste("Titration study", 1:5),
  label_fill = c("#108a27", "#c0270b", "#f080a8", "#1c2e8a", "#6b8e23"),
  tree_file = file.path(base_dir, c(
    "titration1_geo.history.HIPSTR.tree",
    "titration1_2_geo.history.HIPSTR.tree",
    "titration2_geo.history.HIPSTR.tree",
    "titration3_geo.history.HIPSTR.tree",
    "titration4_geo.history.HIPSTR.tree"
  )),
  mj_file = file.path(base_dir, c(
    "mj_history_titration1.csv",
    "mj_history_titration1_2.csv",
    "mj_history_titration2.csv",
    "mj_history_titration3.csv",
    "mj_history_titration4.csv"
  ))
)

missing_files <- c(studies$tree_file, studies$mj_file)[!file.exists(c(studies$tree_file, studies$mj_file))]
if (length(missing_files) > 0) {
  stop("Missing required input files:\n", paste(missing_files, collapse = "\n"))
}

decimal_year <- function(date_value) {
  date_value <- as.Date(date_value)
  yr <- lubridate::year(date_value)
  days <- ifelse(lubridate::leap_year(date_value), 366, 365)
  yr + (lubridate::yday(date_value) - 1) / days
}

extract_tip_dates <- function(labels) {
  raw_dates <- stringr::str_match(labels, "_(\\d{4}-\\d{2}(?:-\\d{2})?)(?:_|$)")[, 2]
  raw_dates <- ifelse(!is.na(raw_dates) & nchar(raw_dates) == 7, paste0(raw_dates, "-15"), raw_dates)
  as.Date(raw_dates)
}

make_year_bands <- function(xmin, xmax) {
  starts <- seq(floor(xmin / 5) * 5, ceiling(xmax / 5) * 5, by = 5)
  tibble::tibble(
    xmin = starts,
    xmax = starts + 5,
    band = seq_along(starts)
  ) |>
    dplyr::filter(xmin < xmax) |>
    dplyr::mutate(fill = ifelse(band %% 2 == 0, "#eeeeee", "#ffffff"))
}

make_tree_plot <- function(study_row, show_legend = FALSE) {
  td <- treeio::read.beast(study_row$tree_file)
  phy <- td@phylo
  tip_dates <- extract_tip_dates(td@phylo$tip.label)
  latest_decimal <- suppressWarnings(max(decimal_year(tip_dates), na.rm = TRUE))
  if (!is.finite(latest_decimal)) {
    latest_decimal <- 2026.4
  }

  if (!"history" %in% names(td@data)) {
    stop("No 'history' trait found in tree file: ", study_row$tree_file)
  }

  n_tip <- length(phy$tip.label)
  n_node <- phy$Nnode
  n_total <- n_tip + n_node

  ape::plot.phylo(
    phy,
    type = "phylogram",
    direction = "rightwards",
    show.tip.label = FALSE,
    plot = FALSE
  )
  ape_layout <- get("last_plot.phylo", envir = .PlotPhyloEnv)
  raw_x <- ape_layout$xx
  y <- ape_layout$yy
  tree_edge <- ape_layout$edge

  node_state <- rep(NA_character_, n_total)
  node_data <- td@data |>
    dplyr::select(node, history) |>
    dplyr::mutate(node = as.integer(node), history = as.character(history)) |>
    dplyr::filter(!is.na(node), node <= n_total)
  node_state[node_data$node] <- node_data$history
  node_state[!(node_state %in% state_levels)] <- NA_character_

  raw_tip_max <- max(raw_x[seq_len(n_tip)], na.rm = TRUE)
  x_coord <- latest_decimal - (raw_tip_max - raw_x)
  y_coord <- y
  children_by_parent <- split(tree_edge[, 2], tree_edge[, 1])

  hseg <- tibble::tibble(
    parent = tree_edge[, 1],
    child = tree_edge[, 2],
    x = x_coord[tree_edge[, 1]],
    xend = x_coord[tree_edge[, 2]],
    y = y_coord[tree_edge[, 2]],
    yend = y_coord[tree_edge[, 2]],
    state = node_state[tree_edge[, 2]]
  )

  vseg <- lapply(names(children_by_parent), function(parent) {
    parent <- as.integer(parent)
    child <- children_by_parent[[as.character(parent)]]
    tibble::tibble(
      node = parent,
      x = x_coord[parent],
      xend = x_coord[parent],
      y = min(y_coord[child], na.rm = TRUE),
      yend = max(y_coord[child], na.rm = TRUE),
      state = node_state[parent]
    )
  }) |>
    dplyr::bind_rows() |>
    dplyr::filter(node > n_tip)

  tip_df <- tibble::tibble(
    node = seq_len(n_tip),
    label = phy$tip.label,
    x = x_coord[seq_len(n_tip)],
    y = y_coord[seq_len(n_tip)],
    state = node_state[seq_len(n_tip)]
  )

  x_min <- floor(min(x_coord, na.rm = TRUE) / 5) * 5
  x_max <- ceiling(max(c(x_coord, latest_decimal, 2026.45), na.rm = TRUE))
  bands <- make_year_bands(x_min, x_max)
  y_max <- max(y_coord, na.rm = TRUE)

  p <- ggplot2::ggplot() +
    ggplot2::geom_rect(
      data = bands,
      aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = fill),
      inherit.aes = FALSE,
      color = NA,
      alpha = 1
    ) +
    ggplot2::geom_segment(
      data = vseg,
      aes(x = x, xend = xend, y = y, yend = yend, color = state),
      linewidth = 0.26,
      lineend = "butt",
      na.rm = TRUE
    ) +
    ggplot2::geom_segment(
      data = hseg,
      aes(x = x, xend = xend, y = y, yend = yend, color = state),
      linewidth = 0.26,
      lineend = "butt",
      na.rm = TRUE
    ) +
    ggplot2::geom_point(
      data = tip_df,
      aes(x = x, y = y, color = state),
      size = 0.42,
      alpha = 0.95,
      na.rm = TRUE
    ) +
    ggplot2::geom_label(
      data = tibble::tibble(
        x = x_min + 0.02 * (x_max - x_min),
        y = y_max + 0.045 * y_max,
        label = study_row$study
      ),
      aes(x = x, y = y, label = label),
      inherit.aes = FALSE,
      hjust = 0,
      vjust = 1,
      fill = study_row$label_fill,
      color = "white",
      linewidth = 0,
      fontface = "bold",
      size = 4.4
    ) +
    ggplot2::scale_color_manual(
      values = state_colors,
      breaks = state_levels,
      labels = state_labels,
      na.value = "#bdbdbd",
      name = NULL
    ) +
    ggplot2::scale_fill_identity() +
    ggplot2::scale_x_continuous(
      limits = c(x_min, x_max),
      breaks = seq(x_min, x_max, by = 5),
      expand = c(0.01, 0.01)
    ) +
    ggplot2::coord_cartesian(ylim = c(0, y_max + 0.06 * y_max), clip = "off") +
    ggplot2::theme(
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),
      legend.position = if (show_legend) c(0.20, 0.22) else "none",
      legend.background = ggplot2::element_rect(fill = "white", color = NA),
      legend.key = ggplot2::element_blank(),
      legend.text = ggplot2::element_text(size = 8.5),
      legend.justification = c(0, 0),
      axis.title.x = ggplot2::element_blank(),
      axis.title.y = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(size = 8.5, color = "black"),
      axis.text.y = ggplot2::element_blank(),
      axis.ticks.x = ggplot2::element_line(linewidth = 0.35),
      axis.ticks.y = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(color = "black", linewidth = 0.35),
      axis.line.y = ggplot2::element_blank(),
      plot.margin = ggplot2::margin(4, 4, 4, 4)
    )
  p
}

read_mj_counts <- function(study_row) {
  message("Reading MJ history: ", basename(study_row$mj_file))
  mj <- readr::read_csv(
    study_row$mj_file,
    col_types = readr::cols(
      treeId = readr::col_character(),
      startLocation = readr::col_character(),
      endLocation = readr::col_character(),
      time = readr::col_double()
    ),
    progress = FALSE
  )

  tree_ids <- sort(unique(mj$treeId))
  counts <- mj |>
    dplyr::filter(startLocation != endLocation, endLocation %in% state_levels) |>
    dplyr::count(treeId, endLocation, name = "imports")

  tidyr::expand_grid(treeId = tree_ids, endLocation = state_levels) |>
    dplyr::left_join(counts, by = c("treeId", "endLocation")) |>
    dplyr::mutate(
      imports = tidyr::replace_na(imports, 0L),
      id = study_row$id,
      study = study_row$study,
      study_short = study_row$study_short,
      violin_title = study_row$violin_title
    ) |>
    dplyr::left_join(state_info, by = c("endLocation" = "state"))
}

read_mj_yearly <- function(study_row, year_min = 2010, year_max = 2026) {
  message("Reading yearly MJ history: ", basename(study_row$mj_file))
  mj <- readr::read_csv(
    study_row$mj_file,
    col_types = readr::cols(
      treeId = readr::col_character(),
      startLocation = readr::col_character(),
      endLocation = readr::col_character(),
      time = readr::col_double()
    ),
    progress = FALSE
  )

  tree_ids <- sort(unique(mj$treeId))
  years <- seq(year_min, year_max, by = 1)
  counts <- mj |>
    dplyr::filter(
      startLocation != endLocation,
      endLocation %in% state_levels,
      is.finite(time)
    ) |>
    dplyr::mutate(year = floor(time)) |>
    dplyr::filter(year >= year_min, year <= year_max) |>
    dplyr::count(treeId, year, endLocation, name = "imports")

  tidyr::expand_grid(treeId = tree_ids, year = years, endLocation = state_levels) |>
    dplyr::left_join(counts, by = c("treeId", "year", "endLocation")) |>
    dplyr::mutate(
      imports = tidyr::replace_na(imports, 0L),
      id = study_row$id,
      study = study_row$study,
      study_short = study_row$study_short,
      violin_title = study_row$violin_title
    ) |>
    dplyr::left_join(state_info, by = c("endLocation" = "state"))
}

hpd_interval <- function(x, prob = 0.95) {
  stats::quantile(x, probs = c((1 - prob) / 2, 1 - (1 - prob) / 2), type = 8, names = FALSE)
}

message("Building tree panels...")
tree_plots <- lapply(seq_len(nrow(studies)), function(i) {
  make_tree_plot(studies[i, ], show_legend = i == 1)
})

message("Summarising Markov jump imports...")
mj_counts <- dplyr::bind_rows(lapply(seq_len(nrow(studies)), function(i) read_mj_counts(studies[i, ]))) |>
  dplyr::mutate(
    endLocation = factor(endLocation, levels = state_levels),
    state_label = factor(state_label, levels = state_info$state_label),
    plot_label = factor(plot_label, levels = state_info$plot_label),
    violin_title = factor(violin_title, levels = studies$violin_title),
    study_short = factor(study_short, levels = studies$study_short)
  )

mj_summary <- mj_counts |>
  dplyr::group_by(id, study, study_short, violin_title, endLocation, state_label, plot_label, table_label) |>
  dplyr::summarise(
    n_trees = dplyr::n(),
    mean_import = mean(imports),
    median_import = stats::median(imports),
    hpd_2_5 = hpd_interval(imports)[1],
    hpd_97_5 = hpd_interval(imports)[2],
    .groups = "drop"
  ) |>
  dplyr::arrange(study_short, endLocation)

readr::write_csv(mj_counts, file.path(out_dir, "MJ_import_counts_per_tree_2026.csv"))
readr::write_csv(mj_summary, file.path(out_dir, "MJ_import_summary_2026.csv"))

summary_table <- mj_summary |>
  dplyr::mutate(mean_import = sprintf("%.1f", mean_import)) |>
  dplyr::select(Study = study_short, table_label, mean_import) |>
  tidyr::pivot_wider(names_from = table_label, values_from = mean_import)

table_grob <- gridExtra::tableGrob(
  summary_table,
  rows = NULL,
  theme = gridExtra::ttheme_minimal(
    core = list(fg_params = list(fontsize = 7.3), padding = unit(c(2.5, 2.5), "pt")),
    colhead = list(fg_params = list(fontsize = 7.1, fontface = "bold"), padding = unit(c(2.5, 2.5), "pt"))
  )
)

p_violin <- ggplot2::ggplot(mj_counts, aes(x = plot_label, y = imports, fill = endLocation)) +
  ggplot2::geom_violin(width = 0.78, scale = "width", trim = TRUE, color = "#5f5f5f", linewidth = 0.25) +
  ggplot2::geom_point(
    data = mj_summary,
    aes(x = plot_label, y = mean_import),
    inherit.aes = FALSE,
    size = 1.7,
    shape = 21,
    fill = "white",
    color = "#4b4b4b",
    stroke = 0.35
  ) +
  ggplot2::facet_wrap(~violin_title, ncol = 2) +
  ggplot2::scale_fill_manual(values = state_colors, guide = "none") +
  ggplot2::labs(x = NULL, y = "Markov jump imports per tree") +
  ggplot2::theme_bw(base_size = 9) +
  ggplot2::theme(
    strip.background = ggplot2::element_rect(fill = "white", color = "#cfcfcf", linewidth = 0.35),
    strip.text = ggplot2::element_text(size = 9),
    panel.grid.major.x = ggplot2::element_blank(),
    panel.grid.minor = ggplot2::element_blank(),
    panel.border = ggplot2::element_rect(color = "#cfcfcf", fill = NA, linewidth = 0.45),
    axis.text.x = ggplot2::element_text(size = 7.2),
    axis.text.y = ggplot2::element_text(size = 8),
    axis.title.y = ggplot2::element_text(size = 9),
    plot.margin = ggplot2::margin(3, 3, 3, 3)
  )

p_table_title <- ggplot2::ggplot() +
  ggplot2::annotate("text", x = 0.5, y = 0.5, label = "Posterior mean MJ imports", size = 3.2) +
  ggplot2::theme_void()

mj_panel <- p_violin / (p_table_title / patchwork::wrap_elements(table_grob)) +
  patchwork::plot_layout(heights = c(4.1, 1.15))

combined <- (tree_plots[[1]] | tree_plots[[2]] | tree_plots[[3]]) /
  (tree_plots[[4]] | tree_plots[[5]] | mj_panel) +
  patchwork::plot_layout(heights = c(1, 1), widths = c(1, 1, 1))

pdf_file <- file.path(out_dir, "Euro_resurgence_4dta_visualization_2026.pdf")
png_file <- file.path(out_dir, "Euro_resurgence_4dta_visualization_2026.png")

message("Saving figure...")
ggplot2::ggsave(pdf_file, combined, width = 21, height = 13, units = "in", device = cairo_pdf)
ggplot2::ggsave(png_file, combined, width = 21, height = 13, units = "in", dpi = 260)

message("Building optional yearly MJ import dynamics...")
mj_yearly <- dplyr::bind_rows(lapply(seq_len(nrow(studies)), function(i) read_mj_yearly(studies[i, ]))) |>
  dplyr::mutate(
    endLocation = factor(endLocation, levels = state_levels),
    plot_label = factor(plot_label, levels = state_info$plot_label),
    violin_title = factor(violin_title, levels = studies$violin_title),
    study_short = factor(study_short, levels = studies$study_short)
  )

mj_yearly_summary <- mj_yearly |>
  dplyr::group_by(id, study, study_short, violin_title, endLocation, state_label, plot_label, year) |>
  dplyr::summarise(
    n_trees = dplyr::n(),
    mean_import = mean(imports),
    hpd_2_5 = hpd_interval(imports)[1],
    hpd_97_5 = hpd_interval(imports)[2],
    .groups = "drop"
  )

readr::write_csv(mj_yearly_summary, file.path(out_dir, "MJ_imports_by_calendar_year_2026.csv"))

p_yearly <- ggplot2::ggplot(
  mj_yearly_summary,
  aes(x = year, y = mean_import, color = endLocation, fill = endLocation)
) +
  ggplot2::geom_ribbon(aes(ymin = hpd_2_5, ymax = hpd_97_5), alpha = 0.12, color = NA) +
  ggplot2::geom_line(linewidth = 0.65) +
  ggplot2::facet_wrap(~violin_title, ncol = 1) +
  ggplot2::scale_color_manual(values = state_colors, breaks = state_levels, labels = state_labels, name = NULL) +
  ggplot2::scale_fill_manual(values = state_colors, breaks = state_levels, labels = state_labels, name = NULL) +
  ggplot2::scale_x_continuous(breaks = seq(2010, 2026, by = 2), limits = c(2010, 2026)) +
  ggplot2::labs(x = NULL, y = "Mean MJ imports per posterior tree-year") +
  ggplot2::theme_bw(base_size = 10) +
  ggplot2::theme(
    legend.position = "top",
    panel.grid.minor = ggplot2::element_blank(),
    strip.background = ggplot2::element_rect(fill = "white", color = "#cfcfcf", linewidth = 0.35),
    strip.text = ggplot2::element_text(size = 10),
    plot.margin = ggplot2::margin(6, 6, 6, 6)
  )

ggplot2::ggsave(
  file.path(out_dir, "MJ_imports_by_calendar_year_2026.pdf"),
  p_yearly,
  width = 8.5,
  height = 11,
  units = "in",
  device = cairo_pdf
)
ggplot2::ggsave(
  file.path(out_dir, "MJ_imports_by_calendar_year_2026.png"),
  p_yearly,
  width = 8.5,
  height = 11,
  units = "in",
  dpi = 240
)

message("Saved:")
message("  ", pdf_file)
message("  ", png_file)
message("  ", file.path(out_dir, "MJ_import_summary_2026.csv"))
message("  ", file.path(out_dir, "MJ_imports_by_calendar_year_2026.pdf"))

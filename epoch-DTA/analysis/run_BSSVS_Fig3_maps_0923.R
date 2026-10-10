library(dplyr)
library(lubridate)
library(ggplot2)
library(tidyr)
library(readr)
library(patchwork)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(tibble)
library(maps)
library(ape)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
stopifnot(length(script_arg) == 1L)
base_dir <- normalizePath(dirname(sub("^--file=", "", script_arg)), winslash = "/", mustWork = TRUE)
comb_dir <- file.path(base_dir, "combined")
out_dir  <- file.path(base_dir, "figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

mj_file <- file.path(comb_dir, "bssvs13_uncal_MJhistory.txt")
bf_file <- file.path(comb_dir, "BSSVS13_region_Bayes.factor.test.result.csv")
tree_file <- file.path(comb_dir, "bssvs13_uncal_HIPSTR.tre")
BF_MIN  <- 3

start_date <- as.Date("2011-01-01")
end_date <- as.Date("2026-05-03")

name_map_display <- c(
  "China" = "China",
  "Eastern_Europe" = "Eastern Europe",
  "Western_Europe" = "Western Europe",
  "Northern_Europe" = "North Europe",
  "North_Europe" = "North Europe",
  "South_Europe" = "South Europe",
  "North_America" = "North America",
  "South_America" = "South America",
  "North_Africa" = "North Africa",
  "Sub-Saharan_Africa" = "Sub-Saharan Africa",
  "South_Asia" = "South Asia",
  "West_Asia" = "West Asia",
  "Japan/Korea" = "Japan/Korea",
  "Oceania" = "Oceania"
)

coord_df <- tibble::tribble(
  ~deme,                  ~long,  ~lati,
  "China",                104.0,  35.0,
  "Japan/Korea",          135.0,  36.0,
  "South Asia",            78.0,  22.0,
  "Eastern Europe",        25.0,  50.0,
  "Western Europe",         5.0,  50.0,
  "North Europe",          15.0,  60.0,
  "South Europe",          15.0,  43.0,
  "North America",       -100.0,  45.0,
  "South America",        -60.0, -15.0,
  "North Africa",          15.0,  30.0,
  "Sub-Saharan Africa",    20.0,   0.0,
  "West Asia",             45.0,  30.0,
  "Oceania",              135.0, -25.0
)

pal_named <- c(
  "Western Europe" = "#e76f51",
  "South Europe" = "#2a9d8f",
  "North Europe" = "#457b9d",
  "Eastern Europe" = "#1d3557",
  "North America" = "#6a4c93",
  "South America" = "#8ab17d",
  "China" = "#8a171a",
  "Japan/Korea" = "#f4a261",
  "South Asia" = "#2d6a4f",
  "West Asia" = "#277da1",
  "North Africa" = "#f77f00",
  "Sub-Saharan Africa" = "#d62828",
  "Oceania" = "#8d99ae"
)

summary_tree <- ape::read.nexus(tree_file)
stopifnot(inherits(summary_tree, "phylo"), length(summary_tree$tip.label) == 3794L)
tip_label_suffixes <- sub(".*_", "", summary_tree$tip.label)

bf_tbl <- readr::read_csv(bf_file, show_col_types = FALSE) %>%
  mutate(startLocation = recode(start_name, !!!name_map_display),
         endLocation = recode(end_name, !!!name_map_display))
stopifnot(!anyNA(bf_tbl), nrow(bf_tbl) == 13L * 12L,
          !anyDuplicated(bf_tbl[c("startLocation", "endLocation")]),
          all(bf_tbl$startLocation != bf_tbl$endLocation),
          setequal(bf_tbl$startLocation, coord_df$deme),
          setequal(bf_tbl$endLocation, coord_df$deme))
keep_routes <- bf_tbl %>% filter(bayes_factor > BF_MIN) %>%
  select(startLocation, endLocation)

jump_raw <- readr::read_csv(mj_file, show_col_types = FALSE) %>%
  mutate(treeId = as.character(treeId),
         startLocation = recode(startLocation, !!!name_map_display),
         endLocation = recode(endLocation, !!!name_map_display),
         date = as.Date(lubridate::date_decimal(time), tz = "UTC"))
stopifnot(!anyNA(jump_raw), all(jump_raw$startLocation != jump_raw$endLocation),
          setequal(c(jump_raw$startLocation, jump_raw$endLocation), coord_df$deme))
trees <- sort(unique(jump_raw$treeId))
locs <- coord_df$deme
jump <- jump_raw %>% filter(date >= start_date, date <= end_date) %>%
  semi_join(keep_routes, by = c("startLocation", "endLocation"))
message("BF > ", BF_MIN, ": ", nrow(keep_routes), " / ", nrow(bf_tbl),
        " directed routes; ", length(trees), " histories; ", nrow(jump), " events in window.")

period_def <- tibble::tribble(
  ~period,           ~start,                ~end,
  "pre-resurgence",  as.Date("2011-01-01"), as.Date("2018-01-31"),
  "resurgence",      as.Date("2018-02-01"), as.Date("2020-03-31"),
  "pandemic",        as.Date("2020-04-01"), as.Date("2023-04-30"),
  "post-pandemic",   as.Date("2023-05-01"), as.Date("2026-05-03")
) %>%
  mutate(years = time_length(interval(start, end + 1), unit = "years"))

period_levels <- period_def$period

jump_period <- jump %>%
  mutate(period = cut(date, breaks = c(period_def$start, end_date + 1),
                      labels = period_def$period, right = FALSE)) %>%
  filter(!is.na(period))

jump_tree <- jump_period %>%
  group_by(treeId, period, startLocation, endLocation) %>%
  summarise(n = n(), .groups = "drop") %>%
  filter(startLocation != endLocation)


jump_full <- jump_tree %>%
  ungroup() %>%
  mutate(period = as.character(period)) %>%
  complete(treeId = trees, period = period_levels,
           startLocation = locs, endLocation = locs, fill = list(n = 0)) %>%
  filter(startLocation != endLocation) %>%
  semi_join(keep_routes, by = c("startLocation", "endLocation")) %>%
  left_join(period_def %>% select(period, years), by = "period") %>%
  mutate(rate = n / years)

edges_4period <- jump_full %>%
  group_by(period, startLocation, endLocation) %>%
  summarise(annual_intensity_mean = mean(rate), .groups = "drop") %>%
  filter(annual_intensity_mean > 0) %>%
  left_join(bf_tbl %>% select(startLocation, endLocation, bayes_factor, posterior_probability),
            by = c("startLocation", "endLocation")) %>%
  left_join(coord_df %>% transmute(startLocation = deme, long.orig = long, lati.orig = lati),
            by = "startLocation") %>%
  left_join(coord_df %>% transmute(endLocation = deme, long.dest = long, lati.dest = lati),
            by = "endLocation") %>%
  mutate(period = factor(period, levels = period_levels)) %>%
  identity()
stopifnot(!anyNA(edges_4period), all(edges_4period$bayes_factor > BF_MIN))

origin_strength <- edges_4period %>%
  group_by(period, startLocation, long.orig, lati.orig) %>%
  summarise(origin_ai = sum(annual_intensity_mean), .groups = "drop")

net_export <- full_join(
  edges_4period %>% group_by(period, deme = startLocation) %>%
    summarise(out_ai = sum(annual_intensity_mean), .groups = "drop"),
  edges_4period %>% group_by(period, deme = endLocation) %>%
    summarise(in_ai = sum(annual_intensity_mean), .groups = "drop"),
  by = c("period", "deme")
) %>%
  mutate(out_ai = coalesce(out_ai, 0), in_ai = coalesce(in_ai, 0),
         net_ai = out_ai - in_ai,
         period = factor(period, levels = period_levels))

region_from_ne <- function(iso3, continent, subregion) {
  dplyr::case_when(
    iso3 %in% c("CHN", "HKG", "MAC", "TWN") ~ "China",
    iso3 %in% c("JPN", "KOR", "PRK") ~ "Japan/Korea",
    continent == "Oceania" ~ "Oceania",
    continent == "South America" ~ "South America",
    continent == "North America" ~ "North America",
    continent == "Africa" & subregion == "Northern Africa" ~ "North Africa",
    continent == "Africa" & subregion != "Northern Africa" ~ "Sub-Saharan Africa",
    continent == "Europe" & subregion == "Eastern Europe" ~ "Eastern Europe",
    continent == "Europe" & subregion == "Western Europe" ~ "Western Europe",
    continent == "Europe" & subregion == "Northern Europe" ~ "North Europe",
    continent == "Europe" & subregion == "Southern Europe" ~ "South Europe",
    continent == "Asia" & subregion %in% c("Southern Asia", "South-Eastern Asia") ~ "South Asia",
    continent == "Asia" & subregion %in% c("Western Asia", "Central Asia") ~ "West Asia",
    TRUE ~ NA_character_
  )
}

world_country_sf <- rnaturalearth::ne_countries(scale = "medium", returnclass = "sf") %>%
  mutate(ISO3 = if_else(iso_a3 == "SDS", "SSD", iso_a3),
         region = region_from_ne(ISO3, continent, subregion))

southeast_asia_check <- world_country_sf %>%
  filter(subregion == "South-Eastern Asia") %>%
  st_drop_geometry() %>% select(admin, ISO3, subregion, region) %>% arrange(admin)
if (any(is.na(southeast_asia_check$region))) {
  stop("Unassigned South-Eastern Asia country: ",
       paste(southeast_asia_check$admin[is.na(southeast_asia_check$region)], collapse = ", "))
}

world_region_sf <- world_country_sf %>%
  filter(!is.na(region)) %>% st_make_valid() %>%
  group_by(region) %>% summarise(geometry = st_union(geometry), .groups = "drop") %>%
  st_make_valid()

region_boundary_to_df <- function(sf_obj) {
  pieces <- lapply(seq_len(nrow(sf_obj)), function(i) {
    geom <- st_boundary(sf_obj$geometry[i])
    xy <- st_coordinates(geom)
    level_cols <- intersect(c("L1", "L2", "L3"), colnames(xy))
    part <- if (length(level_cols) > 0) {
      apply(xy[, level_cols, drop = FALSE], 1, paste, collapse = "_")
    } else rep("1", nrow(xy))
    data.frame(long = xy[, "X"], lat = xy[, "Y"], region = sf_obj$region[i],
               part = paste(sf_obj$region[i], part, sep = "_"), stringsAsFactors = FALSE)
  })
  bind_rows(pieces) %>%
    filter(lat >= -60, lat <= 85) %>%
    group_by(part) %>%
    mutate(split_piece = cumsum(if_else(is.na(lag(long)), 0L,
                                        as.integer(abs(long - lag(long)) > 30))),
           map_group = paste(part, split_piece, sep = "_")) %>%
    ungroup()
}

region_outline <- region_boundary_to_df(world_region_sf)

edge_max <- max(edges_4period$annual_intensity_mean, na.rm = TRUE)
point_max <- max(origin_strength$origin_ai, na.rm = TRUE)

p_map_labels <- c("pre-resurgence" = "Epoch 1: Pre-resurgence",
                  "resurgence" = "Epoch 2: Resurgence",
                  "pandemic" = "Epoch 3: Pandemic",
                  "post-pandemic" = "Epoch 4: Post-pandemic")

p_map_4 <- ggplot() +
  geom_path(data = region_outline, aes(x = long, y = lat, group = map_group),
            color = "black", linewidth = 0.22) +
  geom_curve(data = edges_4period,
             aes(x = long.orig, y = lati.orig, xend = long.dest, yend = lati.dest,
                 linewidth = annual_intensity_mean, color = startLocation),
             curvature = 0.4, alpha = 0.5,
             arrow = grid::arrow(length = grid::unit(0.075, "cm"), type = "closed")) +
  geom_point(data = origin_strength,
             aes(x = long.orig, y = lati.orig, size = origin_ai, fill = startLocation),
             shape = 21, color = "black", stroke = 0.25, alpha = 0.9) +
  scale_color_manual(values = pal_named, guide = "none") +
  scale_fill_manual(values = pal_named, guide = "none") +
  scale_linewidth_continuous("Annual intensity (events / year)",
                             range = c(0.5, 3), limits = c(0, edge_max)) +
  scale_size_area("Origin intensity", max_size = 5, limits = c(0, point_max)) +
  coord_quickmap(xlim = c(-180, 180), ylim = c(-60, 85), expand = FALSE) +
  facet_wrap(~ period, ncol = 2, labeller = as_labeller(p_map_labels)) +
  theme_void() +
  theme(strip.text = element_text(face = "plain", size = 14, color = "black"),
        panel.spacing = unit(0.25, "lines"),
        legend.position = "none",
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        legend.background = element_rect(fill = "white", color = NA))

map_one_period <- function(pr) {
  ggplot() +
    geom_path(data = region_outline, aes(x = long, y = lat, group = map_group),
              color = "black", linewidth = 0.22) +
    geom_curve(data = edges_4period %>% filter(period == pr),
               aes(x = long.orig, y = lati.orig, xend = long.dest, yend = lati.dest,
                   linewidth = annual_intensity_mean, color = startLocation),
               curvature = 0.4, alpha = 0.5,
             arrow = grid::arrow(length = grid::unit(0.075, "cm"), type = "closed")) +
    geom_point(data = origin_strength %>% filter(period == pr),
               aes(x = long.orig, y = lati.orig, size = origin_ai, fill = startLocation),
               shape = 21, color = "black", stroke = 0.25, alpha = 0.9) +
    scale_color_manual(values = pal_named, guide = "none") +
    scale_fill_manual(values = pal_named, guide = "none") +
    scale_linewidth_continuous(range = c(0.5, 3), limits = c(0, edge_max), guide = "none") +
    scale_size_area(max_size = 5, limits = c(0, point_max), guide = "none") +
    coord_quickmap(xlim = c(-180, 180), ylim = c(-60, 85), expand = FALSE) +
    labs(title = unname(p_map_labels[pr])) +
    theme_void() +
    theme(plot.title = element_text(size = 14, hjust = 0.5, margin = margin(b = 2)),
          legend.position = "none",
          plot.background = element_rect(fill = "white", color = NA),
          panel.background = element_rect(fill = "white", color = NA),
          plot.margin = margin(2, 2, 2, 0, "mm"))
}

legend_values <- function(mx) {
  vals <- pretty(c(0, mx), n = 6)
  vals <- vals[vals > 0 & vals <= mx]
  if (!length(vals)) vals <- mx
  head(vals, 4)
}
point_values <- legend_values(point_max)
line_values <- legend_values(edge_max)
legend_points <- tibble(x = 0.20, y = seq(0.60, 0.37, length.out = length(point_values)),
                        value = point_values)
legend_lines <- tibble(x = 0.10, xend = 0.43,
                       y = seq(0.19, -0.02, length.out = length(line_values)), value = line_values)
p_custom_legend <- ggplot() +
  annotate("text", x = 0.5, y = 0.95, label = "Transition\ndirection",
           size = 4.2, lineheight = 0.95) +
  geom_curve(aes(x = 0.18, y = 0.85, xend = 0.82, yend = 0.85), curvature = 0.45,
             arrow = grid::arrow(length = grid::unit(0.12, "cm"), type = "closed"),
             linewidth = 0.4, color = "grey30") +
  annotate("text", x = 0.5, y = 0.78, label = "Colour: origin", size = 3.7) +
  annotate("text", x = 0.50, y = 0.69, label = "Outgoing intensity\n(events / year)",
           size = 4.0, lineheight = 0.95) +
  geom_point(data = legend_points, aes(x = x, y = y, size = value),
             shape = 21, fill = "white", color = "black", stroke = 0.25) +
  geom_text(data = legend_points, aes(x = 0.50, y = y, label = value), hjust = 0, size = 3.9) +
  annotate("text", x = 0.50, y = 0.285, label = "Route intensity\n(events / year)",
           size = 4.0, lineheight = 0.95) +
  geom_segment(data = legend_lines, aes(x = x, xend = xend, y = y, yend = y, linewidth = value),
               color = "grey50", lineend = "butt") +
  geom_text(data = legend_lines, aes(x = 0.57, y = y, label = value), hjust = 0, size = 3.9) +
  annotate("text", x = 0.5, y = -0.11,
           label = paste0("BF > ", BF_MIN, "\n", nrow(keep_routes), " / ", nrow(bf_tbl), " routes"),
           size = 3.7, lineheight = 0.95) +
  scale_size_area(max_size = 5, limits = c(0, point_max), guide = "none") +
  scale_linewidth_continuous(range = c(0.5, 3), limits = c(0, edge_max), guide = "none") +
  coord_cartesian(xlim = c(0, 1), ylim = c(-0.17, 1), clip = "off") +
  theme_void() +
  theme(plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(0, 3, 0, 3, "mm"))

p_map_4_with_legend <- p_map_4 + p_custom_legend + patchwork::plot_layout(widths = c(1, 0.17))

region_levels <- c("China", "Japan/Korea", "South Asia",
                   "Eastern Europe", "Western Europe", "North Europe", "South Europe",
                   "North America", "South America", "North Africa", "Sub-Saharan Africa",
                   "West Asia", "Oceania")

net_export_bar <- net_export %>%
  mutate(start_display = factor(deme, levels = region_levels))

side_net_bar <- function(pr) {
  df <- net_export_bar %>% filter(period == pr)
  rng <- range(c(df$net_ai, 0), na.rm = TRUE)
  pad <- max(diff(rng), 1e-6) * 0.08
  ggplot(df, aes(x = start_display, y = net_ai, fill = start_display)) +
    geom_col(width = 0.72) +
    annotate("segment", x = 0.45, xend = nlevels(df$start_display) + 0.55,
             y = 0, yend = 0, linewidth = 0.25, colour = "black") +
    coord_flip() +
    scale_x_discrete(expand = expansion(add = c(1.5, 5.0)), drop = FALSE) +
    scale_y_continuous(limits = c(rng[1] - pad, rng[2] + pad),
                       expand = c(0, 0),
                       breaks = scales::breaks_pretty(n = 4)) +
    scale_fill_manual(values = pal_named, guide = "none") +
    labs(x = NULL, y = "Net export\n(events / year)") +
    theme_minimal(base_size = 10) +
    theme(panel.grid = element_blank(),
          axis.text.y = element_text(size = 9, color = "black", hjust = 1,
                                     margin = margin(r = 1.5)),
          axis.text.x = element_text(size = 9, color = "black"),
          axis.title.x = element_text(size = 10, color = "black",
                                      margin = margin(t = 1.5)),
          axis.line.x = element_line(linewidth = 0.3, color = "black"),
          axis.ticks.x = element_line(linewidth = 0.3, color = "black"),
          axis.ticks.length = unit(0.07, "cm"),
          plot.background = element_rect(fill = "white", color = NA),
          panel.background = element_rect(fill = "white", color = NA),
          plot.margin = margin(4, 1, 4, 16, "mm"))
}

maps_each <- lapply(period_levels, map_one_period)
bars_side <- lapply(period_levels, side_net_bar)

combined_design <- "ABCDI\nEFGHI"

p_combined <- wrap_plots(
  A = bars_side[[1]], B = maps_each[[1]], C = bars_side[[2]], D = maps_each[[2]],
  E = bars_side[[3]], F = maps_each[[3]], G = bars_side[[4]], H = maps_each[[4]],
  I = p_custom_legend,
  design = combined_design,
  widths = c(0.40, 1, 0.40, 1, 0.25)
) & theme(plot.background = element_rect(fill = "white", color = NA))

stopifnot(all(abs(net_export %>% group_by(period) %>%
  summarise(balance = sum(net_ai)) %>% pull(balance)) < 1e-8))
stopifnot(nrow(anti_join(edges_4period, keep_routes,
  by = c("startLocation", "endLocation"))) == 0)
stopifnot(all(jump_full %>% count(period, startLocation, endLocation) %>%
  pull(n) == length(trees)))
out <- function(stem, ext = "csv") file.path(out_dir, paste0(stem, "_0923.", ext))
write_csv(edges_4period, out("BSSVS_MJ_edges_4periods_BFgt3"))
write_csv(origin_strength, out("BSSVS_MJ_origin_strength_4periods_BFgt3"))
write_csv(net_export, out("BSSVS_MJ_net_export_4periods_BFgt3"))
write_csv(period_def, out("BSSVS_MJ_period_definitions"))
write_csv(bf_tbl %>% filter(bayes_factor > BF_MIN), out("BSSVS_routes_BFgt3"))
write_csv(tibble(input = c(mj_file, bf_file, tree_file),
                 md5 = unname(tools::md5sum(c(mj_file, bf_file, tree_file))),
                 role = c("Posterior events: means include all input history IDs",
                          "Global directed-route filter: BF strictly greater than 3",
                          "Matching HIPSTR summary: taxon-count and label audit only")),
          out("BSSVS_map_source_provenance"))
write_csv(tibble(metric = c("histories", "tree_taxa", "tree_tip_label_suffix_count", "total_routes",
                            "BF_gt_3_routes", "events_in_window", "annual_denominator"),
                 value = c(length(trees), length(summary_tree$tip.label),
                           length(unique(tip_label_suffixes)), nrow(bf_tbl), nrow(keep_routes),
                           nrow(jump), "inclusive start/end calendar dates via interval(start,end+1)")),
          out("BSSVS_map_QC"))
save_both <- function(plot, stem, w, h) {
  ggsave(out(stem, "pdf"), plot, width = w, height = h, units = "in",
         device = grDevices::cairo_pdf, family = "Arial")
  ggsave(out(stem, "png"), plot, width = w, height = h, units = "in", dpi = 300)
}
save_both(p_map_4_with_legend, "BSSVS_Fig3_spatial_network_4periods_BFgt3", 14, 6.8)
save_both(p_combined, "BSSVS_Fig3_network_netexport_4periods_BFgt3", 20, 7.2)
print(edges_4period %>% count(period))
message("Saved maps, posterior-mean summaries and source audit to: ", out_dir)

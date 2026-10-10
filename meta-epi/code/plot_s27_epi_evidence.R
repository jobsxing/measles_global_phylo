suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(sf)
  library(rnaturalearth)
  library(ggplot2)
  library(patchwork)
  library(scales)
})
args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep('^--file=', args, value = TRUE)
script_path <- normalizePath(sub('^--file=', '', script_arg[[1]]), winslash = '/')
project_dir <- normalizePath(file.path(dirname(script_path), '..'), winslash = '/')
data_path <- file.path(project_dir, 'data', 'measles_importation_combined_index_v2.csv')
output_dir <- file.path(project_dir, 'figures')
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
index <- read_csv(data_path, show_col_types = FALSE, progress = FALSE)
required <- c('source_database', 'source_iso3_norm', 'sink_iso3_norm', 'sink_country_norm', 'event_date_norm', 'plot_weight', 'source_region')
missing_columns <- setdiff(required, names(index))
if (length(missing_columns)) stop('Missing columns: ', paste(missing_columns, collapse = ', '))
plot_data <- index |>
  mutate(
    event_date = as.Date(event_date_norm),
    source_region = if_else(is.na(source_region) | source_region == '', 'Unknown source', source_region),
    database_outline = if_else(source_database == 'epi_analysis', 'epi_analysis', 'external_database'),
    sink_label = recode(sink_country_norm, 'Democratic Republic of the Congo' = 'DR Congo')
  ) |>
  filter(!is.na(event_date), event_date >= as.Date('2011-01-01'), !is.na(sink_iso3_norm))
source_levels <- c('Europe eliminated', 'Asia eliminated', 'Americas eliminated', 'Oceania eliminated', 'Endemic region', 'Unclassified', 'Unknown source')
plot_data$source_region <- factor(plot_data$source_region, levels = source_levels)
region_colors <- c(
  'Europe eliminated' = '#5FB6AB',
  'Asia eliminated' = '#D97B7E',
  'Americas eliminated' = '#407AB2',
  'Oceania eliminated' = '#B46A50',
  'Endemic region' = '#AEB7BF',
  'Unclassified' = '#D7C8A8',
  'Unknown source' = '#8B8B8B'
)
frequency_colors <- c('0' = '#CDC9C8', '1-2' = '#D8DBE2', '3-4' = '#A9BCD0', '5-6' = '#58A4B0', '7-8' = '#2B83A3', '>8' = '#065A82')
year_colors <- c('#C51B7D', '#E9A3C9', '#FDE0EF', '#E6F5D0', '#A1D76A', '#4D9221')
sink_frequency <- plot_data |> count(sink_iso3_norm, name = 'import_frequency')
sink_size <- plot_data |> group_by(sink_iso3_norm) |> summarise(total_size = sum(plot_weight, na.rm = TRUE), .groups = 'drop')
world <- ne_countries(scale = 'medium', returnclass = 'sf') |>
  filter(admin != 'Antarctica') |>
  st_transform(4326)
world_plot <- world |>
  mutate(iso3c = iso_a3) |>
  left_join(sink_frequency, by = c('iso3c' = 'sink_iso3_norm')) |>
  mutate(import_frequency = replace_na(import_frequency, 0L))
china_frequency <- world_plot$import_frequency[world_plot$iso3c == 'CHN']
if (length(china_frequency) == 1) {
  world_plot <- world_plot |> mutate(import_frequency = if_else(iso3c %in% c('HKG', 'MAC', 'TWN'), china_frequency, import_frequency))
}
world_plot <- world_plot |>
  mutate(frequency_bin = cut(import_frequency, breaks = c(-0.5, 0.5, 2.5, 4.5, 6.5, 8.5, Inf), labels = names(frequency_colors), right = TRUE))
centroid_sf <- st_transform(world, 6933)
centroid_sf <- suppressWarnings(st_point_on_surface(centroid_sf))
centroid_sf <- st_transform(centroid_sf, 4326)
coordinates <- bind_cols(
  st_drop_geometry(centroid_sf) |> transmute(iso3c = iso_a3),
  as_tibble(st_coordinates(centroid_sf)) |> setNames(c('lon', 'lat'))
) |>
  bind_rows(tibble(iso3c = c('XKX', 'TWN'), lon = c(20.9, 120.9), lat = c(42.6, 23.7))) |>
  distinct(iso3c, .keep_all = TRUE)
routes <- plot_data |>
  mutate(event_id = row_number(), event_date_num = as.numeric(event_date)) |>
  left_join(coordinates, by = c('source_iso3_norm' = 'iso3c')) |>
  rename(lon_source = lon, lat_source = lat) |>
  left_join(coordinates, by = c('sink_iso3_norm' = 'iso3c')) |>
  rename(lon_sink = lon, lat_sink = lat) |>
  filter(is.finite(lon_source), is.finite(lat_source), is.finite(lon_sink), is.finite(lat_sink))
bubbles <- sink_size |>
  left_join(coordinates, by = c('sink_iso3_norm' = 'iso3c')) |>
  filter(is.finite(lon), is.finite(lat), total_size > 0)
year_breaks <- as.numeric(as.Date(paste0(seq(2012, 2026, 2), '-01-01')))
p_map <- ggplot() +
  geom_sf(data = world, fill = '#F1F1F1', color = 'white', linewidth = 0.12) +
  geom_sf(data = world_plot, aes(fill = frequency_bin), color = 'white', linewidth = 0.12) +
  geom_curve(
    data = routes,
    aes(x = lon_source, y = lat_source, xend = lon_sink, yend = lat_sink, color = event_date_num),
    curvature = 0.48,
    linewidth = 0.42,
    alpha = 0.82,
    lineend = 'round',
    arrow = grid::arrow(length = grid::unit(5, 'pt'), type = 'open', ends = 'last')
  ) +
  geom_point(data = bubbles, aes(lon, lat, size = total_size), shape = 21, fill = NA, color = 'black', stroke = 0.32) +
  scale_fill_manual(values = frequency_colors, drop = FALSE, name = 'Import frequency') +
  scale_color_gradientn(colours = year_colors, breaks = year_breaks, labels = format(as.Date(year_breaks, origin = '1970-01-01'), '%Y'), name = 'Transmission date') +
  scale_size_continuous(name = 'Total outbreak/import size', trans = 'log10', range = c(1.5, 6.5), breaks = c(10, 100, 1000), labels = comma) +
  coord_sf(xlim = c(-180, 180), ylim = c(-58, 85), expand = FALSE) +
  labs(title = 'A   Documented measles cross-border transmission evidence') +
  guides(
    fill = guide_legend(order = 1, title.position = 'top', nrow = 2),
    color = guide_colorbar(order = 2, title.position = 'top', barwidth = grid::unit(190, 'pt'), barheight = grid::unit(6, 'pt')),
    size = guide_legend(order = 3, title.position = 'top', nrow = 1)
  ) +
  theme_void(base_size = 10, base_family = 'Arial') +
  theme(
    plot.title = element_text(face = 'bold', size = 12, margin = margin(b = 3)),
    legend.position = 'bottom',
    legend.box = 'horizontal',
    legend.text = element_text(size = 7.5),
    legend.title = element_text(size = 8.5),
    plot.margin = margin(2, 4, 2, 4)
  )
sink_order <- c('Netherlands', 'Israel', 'Germany', 'Ethiopia', 'Croatia', 'Portugal', 'Colombia', 'Brazil', 'Taiwan', 'Italy', 'Japan', 'Belgium', 'United Kingdom', 'Ireland', 'Poland', 'Czechia', 'Canada', 'China', 'South Korea', 'Australia', 'Turkey', 'Democratic Republic of the Congo', 'Chile', 'United States')
plot_time <- plot_data |>
  mutate(
    sink_label = factor(sink_label, levels = rev(recode(sink_order, 'Democratic Republic of the Congo' = 'DR Congo'))),
    database_outline = factor(database_outline, levels = c('epi_analysis', 'external_database'))
  )
epoch_background <- tibble(
  xmin = as.Date(c('2018-02-01', '2020-04-01')),
  xmax = as.Date(c('2020-03-31', '2023-04-30')),
  fill = c('#F7E9D7', '#E7F0F6')
)
p_time <- ggplot(plot_time, aes(event_date, sink_label)) +
  geom_rect(data = epoch_background, aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf), inherit.aes = FALSE, fill = epoch_background$fill, alpha = 0.58) +
  geom_vline(xintercept = as.Date(c('2018-02-01', '2020-03-31', '2020-04-01', '2023-04-30')), color = '#B8B8B8', linewidth = 0.3, linetype = 'dashed') +
  geom_point(aes(fill = source_region, color = database_outline, size = plot_weight), shape = 21, stroke = 0.55, alpha = 0.95, position = position_jitter(height = 0.12, width = 0, seed = 2026)) +
  scale_fill_manual(values = region_colors, drop = FALSE, name = 'Source region') +
  scale_color_manual(values = c('epi_analysis' = '#F03B3B', 'external_database' = '#8D8D8D'), labels = c('epi_analysis' = 'epi meta-analysis', 'external_database' = 'JHU/CDC external database'), name = 'Database outline') +
  scale_size_continuous(trans = 'log10', range = c(1.4, 6.4), breaks = c(10, 100, 1000), guide = 'none') +
  scale_x_date(breaks = as.Date(paste0(2011:2026, '-01-01')), labels = 2011:2026, limits = as.Date(c('2011-01-01', '2026-12-31')), expand = expansion(mult = c(0.01, 0.01))) +
  labs(title = 'B   Timing of documented import-associated outbreaks', x = 'Report date', y = 'Outbreak country') +
  guides(fill = guide_legend(order = 1, override.aes = list(shape = 21, size = 3, color = '#4D4D4D')), color = guide_legend(order = 2, override.aes = list(shape = 21, size = 3, fill = 'white'))) +
  theme_minimal(base_size = 9, base_family = 'Arial') +
  theme(
    plot.title = element_text(face = 'bold', size = 11),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
    legend.position = 'right',
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7),
    plot.margin = margin(3, 3, 3, 3)
  )
sink_levels <- c('Europe eliminated', 'Asia eliminated', 'Americas eliminated', 'Oceania eliminated', 'Endemic region')
source_levels <- c('Unknown source', 'Unclassified', 'Endemic region', 'Oceania eliminated', 'Americas eliminated', 'Asia eliminated', 'Europe eliminated')
flow <- read_csv(file.path(project_dir, 'data', 's27_regional_flow_matrix_from_reference.csv'), show_col_types = FALSE, progress = FALSE) |>
  mutate(source_region = factor(source_region, levels = source_levels), sink_region = factor(sink_region, levels = sink_levels)) |>
  complete(source_region = source_levels, sink_region = sink_levels, fill = list(n = 0)) |>
  mutate(source_region = factor(source_region, levels = source_levels), sink_region = factor(sink_region, levels = sink_levels))
p_matrix <- ggplot(flow, aes(sink_region, source_region)) +
  geom_tile(fill = 'white', color = '#D8D8D8', linewidth = 0.35) +
  geom_point(data = filter(flow, n > 0), aes(size = n, fill = n), shape = 21, color = '#4D4D4D', stroke = 0.25, alpha = 0.98) +
  geom_text(data = filter(flow, n > 0), aes(label = as.character(n)), size = 2.5, color = '#171717') +
  scale_size_area(max_size = 12, guide = 'none') +
  scale_fill_gradient(low = '#F3F4C8', high = '#142D78', guide = 'none') +
  scale_x_discrete(drop = FALSE) +
  scale_y_discrete(drop = FALSE) +
  labs(title = 'C   Source-to-sink regional transmission matrix', x = 'Sink region', y = 'Source region') +
  coord_equal() +
  theme_minimal(base_size = 8, base_family = 'Arial') +
  theme(
    plot.title = element_text(face = 'bold', size = 8.5),
    axis.title = element_text(face = 'bold', size = 9),
    axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1, size = 6.5, color = unname(region_colors[sink_levels])),
    axis.text.y = element_text(size = 6.5, color = unname(region_colors[source_levels])),
    panel.grid = element_blank(),
    panel.border = element_rect(fill = NA, color = '#666666', linewidth = 0.4),
    plot.margin = margin(3, 7, 3, 3)
  )
lower_panels <- p_time | p_matrix
lower_panels <- lower_panels + plot_layout(widths = c(1.45, 1))
combined <- p_map / lower_panels + plot_layout(heights = c(1, 1))
pdf_path <- file.path(output_dir, 'Figure_S27_epi_evidence_combined_plus_jhu_cdc_v2_core.pdf')
png_path <- file.path(output_dir, 'Figure_S27_epi_evidence_combined_plus_jhu_cdc_v2_core.png')
ggsave(pdf_path, combined, width = 11, height = 9.35, device = cairo_pdf, bg = 'white')
ggsave(png_path, combined, width = 11, height = 9.35, dpi = 300, bg = 'white')
message('Events plotted: ', nrow(plot_data))
message('Routes plotted: ', nrow(routes))
message('Matrix events plotted: ', sum(flow$n))
message('PDF: ', pdf_path)














args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
output_dir <- args[2]

library(ggplot2)

data <- read.csv(csv_path)

# Basic checks
required_cols <- c("longitude", "latitude", "timestamp", "animal_id")
if (!all(required_cols %in% names(data))) {
  stop("CSV must include columns: longitude, latitude, timestamp, animal_id")
}

# Convert timestamp
if (!inherits(data$timestamp, "POSIXct")) {
  data$timestamp <- as.POSIXct(data$timestamp)
}

# Plot
p <- ggplot(data, aes(x = longitude, y = latitude, color = animal_id)) +
  geom_path() +
  geom_point(size = 1) +
  theme_minimal() +
  ggtitle("Movement Tracks by Animal")

ggsave(file.path(output_dir, "track_plot.png"), plot = p, width = 7, height = 5)

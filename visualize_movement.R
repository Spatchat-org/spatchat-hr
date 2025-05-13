# visualize_movement.R
args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
output_dir <- args[2]

cat("Reading data from:", csv_path, "\n")

# Load libraries
if (!require("ggplot2")) install.packages("ggplot2", repos = "https://cloud.r-project.org")
library(ggplot2)

# Read CSV
data <- tryCatch({
  read.csv(csv_path)
}, error = function(e) {
  cat("Error reading CSV:", e$message, "\n")
  quit(status = 1)
})

# Basic column check
required_cols <- c("longitude", "latitude", "timestamp", "animal_id")
if (!all(required_cols %in% names(data))) {
  cat("Missing required columns.\n")
  quit(status = 1)
}

# Convert timestamp if needed
if (!inherits(data$timestamp, "POSIXct")) {
  data$timestamp <- as.POSIXct(data$timestamp)
}

# Plot generation
cat("Generating movement track plot...\n")
p <- ggplot(data, aes(x = longitude, y = latitude, color = animal_id)) +
  geom_path() +
  geom_point(size = 1) +
  theme_minimal() +
  ggtitle("Movement Tracks by Animal")

output_file <- file.path(output_dir, "track_plot.png")
tryCatch({
  ggsave(output_file, plot = p, width = 7, height = 5)
  cat("Plot saved to", output_file, "\n")
}, error = function(e) {
  cat("Failed to save plot:", e$message, "\n")
  quit(status = 1)
})

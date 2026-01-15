# Install RavenR package if not already installed
if (!require("RavenR")) {
  install.packages("RavenR")
}
install.packages("languageserver")

# Load the RavenR library
library(RavenR)

# Define the base directory for RVI files
rvi_dir <- "/home/jberg/OneDrive/Raven_worldwide/03_model_setups_coupled/model_structure_plots/rvi_files"

# Define the model types
model_types <- c("HBV", "HMETS", "HYMOD", "MOHYSE", "UBCWM")

# Define the gauge ID (used in the RVI filename)
gauge_id <- "0102"

# Create output directory for plots
output_dir <- file.path(dirname(rvi_dir), "plots")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Loop through each model type
for (model_type in model_types) {
  
  cat(paste0("\nProcessing ", model_type, "...\n"))
  
  # Construct the path to the RVI file
  rvi_file <- file.path(rvi_dir, paste0(gauge_id, "_", model_type, ".rvi"))
  
  # Check if the RVI file exists
  if (file.exists(rvi_file)) {
    
    cat(paste0("  Found RVI file: ", rvi_file, "\n"))
    
    # Read the RVI file
    tryCatch({
      rvi <- rvn_rvi_read(rvi_file)
      
      # Define output PDF path
      pdf_output <- file.path(output_dir, paste0(gauge_id, "_", model_type, "_structure.pdf"))
      
      # Plot the model structure and save to PDF
      rvn_rvi_connections(rvi) %>% 
        rvn_rvi_process_ggplot(pdfout = pdf_output)
      
      cat(paste0("  ✓ Successfully created plot: ", pdf_output, "\n"))
      
    }, error = function(e) {
      cat(paste0("  ✗ Error processing ", model_type, ": ", e$message, "\n"))
    })
    
  } else {
    cat(paste0("  ✗ RVI file not found: ", rvi_file, "\n"))
  }
}

cat("\n========================================\n")
cat("Model structure plotting complete!\n")
cat(paste0("Plots saved to: ", output_dir, "\n"))
cat("========================================\n")
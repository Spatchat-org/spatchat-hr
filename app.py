import shutil
import os
import subprocess

def handle_upload(file):
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)

    # Use full path to Rscript
    subprocess.run(["/usr/bin/Rscript", "visualize_movement.R", filename, "outputs"], check=True)

    return os.path.join("outputs", "track_plot.png")

import gradio as gr
import pandas as pd
import subprocess
import os
import shutil

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def handle_upload(file):
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)

    # Dynamically locate Rscript
    try:
        rscript_path = subprocess.check_output(["which", "Rscript"]).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return "Rscript not found. Is R installed correctly?"

    # Call the R script
    subprocess.run([rscript_path, "visualize_movement.R", filename, "outputs"], check=True)

    return os.path.join("outputs", "track_plot.png")

with gr.Blocks() as demo:
    gr.Markdown("## SpatChat: Movement Data Visualizer")
    with gr.Row():
        file_input = gr.File(label="Upload Movement CSV")
    output_image = gr.Image(label="Movement Plot")

    file_input.change(fn=handle_upload, inputs=file_input, outputs=output_image)

demo.launch()

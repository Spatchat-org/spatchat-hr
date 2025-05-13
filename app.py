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
    filename = os.path.join(UPLOAD_DIR, os.path.basename(file))
    shutil.copy(file, filename)

    # Use absolute path to Rscript
    subprocess.run(["/usr/bin/Rscript", "visualize_movement.R", filename, OUTPUT_DIR], check=True)

    return os.path.join(OUTPUT_DIR, "track_plot.png")

with gr.Blocks() as demo:
    gr.Markdown("## SpatChat: Movement Data Visualizer")
    with gr.Row():
        file_input = gr.File(label="Upload Movement CSV")
    output_image = gr.Image(label="Movement Plot")

    file_input.change(fn=handle_upload, inputs=file_input, outputs=output_image)

demo.launch()

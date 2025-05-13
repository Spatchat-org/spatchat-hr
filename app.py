import gradio as gr
import pandas as pd
import subprocess
import os

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def handle_upload(file):
    filename = os.path.join(UPLOAD_DIR, file.name)
    file.save(filename)

    # Call R script to generate visualization
    subprocess.run(["Rscript", "visualize_movement.R", filename, OUTPUT_DIR], check=True)

    # Output is saved as PNG
    return os.path.join(OUTPUT_DIR, "track_plot.png")

with gr.Blocks() as demo:
    gr.Markdown("## SpatChat: Movement Data Visualizer")
    with gr.Row():
        file_input = gr.File(label="Upload Movement CSV")
        submit_btn = gr.Button("Visualize")
    output_image = gr.Image(label="Movement Plot")

    submit_btn.click(fn=handle_upload, inputs=file_input, outputs=output_image)

demo.launch()

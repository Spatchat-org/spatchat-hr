import gradio as gr
import pandas as pd
import subprocess
import os
import shutil

def handle_upload(file):
    # Ensure folders exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Copy uploaded file to working dir
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)

    # Find Rscript path
    try:
        rscript_path = subprocess.check_output(["which", "Rscript"]).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return "Rscript not found. Is R installed correctly?"

    # Run R script
    try:
        subprocess.run([rscript_path, "visualize_movement.R", filename, "outputs"], check=True)
    except subprocess.CalledProcessError as e:
        return f"R script failed: {e}"

    # Verify image was created
    result_path = os.path.join("outputs", "track_plot.png")
    if not os.path.exists(result_path):
        return "R script did not generate an output image. Check logs and R package installation."

    return result_path

with gr.Blocks() as demo:
    gr.Markdown("## SpatChat: Movement Data Visualizer")

    with gr.Row():
        file_input = gr.File(label="Upload Movement CSV")
    # Keep the map preview always visible with fixed height
    output_image = gr.Image(label="Movement Plot", value=None, height=400, show_label=True)

    file_input.change(fn=handle_upload, inputs=file_input, outputs=output_image)

demo.launch()

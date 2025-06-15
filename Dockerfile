# Use official lightweight Python base image
FROM python:3.10-slim

# Install system dependencies and R
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    r-cran-sf \
    r-cran-move \
    r-cran-dplyr \
    r-cran-data.table \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libudunits2-dev \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    rsync \
    libgl1-mesa-glx \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install R CRAN packages that are NOT in apt (add more as needed)
RUN Rscript -e "install.packages(c('adehabitatHR','move','sf','raster','dplyr'), dependencies=TRUE, repos='https://cloud.r-project.org')"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code (everything)
COPY . .

# Expose port (usually 7860 for Gradio, but not strictly needed)
EXPOSE 7860

# Start your Gradio app
CMD ["python", "app.py"]

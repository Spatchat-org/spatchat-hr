FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    r-base \
    r-base-core \
    r-base-dev \
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

RUN if [ ! -f /usr/bin/Rscript ]; then ln -s /usr/lib/R/bin/Rscript /usr/bin/Rscript; fi
RUN which Rscript && Rscript --version

# Install just sf for now
RUN Rscript -e "install.packages('sf', dependencies=TRUE, repos='https://cloud.r-project.org')"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]

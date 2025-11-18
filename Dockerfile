# Minimal base image
FROM python:3.11-slim

# Saner Python/pip defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install OS deps (make, gcc, etc.).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        make \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Sources after deps.
COPY requirements.txt .
COPY Makefile .
COPY notebook_pca_anls.py .
COPY notebook_pipeline_anls.py .
COPY data/ ./data
COPY src/ ./src

# Virtual env, data, pretrains
RUN make venv VERBOSE=1
RUN make process-data
RUN make pca-params

# Run bash
CMD ["/bin/bash"]
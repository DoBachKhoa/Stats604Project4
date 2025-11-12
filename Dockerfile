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
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Sources after deps.
COPY requirements.txt .
COPY Makefile .
COPY src/ ./src

# Virtual env
RUN make venv

# Run bash
CMD ["/bin/bash"]
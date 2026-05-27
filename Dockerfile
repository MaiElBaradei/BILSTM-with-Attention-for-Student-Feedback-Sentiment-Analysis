ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps (minimal — curl for healthchecks, git for wandb artifact logging)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source so this layer is cached
COPY pyproject.toml setup.py ./
COPY bilstm_attention/__init__.py bilstm_attention/__init__.py
RUN pip install --no-cache-dir -e ".[viz]"

# Copy the rest of the source
COPY bilstm_attention/ bilstm_attention/
COPY train.py .

# Runtime directories — actual data is mounted via volumes at run time
RUN mkdir -p data checkpoints visualizations embeddings

ENTRYPOINT ["python", "train.py"]

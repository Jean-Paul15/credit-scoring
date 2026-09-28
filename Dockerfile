FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN python -m pip install --upgrade pip && \
    pip install .[dev]

COPY . /app

EXPOSE 8000 5000 9090 3000

CMD ["bash"]

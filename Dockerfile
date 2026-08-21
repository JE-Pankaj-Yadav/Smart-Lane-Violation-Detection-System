FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=120

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-linux-cpu.txt ./

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-linux-cpu.txt

RUN python -m pip install --no-cache-dir \
    -r requirements.txt

COPY . .

RUN python verify_environment.py && \
    mkdir -p uploads outputs temp logs frames static/reports

EXPOSE 5000

CMD ["python", "app.py"]
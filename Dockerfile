# GPU deployment of the speech-to-speech realtime pipeline (HF Spaces or Inference
# Endpoints): Parakeet STT -> Gemma 4 on Cerebras via the HF router -> native
# Keenable web search -> Qwen3 TTS (qwentts.cpp, CUDA). Traffic port 7860; the
# realtime websocket lives at /v1/realtime, health at /v1/pool.
# NOTE: this file intentionally differs from the repo's own Dockerfile (which has
# no CMD) — keep it when syncing from upstream.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/tmp/hf \
    TORCH_HOME=/tmp/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git libportaudio2 libsndfile1 python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --break-system-packages uv

WORKDIR /app

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
RUN uv sync --python /usr/bin/python3 --no-install-project --no-dev

COPY . .
RUN uv sync --python /usr/bin/python3 --no-dev
RUN chmod -R a+rX /app

EXPOSE 7860

# Shell form so the HF_TOKEN / KEENABLE_API_KEY secrets expand at runtime.
CMD speech-to-speech \
    --mode realtime \
    --stt parakeet-tdt \
    --llm_backend chat-completions \
    --tts qwen3 \
    --model_name "google/gemma-4-31B-it:cerebras" \
    --responses_api_base_url "https://router.huggingface.co/v1" \
    --responses_api_api_key "$HF_TOKEN" \
    --responses_api_reasoning_effort none \
    --responses_api_stream \
    --keenable_web_search \
    --ws_host 0.0.0.0 \
    --ws_port 7860

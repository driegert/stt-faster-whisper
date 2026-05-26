# faster-whisper STT Service on lilripper

Lightweight STT service using faster-whisper + Silero VAD on NVIDIA GPUs.
Exposes two HTTP endpoints from the same model:

- `POST /api/transcribe` — voicenotes-style (raw request body). Used by Octavius.
- `POST /v1/audio/transcriptions` — OpenAI-compatible (multipart upload). For
  generic OpenAI STT clients.

## Why faster-whisper?

- **~4x faster** than OpenAI Whisper (CTranslate2 backend on NVIDIA CUDA)
- **~1.5 GB VRAM** for large-v3 with int8 quantization (vs ~10 GB for standard Whisper)
- **Built-in Silero VAD** — automatically trims silence, segments speech
- **Same model quality** — uses the same large-v3 weights, just runs them faster

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for all Python management —
do not use `pip`/`venv` directly.

### 1. Get the project

```bash
cd ~/git_repos/stt-faster-whisper   # the repo lives here on lilripper
```

### 2. Install dependencies

```bash
uv sync
```

This creates `.venv/` and installs everything pinned in `uv.lock`
(faster-whisper, fastapi, uvicorn, python-multipart, numpy, and the
`nvidia-cublas` / `nvidia-cudnn` CUDA libraries).

### 3. Pre-download the model (optional — first run does this automatically)

```bash
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='int8_float16')"
```

This downloads ~3 GB and caches it in `~/.cache/huggingface/`.

### 4. The service script

The service is `stt_server.py` in this repo (no need to copy it elsewhere —
the systemd unit runs it in place). It defines:

- `transcribe_audio(path, language=...)` / `transcribe_pcm(bytes)` — the shared
  faster-whisper calls with Silero VAD.
- `POST /api/transcribe` — branches on `Content-Type`:
  - `application/json` → `{"audio": "<base64 float32 PCM>"}`
  - `application/octet-stream` → raw float32 PCM bytes (16 kHz)
  - anything else (`audio/wav`, `audio/webm`, `audio/mp3`, `audio/ogg`) → the
    audio file bytes sent **directly as the body**
  - returns `{"text": "..."}`
- `POST /v1/audio/transcriptions` — OpenAI-compatible. `multipart/form-data`
  with a `file` part plus the usual `model` / `language` / `response_format`
  fields. `model` is accepted but ignored (the server always uses `STT_MODEL`).
  Returns `{"text": "..."}` for the default `json` format, or a plain-text body
  for `response_format=text`.
- `GET /health` — returns model/device/compute-type.

All config is via environment variables (see [Configuration](#configuration-options)).
The server binds `STT_HOST:STT_PORT`, defaulting to `127.0.0.1:8502` — it listens
on localhost only and is fronted by Caddy (see [Networking](#networking)).

### 5. Run it

```bash
uv run python stt_server.py

# Health:
curl http://localhost:8502/health

# voicenotes-style (raw body):
curl -X POST -H "Content-Type: audio/wav" --data-binary @test.wav \
  http://localhost:8502/api/transcribe

# OpenAI-compatible (multipart upload):
curl -X POST http://localhost:8502/v1/audio/transcriptions \
  -F "file=@test.wav" -F "model=whisper-1"
```

### 6. systemd service

The deployed unit at `~/.config/systemd/user/stt-faster-whisper.service`:

```ini
[Unit]
Description=faster-whisper STT Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/dave/git_repos/stt-faster-whisper
ExecStart=/home/dave/.local/bin/uv run python stt_server.py
Restart=always
RestartSec=5
Environment=CUDA_VISIBLE_DEVICES=4
Environment=LD_LIBRARY_PATH=/home/dave/git_repos/stt-faster-whisper/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/dave/git_repos/stt-faster-whisper/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
Environment=STT_MODEL=large-v3
Environment=STT_DEVICE=cuda
Environment=STT_COMPUTE_TYPE=int8_float16
Environment=STT_LANGUAGE=en
Environment=STT_HOST=127.0.0.1
Environment=STT_PORT=8502

[Install]
WantedBy=default.target
```

Notes:
- `CUDA_VISIBLE_DEVICES=4` pins the service to one 3090.
- `LD_LIBRARY_PATH` points at the cuBLAS/cuDNN libs that `uv` installed into the
  venv — CTranslate2 needs these on the loader path to use CUDA.

```bash
systemctl --user daemon-reload
systemctl --user enable stt-faster-whisper
systemctl --user start stt-faster-whisper

# After editing stt_server.py, pick up changes with:
systemctl --user restart stt-faster-whisper
```

## Networking

The service binds `127.0.0.1:8502` and is exposed externally by **Caddy** on
port **8552** as a plain pass-through (no path or body rewriting):

```caddy
:8552 {
        reverse_proxy 127.0.0.1:8502 {
                flush_interval -1
                request_buffers 0
                response_buffers 0
                header_up Host 127.0.0.1:8502
        }
}
```

So external clients use `http://lilripper:8552/...` and Caddy forwards to
`127.0.0.1:8502`. Both endpoint paths pass straight through unchanged.

## Pointing clients at it

**Octavius** (voicenotes-style, on lilbuddy) — set the env var (or `.env`):

```bash
OCTAVIUS_STT_URL=http://lilripper:8552/api/transcribe
```

**OpenAI-compatible clients** — configure as an OpenAI STT provider:

- Base URL: `http://lilripper:8552/v1`
- Endpoint: `POST /v1/audio/transcriptions` (multipart `file` upload)
- Model: any value (e.g. `whisper-1`) — accepted but ignored
- API key: none required (send a dummy if the client insists)
- Response: `{ "text": "..." }`

## Expected performance

| Metric | Previous (lilbuddy ROCm) | faster-whisper (lilripper CUDA) |
|--------|------------------------|-------------------------------|
| Model | Whisper large-v3 | Whisper large-v3 (same quality) |
| 5s audio | ~2s | ~0.5s |
| 15s audio | ~5-7s | ~1-2s |
| 30s audio | ~10-15s | ~3-4s |
| VRAM | ~10 GB (PyTorch) | ~1.5 GB (int8 CTranslate2) |
| VAD | None (manual silence detect) | Silero VAD built-in |

## Configuration options

All configurable via environment variables:

- `STT_MODEL`: `large-v3` (best quality), `distil-large-v3` (faster, slightly lower quality), `medium` (faster still)
- `STT_DEVICE`: `cuda` (default), `cuda:0` etc. (or pin a GPU with `CUDA_VISIBLE_DEVICES`)
- `STT_COMPUTE_TYPE`: `int8_float16` (recommended), `float16`, `int8`
- `STT_LANGUAGE`: `en` (skip language detection for speed), or omit for auto-detect
- `STT_HOST`: default `127.0.0.1` (localhost only; Caddy fronts it)
- `STT_PORT`: default `8502`

## Notes

- First request after startup takes ~1-2s extra (CUDA warmup). Subsequent requests are fast.
- With `vad_filter=True`, Silero VAD automatically skips silence segments, so trailing silence in recordings won't produce phantom text.
- The `beam_size=3` matches what voicenotes uses. Increase to 5 for slightly better accuracy at the cost of speed.
- `/v1/audio/transcriptions` honors the `language` form field if a client sends
  one, otherwise falls back to `STT_LANGUAGE`.

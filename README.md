# stt-faster-whisper

A small, self-hosted speech-to-text HTTP service. One Python file wraps
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper weights run
through CTranslate2) with Silero voice-activity detection, loads a model once,
and serves it over two endpoints:

| Endpoint | For |
|---|---|
| `POST /v1/audio/transcriptions` | Anything that speaks the OpenAI audio API. Point the tool's base URL at this server instead of a paid API. |
| `POST /api/transcribe` | Custom clients that just want to POST audio bytes and get text back. Optional segment or word timestamps. |
| `GET /health` | Reports the model, device and compute type actually loaded. |

It is meant to run on one machine you own, on a private network, as a systemd
service. There is no auth, no queue and no multi-tenancy by design; put a
reverse proxy in front of it if you need any of that.

## Why this and not `openai-whisper`

Same `large-v3` weights, same output quality, but CTranslate2 runs them several
times faster in roughly 1.8 GB of VRAM instead of about 10 GB. On a single RTX
3090 a 15-second clip round-trips in under half a second. Without a GPU,
`large-v3` on four threads of a Threadripper PRO 5955WX transcribes a 53-second
dictation in 24 seconds, and `large-v3-turbo` in 12. Silence is trimmed by VAD
before the model sees it, so cost tracks speech, not file length.

## Two ways to run it

**With an NVIDIA GPU** (the fast path). `large-v3` at `int8_float16` is the
default configuration and needs about 2 GB of VRAM and a CUDA 12 capable
driver. The CUDA libraries are installed by `uv sync` automatically.

**CPU only.** The same code runs without a GPU, using a smaller
model. Install with `uv sync --no-group cuda` to skip the CUDA libraries, set
the device to `cpu` and the compute type to `int8`, and pick a model sized for
your CPU. `small` is close to realtime on a modern laptop; `tiny` and `base`
are faster and rougher; `large-v3` is too slow to be useful interactively.

Both paths are covered step by step in the setup guide.

## Getting started

Everything you need is in **[SETUP.md](SETUP.md)**: requirements, installing
with [uv](https://docs.astral.sh/uv/), the CUDA loader-path step that GPU
installs cannot skip, running in the foreground, the full configuration table,
the CPU-only recipe, a generated systemd unit, exposing the service through a
reverse proxy, client examples for both endpoints, measured performance and
troubleshooting.

The short version:

```bash
git clone <this repo> && cd stt-faster-whisper
uv sync                      # GPU. CPU-only: uv sync --no-group cuda
uv run python stt_server.py  # then read SETUP.md for the loader-path step
```

## Repository layout

| File | What it is |
|---|---|
| `stt_server.py` | The whole service. FastAPI app, model loading, both endpoints. |
| `pyproject.toml`, `uv.lock` | Dependencies. The `cuda` dependency group holds the NVIDIA libraries and is on by default. |
| `SETUP.md` | The setup and operations guide. Start here. |
| `stt-lilripper-setup.md` | The author's notes for one specific deployment. Kept as a worked example; the paths and ports in it are not yours. |

## Configuration at a glance

Everything is an environment variable with a sensible default. The ones you are
most likely to touch are `STT_MODEL`, `STT_DEVICE` and `STT_COMPUTE_TYPE`.
The service binds to localhost only unless you set `STT_HOST`. The full table,
with the model ladder from `tiny` to `large-v3`, is in
[SETUP.md](SETUP.md#7-configuration).

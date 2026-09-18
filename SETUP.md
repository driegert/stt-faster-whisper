# faster-whisper STT Service — Setup Guide

A small, self-hosted speech-to-text HTTP service built on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) with
Silero VAD. One loaded model serves two endpoints:

| Endpoint | Style | Body |
|---|---|---|
| `POST /v1/audio/transcriptions` | OpenAI-compatible | `multipart/form-data` with a `file` part |
| `POST /api/transcribe` | Raw-body | The audio bytes sent directly as the request body |
| `GET /health` | Status | — |

The OpenAI-compatible endpoint lets you point any tool that speaks the OpenAI
audio API at your own GPU instead of a paid API. The raw-body endpoint is a
simpler alternative for custom clients that just want to POST bytes and get text
back.

**Why faster-whisper rather than `openai-whisper`:** it runs the same
`large-v3` weights through CTranslate2, which is several times faster and uses
far less VRAM (roughly 1.5–2 GB with int8 quantization, versus ~10 GB for the
PyTorch implementation), and it ships Silero VAD so silence is trimmed
automatically.

---

## 1. Requirements

**Hardware / OS**

- Linux (this guide uses systemd for the service; everything else is portable).
- An NVIDIA GPU for the fast path. CTranslate2 4.x requires **CUDA 12** and
  **cuDNN 9**, which in practice means an NVIDIA driver of **525 or newer**
  (newer is better; 12.8-era features want a recent driver). Check with
  `nvidia-smi` — the "CUDA Version" it reports is the *maximum* your driver
  supports, and it needs to say 12.x or higher.
- ~2 GB of free VRAM for `large-v3` at `int8_float16`, plus ~3 GB of disk for
  the downloaded model weights.
- No GPU? It still runs on CPU with a smaller model — see
  [CPU-only](#cpu-only-no-nvidia-gpu). `small` needs ~1 GB of RAM and is
  roughly realtime on a modern laptop; `large-v3` is not usable interactively
  on CPU.

**Software**

- [uv](https://docs.astral.sh/uv/) for all Python management. This project uses
  uv exclusively — do not use `pip` or `venv` directly.
- Python 3.12+ (uv will download it for you if it isn't installed).
- You do **not** need a system `ffmpeg` install. Audio decoding is handled by
  PyAV, which bundles its own FFmpeg libraries. (`ffmpeg` is handy for *creating*
  a test clip in step 6, but the service doesn't need it.)

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

That puts the binary at `~/.local/bin/uv`. Remember the full path — the systemd
unit in step 8 needs it, because systemd does not read your shell profile.

---

## 2. Get the code

**If you were given the repository**, just enter it:

```bash
cd /path/to/stt-faster-whisper
```

**If you were given only this document**, create the project from scratch:

```bash
mkdir stt-faster-whisper && cd stt-faster-whisper
uv init --python 3.12 --no-workspace
rm -f main.py                         # uv's placeholder; keep README.md,
                                      # pyproject.toml references it
uv add \
  "faster-whisper>=1.2.1" \
  "fastapi>=0.135.3" \
  "uvicorn>=0.44.0" \
  "python-multipart>=0.0.27" \
  "numpy>=2.4.4"
uv add --group cuda \
  "nvidia-cublas-cu12>=12.9.2.10" \
  "nvidia-cudnn-cu12>=9.21.0.82"
```

The CUDA libraries go in a `cuda` dependency group rather than the main
dependency list so CPU-only installs can leave them out. Make the group install
by default by adding this to `pyproject.toml` (the full file is in
[Appendix B](#appendix-b--pyprojecttoml)):

```toml
[tool.uv]
default-groups = ["cuda"]
```

Then save the server source from [Appendix A](#appendix-a--stt_serverpy) as
`stt_server.py` in that directory. (Going CPU-only? Skip the `uv add --group
cuda` command and the `[tool.uv]` block entirely.)

> **`python-multipart` must be 0.0.27 or newer.** Earlier versions carry
> [CVE-2026-42561](https://github.com/Kludex/python-multipart/security/advisories/GHSA-pp6c-gr5w-3c5g)
> (CVSS 7.5), an unbounded-multipart-header denial of service. This service
> exposes a multipart upload endpoint, so it is directly in the blast radius.

> **Do not substitute the `cu13` packages.** `nvidia-cublas-cu13` /
> `nvidia-cudnn-cu13` exist on PyPI, but CTranslate2 4.x is built against
> CUDA 12. Stay on `cu12`.

---

## 3. Install dependencies

If you were given the repository (it contains a `uv.lock`):

```bash
uv sync
```

This creates `.venv/` and installs the exact versions pinned in `uv.lock`
(faster-whisper, ctranslate2, fastapi, uvicorn, python-multipart, numpy, and the
cuBLAS/cuDNN CUDA libraries).

**CPU-only:** the CUDA libraries live in a `cuda` dependency group that is on
by default. Leave it out — it is several hundred megabytes you cannot use:

```bash
uv sync --no-group cuda
```

Re-running plain `uv sync` later will add the CUDA libraries back, so keep the
flag in whatever script or alias you use to update the install.

If you bootstrapped from scratch in step 2, `uv add` already did this — there is
nothing more to do here.

To deliberately move an existing lockfile to the newest compatible releases:

```bash
uv lock --upgrade && uv sync
```

Do this on purpose, not by reflex — a pinned, known-good lockfile is the point.

### Verified version set

Every request path of this service was tested end to end — on CPU, and on an
NVIDIA GPU with `large-v3` at `int8_float16` — against the stack below, which is
what a fresh install resolves to as of August 2026:

```
faster-whisper 1.2.1    ctranslate2 4.8.1      nvidia-cudnn-cu12 9.24.0.43
fastapi 0.141.1         starlette 1.3.1        nvidia-cublas-cu12 12.9.2.10
uvicorn 0.52.1          numpy 2.5.1            python-multipart 0.0.32
```

Two recent upstream changes are worth knowing about even though neither breaks
this service: FastAPI 0.137+ enables strict JSON `Content-Type` checking by
default, and Starlette has moved to 1.x. This service is unaffected because its
endpoints read the request body directly rather than through request-model
validation — but keep both in mind if you extend the API.

---

## 4. Put the CUDA libraries on the loader path (required)

This is the step people miss. The cuBLAS and cuDNN shared libraries are
installed *inside the virtualenv*, where the dynamic linker will not find them
on its own. CTranslate2 loads them by name at model-load time, so without this
you get a crash the first time you transcribe:

```
Unable to load libcudnn_ops.so.9. Error: libcudnn_ops.so.9: cannot open shared object file
```

Export `LD_LIBRARY_PATH` derived from the venv, so it stays correct regardless
of your username, install location, or Python version:

```bash
SITE_PACKAGES=$(uv run python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

Verify it before going further:

```bash
uv run python -c "
import ctypes
for lib in ('libcudnn_ops.so.9', 'libcublas.so.12'):
    ctypes.CDLL(lib); print('OK', lib)
"
```

Both lines must print `OK`. This export lasts only for the current shell — the
systemd unit in step 8 sets the same variable permanently, which is how the
deployed service gets it. CPU-only installs can skip this entire step.

---

## 5. Pre-download the model (optional)

The first run downloads the weights automatically, but doing it up front keeps
the first request from timing out:

```bash
uv run python -c "
from faster_whisper import WhisperModel
WhisperModel('large-v3', device='cuda', compute_type='int8_float16')
"
```

This pulls ~3 GB into `~/.cache/huggingface/`. Set `HF_HOME` beforehand to cache
it somewhere else. For an air-gapped machine, copy that cache directory across
and set `HF_HUB_OFFLINE=1`.

---

## 6. Run it in the foreground

```bash
uv run python stt_server.py
```

The model loads at import time, so startup takes ten to thirty seconds before
the port accepts connections. Watch for `Model loaded.` in the log.

Get a test clip — any speech file works; to record five seconds from your
default microphone:

```bash
arecord -f cd -d 5 test.wav                # ALSA
ffmpeg -f pulse -i default -t 5 test.wav   # PulseAudio / PipeWire
```

Then, in a second terminal:

```bash
# Health — reports the model, device and compute type actually in use
curl http://localhost:8502/health

# OpenAI-compatible (multipart upload)
curl -X POST http://localhost:8502/v1/audio/transcriptions \
  -F "file=@test.wav" -F "model=whisper-1"

# Raw-body
curl -X POST -H "Content-Type: audio/wav" --data-binary @test.wav \
  http://localhost:8502/api/transcribe
```

Both transcription calls return `{"text": "..."}`.

---

## 7. Configuration

Everything is configured with environment variables. Defaults are in
parentheses.

| Variable | Default | Notes |
|---|---|---|
| `STT_MODEL` | `large-v3` | See the model table below. |
| `STT_DEVICE` | `cuda` | `cuda`, `cuda:0`, or `cpu`. |
| `STT_COMPUTE_TYPE` | `int8_float16` | `int8_float16`, `float16`, `int8`, `float32`. |
| `STT_LANGUAGE` | `en` | An ISO code skips language detection and is faster. Set to empty for auto-detect. |
| `STT_HOST` | `127.0.0.1` | Localhost only by default — deliberately. See [Exposing it](#9-exposing-it-on-a-network). |
| `STT_PORT` | `8502` | |
| `CUDA_VISIBLE_DEVICES` | unset | Standard NVIDIA variable; set to e.g. `0` to pin the service to one GPU on a multi-GPU box. |

Useful `STT_MODEL` values, fastest to most accurate:

| Model | Notes |
|---|---|
| `tiny`, `base` | The CPU-oriented sizes. `base` is a reasonable rough draft; `tiny` is fast but error-prone. |
| `small`, `medium` | Lower quality than the large models, low VRAM. `small` is the usual CPU choice. |
| `distil-large-v3`, `distil-large-v3.5` | Distilled; noticeably faster than `large-v3`, slightly lower quality. English-focused. |
| `large-v3-turbo` (alias `turbo`) | Much faster than `large-v3` with a small accuracy cost. A good default if `large-v3` is too slow. |
| `large-v3` | Best quality. What this guide defaults to. |

Compute-type guidance: `int8_float16` needs a GPU with fp16 support (compute
capability 7.0+, i.e. Turing or newer). On older GPUs use `int8`. Use `float16`
if you want maximum fidelity and have the VRAM.

### CPU-only (no NVIDIA GPU)

Install without the CUDA libraries, skip step 4, and pick a smaller model:

```bash
uv sync --no-group cuda
STT_DEVICE=cpu STT_COMPUTE_TYPE=int8 STT_MODEL=small uv run python stt_server.py
```

Two settings differ from the GPU defaults and both are required:

- **`STT_COMPUTE_TYPE` must be `int8`** (or `float32`). The default
  `int8_float16` is a GPU compute type; on CPU the model refuses to load with
  `Requested int8_float16 compute type, but the target device or backend do not
  support efficient int8_float16 computation`. `int8` is also the fastest CPU
  option.
- **`STT_MODEL` should be a small model.** Rough guidance for an English-only
  workload on a modern desktop or laptop CPU:

  | Model | RAM | When to use it |
  |---|---|---|
  | `tiny` | ~0.5 GB | Very fast, noticeably error-prone. Rough drafts only. |
  | `base` | ~0.5 GB | Weak machines. Usable quality for clear speech. |
  | `small` | ~1 GB | The usual CPU choice: near realtime, decent accuracy. |
  | `distil-large-v3`, `large-v3-turbo` | ~2–3 GB | Much better accuracy than `small`. Viable on a fast desktop CPU because they have far fewer decoder layers than `large-v3`; try one if quality matters more than speed. |
  | `medium`, `large-v3` | 3–6 GB | Too slow on CPU to be useful interactively. |

  All sizes are quantised `int8` weights on disk; the first run downloads them.

**Use your cores.** faster-whisper runs on **4 threads by default, however
many cores you have.** Set `OMP_NUM_THREADS` to your physical core count (add
an `Environment=OMP_NUM_THREADS=8` line to the systemd unit). Going from 4 to
8 threads cut `large-v3` time by about a quarter in the measurements below;
hyperthreads add little.

**What "slow" means in practice.** Measured on 4 threads of a 16-core desktop
Zen 3 CPU (Threadripper PRO 5955WX), `int8`, beam size 3, VAD on, with a 53.5 s
clip of continuous dictation; transcription time only, model already loaded
(full details under [Performance](#cpu)):

| Model | Transcribe time | Fraction of realtime |
|---|---|---|
| `small` | 5.5 s | 0.10× |
| `large-v3-turbo` | 11.5 s | 0.22× |
| `large-v3` | 24.1 s | 0.45× |
| `large-v3`, 8 threads | 18.2 s | 0.34× |

All four produced the same transcript of that clean, synthetic clip. A
four-year-old laptop i5 is roughly half the per-core speed and throttles under
sustained load, so expect about double these times there: `large-v3` at or a
little slower than realtime, `large-v3-turbo` around half of realtime. That
makes `large-v3` fine for record-then-transcribe dictation and batch jobs, and
`large-v3-turbo` the better choice when you want the text back quickly.

Everything else — endpoints, VAD, timestamps, the systemd unit — is unchanged.

---

## 8. Run it as a service (systemd)

A **user** service is the simplest option and needs no root. Generate the unit
so the paths are correct for your machine rather than hand-editing them:

```bash
mkdir -p ~/.config/systemd/user
PROJECT_DIR=$(pwd)
UV_BIN=$(command -v uv)
SITE_PACKAGES=$(uv run python -c "import site; print(site.getsitepackages()[0])")

cat > ~/.config/systemd/user/stt-faster-whisper.service <<EOF
[Unit]
Description=faster-whisper STT Service
After=network.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
ExecStart=${UV_BIN} run python stt_server.py
Restart=always
RestartSec=5

# Required so CTranslate2 can find the CUDA libs inside the venv (see step 4).
# CPU-only: delete this line.
Environment=LD_LIBRARY_PATH=${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/cudnn/lib

# Pin to a single GPU on a multi-GPU machine; drop this line otherwise.
#Environment=CUDA_VISIBLE_DEVICES=0

# CPU-only: use STT_MODEL=small, STT_DEVICE=cpu, STT_COMPUTE_TYPE=int8 instead
# (see the CPU-only section above).
Environment=STT_MODEL=large-v3
Environment=STT_DEVICE=cuda
Environment=STT_COMPUTE_TYPE=int8_float16
Environment=STT_LANGUAGE=en
Environment=STT_HOST=127.0.0.1
Environment=STT_PORT=8502

[Install]
WantedBy=default.target
EOF
```

Enable and start it:

```bash
systemctl --user daemon-reload
systemctl --user enable --now stt-faster-whisper
systemctl --user status stt-faster-whisper
journalctl --user -u stt-faster-whisper -f
```

**Start at boot without logging in.** A user service normally stops when your
last session ends and does not start until you log in. Enable lingering:

```bash
sudo loginctl enable-linger "$USER"
```

After editing `stt_server.py`, pick up the change with:

```bash
systemctl --user restart stt-faster-whisper
```

If you'd rather use a system-wide unit, put the same file in
`/etc/systemd/system/`, add `User=` and `Group=` lines, use absolute paths
throughout, and manage it with `sudo systemctl` (no `--user`). Lingering is then
irrelevant, but the service account must be able to reach the GPU device nodes
and the model cache.

> **Don't add uvicorn `--workers`.** Each worker loads its own full copy of the
> model into VRAM. One process is the intended deployment.

---

## 9. Exposing it on a network

The service binds `127.0.0.1` by default, and **it has no authentication of any
kind**. Anyone who can reach the port can use your GPU and send it audio. Treat
the port as trusted-network-only.

Two options:

**A. Bind directly** — simplest, appropriate on a private/VPN interface:

```
STT_HOST=0.0.0.0
```

**B. Front it with a reverse proxy** — do this if you want TLS, access control,
or a hostname. Both endpoint paths pass straight through unchanged; no path or
body rewriting is needed. The one thing that matters is disabling buffering, so
large uploads stream instead of being held in memory.

Caddy:

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

nginx:

```nginx
server {
    listen 8552;
    client_max_body_size 200m;       # audio uploads
    location / {
        proxy_pass http://127.0.0.1:8502;
        proxy_request_buffering off;
        proxy_buffering off;
        proxy_read_timeout 300s;     # long clips take a while
    }
}
```

If you add authentication at the proxy, note that OpenAI clients send their key
as `Authorization: Bearer <key>` — you can check that header at the proxy and
have real API-key auth without touching the application.

Whichever you choose, restrict access at the firewall too, e.g.
`sudo ufw allow from 192.168.0.0/16 to any port 8552`.

---

## 10. Pointing clients at it

Assume the service is reachable at `http://YOUR_HOST:8552` (or
`http://localhost:8502` if you skipped the proxy).

### OpenAI-compatible clients

Configure the tool as an OpenAI STT provider:

- **Base URL:** `http://YOUR_HOST:8552/v1`
- **Endpoint:** `POST /v1/audio/transcriptions`, `multipart/form-data`
- **Model:** any string, e.g. `whisper-1`. It is accepted and **ignored** — the
  server always uses whatever `STT_MODEL` is set to.
- **API key:** none required. Send a dummy value if the client insists on one.
- **Response:** `{"text": "..."}`, or a plain-text body if you pass
  `response_format=text`.

Accepted form fields: `file` (required), `model`, `language`,
`response_format`, `prompt`, `temperature`. The last two are accepted for
compatibility but currently ignored. `language` **is** honoured per-request and
overrides `STT_LANGUAGE`.

With the official Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://YOUR_HOST:8552/v1", api_key="unused")
with open("test.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper-1", file=f).text)
```

### Raw-body clients

`POST /api/transcribe` branches on the request's `Content-Type`:

| `Content-Type` | Body |
|---|---|
| `audio/wav`, `audio/webm`, `audio/mp3`, `audio/ogg`, anything unrecognised | The audio file's bytes, sent directly as the body (not multipart, not form-encoded) |
| `application/octet-stream` | Raw **16 kHz mono float32 little-endian** PCM samples in `[-1.0, 1.0]` |
| `application/json` | `{"audio": "<base64 of that same float32 PCM>"}` |

Returns `{"text": "..."}`, or `{"error": "..."}` with HTTP 500 on failure.

Note that unlike the OpenAI endpoint, this one has no per-request language
option — it always uses `STT_LANGUAGE`.

Raw PCM example:

```bash
ffmpeg -i test.wav -f f32le -ac 1 -ar 16000 test.raw
curl -X POST -H "Content-Type: application/octet-stream" --data-binary @test.raw \
  http://YOUR_HOST:8552/api/transcribe
```

---

## 11. Performance

End-to-end HTTP round trips measured on one NVIDIA RTX 3090 with `large-v3` at
`int8_float16`, ctranslate2 4.8.1, median of three warm requests:

| Audio length | Round trip |
|---|---|
| 5 s | ~0.25 s |
| 15 s | ~0.45 s |

Transcription time tracks the amount of *speech*, not the length of the file —
VAD discards silence before the model ever sees it, and there is a floor of
roughly 0.2 s of fixed overhead per request.

One failure mode is worth knowing about: highly repetitive audio can push the
decoder into emitting long repeated output, which is disproportionately slow. A
synthetic 30 s clip made by looping the same 11 s of speech four times took
~4 s — nearly ten times the 15 s figure above. Real 30 s recordings do not
behave like that, but if you see a request take far longer than these numbers
suggest, suspect repetition in the audio rather than the GPU.

Resident VRAM is ~1.8 GB. For comparison, the PyTorch `openai-whisper`
implementation of the same `large-v3` weights uses roughly 10 GB and takes
several times longer.

The first request after startup costs an extra second or two of CUDA warmup.
If you need more speed, try `large-v3-turbo` or `distil-large-v3.5` before
reaching for a bigger GPU.

### CPU

Measured on an AMD Ryzen Threadripper PRO 5955WX (16 cores / 32 threads, Zen 3,
up to 4.5 GHz), `int8`, beam size 3, VAD on, faster-whisper 1.2.1 /
ctranslate2 4.8.1, on a 53.5 s clip of continuous synthetic dictation. The
thread column is what was actually used, set with `cpu_threads`; the rest of
the machine was idle. Times are transcription only. Model load is separate and
happens once at startup: about 1 s for `small` and 3 s for `large-v3` from a
warm local disk.

| Model | Threads | Transcribe | Fraction of realtime |
|---|---|---|---|
| `small` | 4 | 5.5 s | 0.10× |
| `large-v3-turbo` | 4 | 11.5 s | 0.22× |
| `large-v3` | 4 | 24.1 s | 0.45× |
| `large-v3` | 8 | 18.2 s | 0.34× |

All four runs produced an identical transcript. Doubling the threads from 4 to
8 bought about 25% on `large-v3`, so it is far from linear; the decoder is
memory-bound. 4 threads is the faster-whisper default whatever the core count —
see [CPU-only](#cpu-only-no-nvidia-gpu) for the `OMP_NUM_THREADS` setting and
for how to read these numbers on a laptop.

---

## 12. Troubleshooting

**`Unable to load libcudnn_ops.so.9` / `libcublas.so.12: cannot open shared
object file`**
`LD_LIBRARY_PATH` isn't set. Redo [step 4](#4-put-the-cuda-libraries-on-the-loader-path-required)
for a foreground run, or confirm the `Environment=LD_LIBRARY_PATH=` line in the
systemd unit points at paths that actually exist (they change if you recreate
the venv under a different Python version).

**`CUDA driver version is insufficient for CUDA runtime version`**
Your NVIDIA driver is older than CUDA 12 requires. Update it, or run CPU-only.

**`ValueError: Requested int8_float16 compute type, but the target device or
backend do not support efficient int8_float16 computation`**
The GPU is pre-Turing. Use `STT_COMPUTE_TYPE=int8`.

**`CUDA failed with error out of memory`**
Something else is using the GPU, or the model is too large. Pin to a free GPU
with `CUDA_VISIBLE_DEVICES`, switch to a smaller `STT_MODEL`, or confirm you
aren't running multiple copies of the service.

**`Form data requires "python-multipart" to be installed`**
The dependency is missing — run `uv sync`.

**`ValueError: Requested int8_float16 compute type, but the target device or backend do not support efficient int8_float16 computation`**
You are running on CPU (`STT_DEVICE=cpu`, or CUDA was not found) with the GPU
default compute type. Set `STT_COMPUTE_TYPE=int8` — see
[CPU-only](#cpu-only-no-nvidia-gpu). The same message with `float16` means the
same thing.

**Empty `{"text": ""}` response**
Usually correct behaviour: Silero VAD found no speech. Check that your clip
actually contains audible speech, and that `STT_LANGUAGE` matches the language
being spoken.

**`[Errno 98] Address already in use`**
Another process holds the port — often an older copy of this service. Check with
`ss -ltnp | grep 8502`.

**Service never becomes reachable, no error in the log**
It's probably still downloading the model on first run. Watch
`journalctl --user -u stt-faster-whisper -f` and wait for `Model loaded.`

---

## Appendix A — `stt_server.py`

Only needed if you're bootstrapping from this document alone. If the project
directory already contains `stt_server.py`, use that file — it is the source of
truth and this copy may lag behind it.

```python
"""Lightweight STT HTTP service using faster-whisper + Silero VAD."""

import base64
import logging
import os
import tempfile

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_SIZE = os.environ.get("STT_MODEL", "large-v3")
DEVICE = os.environ.get("STT_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "int8_float16")
LANGUAGE = os.environ.get("STT_LANGUAGE", "en")
HOST = os.environ.get("STT_HOST", "127.0.0.1")
PORT = int(os.environ.get("STT_PORT", "8502"))

log.info("Loading %s on %s (%s)...", MODEL_SIZE, DEVICE, COMPUTE_TYPE)
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
log.info("Model loaded.")

app = FastAPI(title="faster-whisper STT")

VAD_PARAMETERS = dict(
    threshold=0.5,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)


def transcribe_audio(audio_path: str, language: str = LANGUAGE) -> str:
    segments, _info = model.transcribe(
        audio_path,
        language=language,
        beam_size=3,
        vad_filter=True,
        vad_parameters=VAD_PARAMETERS,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe_pcm(pcm_bytes: bytes) -> str:
    audio = np.frombuffer(pcm_bytes, dtype=np.float32)
    segments, _info = model.transcribe(
        audio,
        language=LANGUAGE,
        beam_size=3,
        vad_filter=True,
        vad_parameters=VAD_PARAMETERS,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


@app.post("/api/transcribe")
async def api_transcribe(request: Request):
    """Transcribe audio. Accepts audio files, raw PCM, or base64 JSON."""
    content_type = request.headers.get("content-type", "")

    try:
        if "application/json" in content_type:
            body = await request.json()
            raw = base64.b64decode(body.get("audio", ""))
            text = transcribe_pcm(raw)

        elif "application/octet-stream" in content_type:
            raw = await request.body()
            text = transcribe_pcm(raw)

        else:
            raw = await request.body()
            ext = ".webm"
            if "wav" in content_type:
                ext = ".wav"
            elif "mp3" in content_type or "mpeg" in content_type:
                ext = ".mp3"
            elif "ogg" in content_type:
                ext = ".ogg"

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(raw)
                tmp_path = f.name
            try:
                text = transcribe_audio(tmp_path)
            finally:
                os.unlink(tmp_path)

        return JSONResponse({"text": text})

    except Exception as e:
        log.exception("Transcription failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/v1/audio/transcriptions")
async def openai_transcribe(
    file: UploadFile = File(...),
    model: str = Form(None),  # accepted for OpenAI compat; ignored (server uses STT_MODEL)
    language: str = Form(None),
    response_format: str = Form("json"),
    prompt: str = Form(None),  # accepted for compat; not yet used
    temperature: str = Form(None),  # accepted for compat; not yet used
):
    """OpenAI-compatible transcription endpoint."""
    ext = os.path.splitext(file.filename or "")[1] or ".webm"
    try:
        raw = await file.read()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(raw)
            tmp_path = f.name
        try:
            text = transcribe_audio(tmp_path, language=language or LANGUAGE)
        finally:
            os.unlink(tmp_path)

        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text})

    except Exception as e:
        log.exception("Transcription failed")
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=500,
        )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
```

## Appendix B — `pyproject.toml`

```toml
[project]
name = "stt-faster-whisper"
version = "0.1.0"
description = "Self-hosted faster-whisper STT service"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.135.3",
    "faster-whisper>=1.2.1",
    "numpy>=2.4.4",
    # >=0.0.27 fixes CVE-2026-42561 (unbounded multipart part headers, DoS)
    "python-multipart>=0.0.27",
    "uvicorn>=0.44.0",
]

# CUDA runtime libraries (cuBLAS + cuDNN) that CTranslate2 loads at model-load
# time. Installed by `uv sync` by default; CPU-only installs skip them with
# `uv sync --no-group cuda`. Stay on cu12: CTranslate2 4.x is built against CUDA 12.
[dependency-groups]
cuda = [
    "nvidia-cublas-cu12>=12.9.2.10",
    "nvidia-cudnn-cu12>=9.21.0.82",
]

[tool.uv]
default-groups = ["cuda"]
```

These are floors, not pins — `uv` resolves them upward and records the exact
result in `uv.lock`. See [Verified version set](#verified-version-set) for what
they resolve to today.

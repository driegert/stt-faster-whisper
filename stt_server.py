"""Lightweight STT HTTP service using faster-whisper + Silero VAD."""

import base64
import logging
import os
import tempfile

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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


def transcribe_audio(audio_path: str) -> str:
    segments, _info = model.transcribe(
        audio_path,
        language=LANGUAGE,
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
    """Transcribe audio. Accepts audio files, raw PCM, or base64 JSON.

    Compatible with the voicenotes /api/transcribe endpoint that Octavius
    already uses — same content-type handling, same response shape.
    """
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

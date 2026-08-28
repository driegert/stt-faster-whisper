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


def transcribe(source, language: str = LANGUAGE, word_timestamps: bool = False):
    """Run the model and materialise the (lazy) segment generator.

    Returns (segments, info). Segment timings are always produced by the model;
    it is the callers below that decide whether to serialise or discard them.
    """
    segments, info = model.transcribe(
        source,
        language=language,
        beam_size=3,
        vad_filter=True,
        vad_parameters=VAD_PARAMETERS,
        word_timestamps=word_timestamps,
    )
    return list(segments), info


def joined_text(segments) -> str:
    return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe_audio(audio_path: str, language: str = LANGUAGE) -> str:
    segments, _info = transcribe(audio_path, language=language)
    return joined_text(segments)


def transcribe_pcm(pcm_bytes: bytes) -> str:
    audio = np.frombuffer(pcm_bytes, dtype=np.float32)
    segments, _info = transcribe(audio)
    return joined_text(segments)


# --- timestamp serialisation -------------------------------------------------


def segment_dict(seg) -> dict:
    """OpenAI verbose_json segment shape."""
    return {
        "id": seg.id,
        "seek": seg.seek,
        "start": round(seg.start, 3),
        "end": round(seg.end, 3),
        "text": seg.text,
        "tokens": list(seg.tokens),
        "temperature": seg.temperature,
        "avg_logprob": seg.avg_logprob,
        "compression_ratio": seg.compression_ratio,
        "no_speech_prob": seg.no_speech_prob,
    }


def word_dicts(segments) -> list:
    words = []
    for seg in segments:
        for w in seg.words or ():
            words.append(
                {
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": w.probability,
                }
            )
    return words


def _clock(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def format_srt(segments) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n{_clock(seg.start, ',')} --> {_clock(seg.end, ',')}\n"
            f"{seg.text.strip()}\n"
        )
    return "\n".join(blocks)


def format_vtt(segments) -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        blocks.append(
            f"{_clock(seg.start, '.')} --> {_clock(seg.end, '.')}\n"
            f"{seg.text.strip()}\n"
        )
    return "\n".join(blocks)


# --- endpoints ---------------------------------------------------------------


@app.post("/api/transcribe")
async def api_transcribe(request: Request, timestamps: str = None):
    """Transcribe audio. Accepts audio files, raw PCM, or base64 JSON.

    Compatible with the voicenotes /api/transcribe endpoint that Octavius
    already uses — same content-type handling, same response shape.

    Timings are dropped by default. Pass ?timestamps=segment (or =1/true) to
    add a "segments" list, or ?timestamps=word to also add per-word "words".
    """
    content_type = request.headers.get("content-type", "")

    want = (timestamps or "").strip().lower()
    want_words = want == "word"
    want_segments = want_words or want in ("segment", "1", "true", "yes", "on")

    try:
        if "application/json" in content_type:
            body = await request.json()
            raw = base64.b64decode(body.get("audio", ""))
            source = np.frombuffer(raw, dtype=np.float32)
            tmp_path = None

        elif "application/octet-stream" in content_type:
            raw = await request.body()
            source = np.frombuffer(raw, dtype=np.float32)
            tmp_path = None

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
            source = tmp_path

        try:
            segments, info = transcribe(source, word_timestamps=want_words)
        finally:
            if tmp_path:
                os.unlink(tmp_path)

        payload = {"text": joined_text(segments)}
        if want_segments:
            payload["duration"] = round(info.duration, 3)
            payload["segments"] = [segment_dict(s) for s in segments]
        if want_words:
            payload["words"] = word_dicts(segments)

        return JSONResponse(payload)

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
    timestamp_granularities: list[str] = Form(None, alias="timestamp_granularities[]"),
    timestamp_granularities_alt: list[str] = Form(None, alias="timestamp_granularities"),
):
    """OpenAI-compatible transcription endpoint.

    Mirrors POST /v1/audio/transcriptions: multipart/form-data with a `file`
    part plus the usual `model`/`language`/`response_format` fields. Reuses the
    same faster-whisper model as /api/transcribe.

    response_format: json (default) | text | verbose_json | srt | vtt.
    Timings come back with verbose_json/srt/vtt; per-word timings additionally
    need timestamp_granularities[]=word, as in the OpenAI API.
    """
    ext = os.path.splitext(file.filename or "")[1] or ".webm"
    fmt = (response_format or "json").strip().lower()
    granularities = {
        g.strip().lower()
        for g in (timestamp_granularities or timestamp_granularities_alt or [])
    }
    want_words = fmt == "verbose_json" and "word" in granularities

    try:
        raw = await file.read()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(raw)
            tmp_path = f.name
        try:
            segments, info = transcribe(
                tmp_path,
                language=language or LANGUAGE,
                word_timestamps=want_words,
            )
        finally:
            os.unlink(tmp_path)

        text = joined_text(segments)

        if fmt == "text":
            return PlainTextResponse(text)
        if fmt == "srt":
            return PlainTextResponse(format_srt(segments), media_type="text/plain")
        if fmt == "vtt":
            return PlainTextResponse(format_vtt(segments), media_type="text/vtt")
        if fmt == "verbose_json":
            payload = {
                "task": "transcribe",
                "language": info.language,
                "duration": round(info.duration, 3),
                "text": text,
            }
            # OpenAI omits segments when only word granularity was asked for.
            if granularities != {"word"}:
                payload["segments"] = [segment_dict(s) for s in segments]
            if want_words:
                payload["words"] = word_dicts(segments)
            return JSONResponse(payload)

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

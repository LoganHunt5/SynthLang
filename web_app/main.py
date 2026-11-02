import threading
import uuid
import lameenc
import numpy as np
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from TTS.utils.synthesizer import Synthesizer

import asyncio
import time
from contextlib import asynccontextmanager

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
AUDIO_DIR = BASE / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def discover_models() -> dict[str, dict[str, dict]]:
    """Scan models/<language>/<model_name>/ for a checkpoint + config.json."""
    registry: dict[str, dict[str, dict]] = {}
    if not MODELS_DIR.exists():
        return registry

    for lang_dir in MODELS_DIR.iterdir():
        if not lang_dir.is_dir():
            continue
        for model_dir in lang_dir.iterdir():
            config = model_dir / "config.json"
            checkpoints = sorted(
                [
                    *model_dir.glob("*.pt"),
                    *model_dir.glob("*.pth"),
                    *model_dir.glob("*.pth.tar"),
                ]
            )
            if model_dir.is_dir() and config.exists() and checkpoints:
                registry.setdefault(lang_dir.name.lower(), {})[model_dir.name] = {
                    "model_path": str(checkpoints[0]),
                    "config_path": str(config),
                }
    return registry


# model_path -> (file mtime, Synthesizer)
_loaded: dict[str, tuple[int, Synthesizer]] = {}
_load_lock = threading.Lock()
_tts_lock = threading.Lock()

USE_CUDA = torch.cuda.is_available()


def get_synthesizer(info: dict) -> Synthesizer:
    path = info["model_path"]
    mtime = Path(path).stat().st_mtime_ns
    with _load_lock:
        cached = _loaded.get(path)
        # Reload if the checkpoint was replaced on disk
        if cached is None or cached[0] != mtime:
            synth = Synthesizer(
                tts_checkpoint=path,
                tts_config_path=info["config_path"],
                use_cuda=USE_CUDA,
            )
            _loaded[path] = (mtime, synth)
            return synth
        return cached[1]


def synthesize(info: dict, text: str, wav_path: Path, mp3_path: Path) -> None:
    synth = get_synthesizer(info)
    with _tts_lock:
        wav = synth.tts(text)
        synth.save_wav(wav, str(wav_path))
        sample_rate = synth.output_sample_rate

    audio = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    mp3_path.write_bytes(encoder.encode(pcm.tobytes()) + encoder.flush())

class Generation(BaseModel):
    language: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=1000)

AUDIO_TTL_SECONDS = 60
SWEEP_INTERVAL_SECONDS = 15

def delete_old_audio() -> None:
    """Remove generated .wav/.mp3 files older than AUDIO_TTL_SECONDS (60)."""
    cutoff = time.time() - AUDIO_TTL_SECONDS
    for path in AUDIO_DIR.iterdir():
        if path.suffix not in {".wav", ".mp3"}:
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            # File was already removed or is still in use; try again next sweep
            pass


async def cleanup_loop() -> None:
    while True:
        await run_in_threadpool(delete_old_audio)
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    delete_old_audio()  # clear leftovers from previous runs
    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


@app.get("/")
async def serve_home(request: Request):
    registry = await run_in_threadpool(discover_models)
    return templates.TemplateResponse(
        request, "index.html", {"languages": sorted(registry)}
    )


@app.get("/language")
async def get_models(request: Request, language: str):
    registry = await run_in_threadpool(discover_models)
    models = sorted(registry.get(language.strip().lower(), {}))
    return templates.TemplateResponse(request, "partials/model.html", {"models": models})


@app.post("/generate")
async def handle_submit_form(
    request: Request, generation: Annotated[Generation, Form()]
):
    language = generation.language.strip().lower()

    registry = await run_in_threadpool(discover_models)
    info = registry.get(language, {}).get(generation.model)
    if info is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown language/model: {generation.language!r} / {generation.model!r}",
        )

    stem = uuid.uuid4().hex
    wav_path = AUDIO_DIR / f"{stem}.wav"
    mp3_path = AUDIO_DIR / f"{stem}.mp3"

    await run_in_threadpool(synthesize, info, generation.prompt, wav_path, mp3_path)

    return templates.TemplateResponse(
        request,
        "partials/audio-player.html",
        {
            "wav_url": f"/static/audio/{stem}.wav",
            "mp3_url": f"/static/audio/{stem}.mp3",
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
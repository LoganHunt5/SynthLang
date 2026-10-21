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


MODEL_REGISTRY = discover_models()

_loaded: dict[tuple[str, str], Synthesizer] = {}
_load_lock = threading.Lock()
_tts_lock = threading.Lock()

USE_CUDA = torch.cuda.is_available()


def get_synthesizer(language: str, model: str) -> Synthesizer:
    key = (language, model)
    with _load_lock:
        if key not in _loaded:
            info = MODEL_REGISTRY[language][model]
            _loaded[key] = Synthesizer(
                tts_checkpoint=info["model_path"],
                tts_config_path=info["config_path"],
                use_cuda=USE_CUDA,
            )
        return _loaded[key]


def synthesize(language: str, model: str, text: str, wav_path: Path, mp3_path: Path) -> None:
    synth = get_synthesizer(language, model)
    with _tts_lock:
        wav = synth.tts(text)
        synth.save_wav(wav, str(wav_path))
        sample_rate = synth.output_sample_rate

    # float audio -> 16-bit PCM -> MP3
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

#lifespan. Deletes audio files after an amount of time to avoid buildup.


#app = FastAPI(lifespan=lifespan)
app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


@app.get("/")
async def serve_home(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"languages": sorted(MODEL_REGISTRY)}
    )


@app.get("/language")
async def get_models(request: Request, language: str):
    models = sorted(MODEL_REGISTRY.get(language.lower(), {}))
    return templates.TemplateResponse(request, "partials/model.html", {"models": models})


@app.post("/generate")
async def handle_submit_form(
    request: Request, generation: Annotated[Generation, Form()]
):
    language = generation.language.lower()

    # Validating against the registry also blocks path traversal tricks
    # if generation.model not in MODEL_REGISTRY.get(language, {}):
    #     raise HTTPException(status_code=400, detail="Unknown language/model")
    if generation.model not in MODEL_REGISTRY.get(language, {}):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown language/model: {language!r} / {generation.model!r}. "
                f"Registry: { {k: list(v) for k, v in MODEL_REGISTRY.items()} }",
        )
    stem = uuid.uuid4().hex
    wav_path = AUDIO_DIR / f"{stem}.wav"
    mp3_path = AUDIO_DIR / f"{stem}.mp3"

    await run_in_threadpool(
        synthesize, language, generation.model, generation.prompt, wav_path, mp3_path
    )

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
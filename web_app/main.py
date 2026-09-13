#imports 
from fastapi import FastAPI, Request, Header, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from TTS.api import TTS

import os
import uuid
#others imports
from uuid import uuid4
from typing import Annotated, Union
from pydantic import BaseModel, Field

class Generation(BaseModel):
    language: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
tts = TTS("tts_models/en/ljspeech/vits")


#data structures
MODELS_BY_LANGUAGE = {
    "select": [],
    "english": ["English Test", "English test2"],
    "spanish": ["Spanish Test", "Spanish test2"]
}


#configuring templates directory for Jinja2
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def serve_home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.get("/language")
async def get_models(request: Request, language: str):
    models = MODELS_BY_LANGUAGE.get(language.lower(), [])

    return templates.TemplateResponse(
        "partials/model.html",
        {"request": request, "models":models}
    )

@app.post("/generate")
async def handle_submit_form(request:Request, generation: Annotated[Generation, Form()]):
    filename = f"{uuid.uuid4()}.wav"
    filepath = f"static/audio/{filename}"

    tts.tts_to_file(text=generation.prompt, file_path=filepath)

    # Return audio file
    return templates.TemplateResponse(
        "partials/audio-player.html",
        {"request": request, "audio_url": f"/static/audio/{filename}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

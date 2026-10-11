"""Synthesize a sentence with a trained Coqui Tacotron2 checkpoint.

Writes a wav file (Griffin-Lim, no vocoder) and prints alignment checks, so a
checkpoint can be judged quickly: does attention move through the whole text,
and does the model stop on its own?

    uv run python scripts/synthesize.py tmp/sss_finetune/<run>/ \
        "Hola, este es un modelo entrenado con muy pocos datos." --out hola.wav

Coqui's Tacotron2 decoder means to ignore "stop" until the first half of the
text has been read, but compares against the batch size instead of the text
length, so an early stop signal can end the utterance after a few sounds. By
default this script only lets the model stop once attention reaches the end
of the text (--no-stop-guard restores Coqui's behavior).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.utils.audio import AudioProcessor


def _latest_checkpoint(run_dir: Path) -> Path:
    # checkpoint_<step>.pth / best_model_<step>.pth: pick the highest step
    candidates = [
        path for path in run_dir.glob("*.pth")
        if path.stem.rsplit("_", 1)[-1].isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return max(candidates, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def _guard_stop(model: Tacotron2, num_tokens: int, margin: int) -> None:
    """Suppress the stop token until attention reaches the last tokens."""
    decode = model.decoder.decode

    def guarded_decode(memory):
        output, alignment, stop_token = decode(memory)
        if alignment.argmax().item() < num_tokens - 1 - margin:
            stop_token = torch.full_like(stop_token, -1e4)
        return output, alignment, stop_token

    model.decoder.decode = guarded_decode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Training run folder (has config.json).")
    parser.add_argument("text", help="Sentence to synthesize.")
    parser.add_argument("--checkpoint", type=Path, help="Default: latest in run_dir.")
    parser.add_argument("--out", type=Path, default=Path("synth.wav"))
    parser.add_argument("--r", type=int, help="Frames per decoder step. Default: the "
                        "last gradual_training entry, else config r.")
    parser.add_argument("--no-stop-guard", action="store_true")
    parser.add_argument("--stop-margin", type=int, default=2,
                        help="Allow stopping this many tokens before the end.")
    args = parser.parse_args(argv)

    config = Tacotron2Config()
    config.load_json(str(args.run_dir / "config.json"))
    checkpoint = args.checkpoint or _latest_checkpoint(args.run_dir)

    model = Tacotron2.init_from_config(config)
    # checkpoints written by our own training runs, so loading them is trusted
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    r = args.r or (config.gradual_training[-1][1] if config.gradual_training else config.r)
    model.decoder.set_r(r)

    token_ids = model.tokenizer.text_to_ids(args.text)
    if not args.no_stop_guard:
        _guard_stop(model, len(token_ids), args.stop_margin)

    with torch.no_grad():
        outputs = model.inference(torch.LongTensor(token_ids).unsqueeze(0))

    mel = outputs["model_outputs"][0].numpy()
    alignment = outputs["alignments"][0].numpy()
    ap = AudioProcessor.init_from_config(config)
    wav = ap.inv_melspectrogram(mel.T)
    ap.save_wav(wav, str(args.out))

    peaks = alignment.argmax(1)
    steps = alignment.shape[0]
    stopped = steps < model.decoder.max_decoder_steps
    print(f"checkpoint: {checkpoint.name}  r={r}")
    print(f"tokens={len(token_ids)} decoder_steps={steps} "
          f"stopped={'yes' if stopped else 'NO (hit max_decoder_steps)'}")
    print(f"attention: reached token {peaks[-1]}/{len(token_ids) - 1}, "
          f"monotonic {np.mean(np.diff(peaks) >= -1):.0%}, "
          f"mean peak {alignment.max(1).mean():.2f}")
    print(f"wrote {args.out} ({len(wav) / config.audio.sample_rate:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

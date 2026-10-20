"""Score every checkpoint in a Tacotron2 run by how well it aligns at inference.

best_model is picked by eval loss, which feeds the model the real audio
(teacher forcing), so it cannot tell whether the model can read through a
sentence on its own. This script synthesizes test sentences with each
checkpoint and checks the attention:

  - longest dwell: the most time attention spends on one token. Stuck
    attention shows up as one sound held for seconds (drone/buzz).
  - reached end: attention gets to the last tokens of the text.
  - stopped: the model predicts "stop" before max_decoder_steps.
  - s/token: length per input token (~0.06-0.10 s is normal speech).
  - peak: mean attention peak; sharp, confident attention is close to 1.

    uv run python scripts/check_alignment.py tmp/sss_finetune/<run>/
    uv run python scripts/check_alignment.py tmp/sss_finetune/<run>/ \
        --checkpoint tmp/sss-finetune-snapshot/best_model.pth --wav-dir tmp/align_wavs

Like synthesize.py, it only lets the model stop once attention reaches the end
of the text (Coqui's own early-stop check is broken; --no-stop-guard to
disable).
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.utils.audio import AudioProcessor

from synthesize import _guard_stop

# The config's test sentences plus one that is not in the training data.
DEFAULT_SENTENCES = [
    "Hola, este es un modelo entrenado con muy pocos datos.",
    "La prevalencia mundial estimada es muy alta en niños.",
    "Wikipedia es una enciclopedia libre y colaborativa.",
    "Mañana vamos a visitar a mis abuelos en el campo.",
]


@dataclass
class SentenceResult:
    text: str
    tokens: int
    seconds: float
    dwell_char: str
    dwell_seconds: float
    reached_end: bool
    stopped: bool
    mean_peak: float
    passed: bool


def _checkpoints(run_dir: Path) -> list[Path]:
    paths = sorted(run_dir.glob("*.pth"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return paths


def _score(model: Tacotron2, ap: AudioProcessor, text: str, r: int, args,
           wav_path: Path | None) -> SentenceResult:
    token_ids = model.tokenizer.text_to_ids(text)
    chars = [model.tokenizer.characters.id_to_char(i) for i in token_ids]
    n = len(token_ids)

    decode = model.decoder.decode
    if not args.no_stop_guard:
        _guard_stop(model, n, args.stop_margin)
    try:
        with torch.no_grad():
            outputs = model.inference(torch.LongTensor(token_ids).unsqueeze(0))
    finally:
        model.decoder.decode = decode

    alignment = outputs["alignments"][0].numpy()
    peaks = alignment.argmax(1)
    step_seconds = r * ap.hop_length / ap.sample_rate
    seconds = len(peaks) * step_seconds

    dwell_token, dwell_steps = collections.Counter(peaks).most_common(1)[0]
    dwell_seconds = dwell_steps * step_seconds
    reached_end = peaks.max() >= n - 1 - args.stop_margin
    stopped = len(peaks) < model.decoder.max_decoder_steps
    mean_peak = float(alignment.max(1).mean())

    passed = (dwell_seconds <= args.max_dwell and reached_end and stopped
              and seconds / n <= args.max_sec_per_token)

    if wav_path is not None:
        mel = outputs["model_outputs"][0].numpy()
        ap.save_wav(ap.inv_melspectrogram(mel.T), str(wav_path))

    return SentenceResult(text, n, seconds, chars[dwell_token], dwell_seconds,
                          reached_end, stopped, mean_peak, passed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Training run folder (has config.json).")
    parser.add_argument("--checkpoint", type=Path, action="append",
                        help="Score only this checkpoint (repeatable). "
                        "Default: every .pth in run_dir.")
    parser.add_argument("--sentence", action="append",
                        help="Test sentence (repeatable). Default: built-in Spanish set.")
    parser.add_argument("--r", type=int, help="Frames per decoder step. Default: the "
                        "last gradual_training entry, else config r.")
    parser.add_argument("--max-dwell", type=float, default=0.5,
                        help="Fail if attention holds one token longer (seconds).")
    parser.add_argument("--max-sec-per-token", type=float, default=0.12,
                        help="Fail if the audio is longer than this per token.")
    parser.add_argument("--wav-dir", type=Path,
                        help="Also write each synthesized sentence here (Griffin-Lim).")
    parser.add_argument("--no-stop-guard", action="store_true")
    parser.add_argument("--stop-margin", type=int, default=2,
                        help="Allow stopping this many tokens before the end.")
    args = parser.parse_args(argv)

    config = Tacotron2Config()
    config.load_json(str(args.run_dir / "config.json"))
    model = Tacotron2.init_from_config(config)
    model.eval()
    ap = AudioProcessor.init_from_config(config)
    r = args.r or (config.gradual_training[-1][1] if config.gradual_training else config.r)
    sentences = args.sentence or DEFAULT_SENTENCES
    if args.wav_dir:
        args.wav_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for path in args.checkpoint or _checkpoints(args.run_dir):
        # checkpoints written by our own training runs, so loading them is trusted
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.decoder.set_r(r)
        step = int(checkpoint.get("step", -1))
        del checkpoint

        print(f"\n{path.name} (step {step}, r={r})")
        print(f"  {'sentence':34s} {'tok':>3s} {'len':>6s} {'s/tok':>5s} "
              f"{'longest dwell':>14s} {'end':>4s} {'stop':>4s} {'peak':>5s}")
        results = []
        for i, text in enumerate(sentences, 1):
            wav = args.wav_dir / f"{path.stem}_{i}.wav" if args.wav_dir else None
            res = _score(model, ap, text, r, args, wav)
            results.append(res)
            label = text if len(text) <= 34 else text[:31] + "..."
            print(f"  {label:34s} {res.tokens:3d} {res.seconds:5.1f}s "
                  f"{res.seconds / res.tokens:5.2f} {res.dwell_char!r:>8s} {res.dwell_seconds:4.1f}s "
                  f"{'yes' if res.reached_end else 'NO':>4s} {'yes' if res.stopped else 'NO':>4s} "
                  f"{res.mean_peak:5.2f}  {'ok' if res.passed else 'FAIL'}")
        summary.append((path.name, step, results))

    # best first: most sentences passed, then shortest worst dwell, then sharpest
    summary.sort(key=lambda item: (
        -sum(res.passed for res in item[2]),
        max(res.dwell_seconds for res in item[2]),
        -np.mean([res.mean_peak for res in item[2]]),
    ))
    print(f"\nRanking (pass = dwell <= {args.max_dwell}s, reaches end, stops, "
          f"<= {args.max_sec_per_token}s/token)")
    print(f"  {'checkpoint':28s} {'step':>6s} {'passed':>6s} {'worst dwell':>11s} "
          f"{'mean peak':>9s} {'total len':>9s}")
    for name, step, results in summary:
        print(f"  {name:28s} {step:6d} {sum(r.passed for r in results):3d}/{len(results):<2d} "
              f"{max(r.dwell_seconds for r in results):10.1f}s "
              f"{np.mean([r.mean_peak for r in results]):9.2f} "
              f"{sum(r.seconds for r in results):8.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

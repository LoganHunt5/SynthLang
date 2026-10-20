"""Write a small random subset of a ``path|text`` metadata file.

Simulates a low-resource dataset by keeping clips until a target duration is
reached. The output keeps the input format (Coqui's ``css10`` formatter), so it
can point at the original audio without copying it.

    uv run python scripts/make_small_subset.py tmp/SSS_train.txt tmp/SSS_small_15min.txt \
        --minutes 15
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf


# Letters, Spanish accents, and the punctuation the model's vocabulary covers.
# Lines with anything else (digits, quotes, foreign letters) are skipped rather
# than rewritten, since the cleaners would otherwise mispronounce them.
ALLOWED_TEXT = re.compile(r"^[A-Za-zÁÉÍÓÚÜÑáéíóúüñ!'(),\-.:;? ¿¡]+$")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metadata", type=Path, help="Input path|text metadata file.")
    parser.add_argument("output", type=Path, help="Where to write the subset.")
    parser.add_argument("--minutes", type=float, default=15.0, help="Target audio duration.")
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--min-peak", type=float, default=1e-3,
                        help="Skip clips whose loudest sample is below this (silent "
                        "recordings; one session of SSS_Output is all silence).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    # audio paths in the metadata are relative to the metadata file's folder
    root = args.metadata.parent
    lines = args.metadata.read_text(encoding="utf-8").splitlines()

    candidates = []
    skipped = {"text": 0, "duration": 0, "missing": 0, "silent": 0}
    for line in lines:
        if "|" not in line:
            continue
        audio, text = (part.strip() for part in line.split("|", 1))
        if not ALLOWED_TEXT.match(text):
            skipped["text"] += 1
            continue
        audio_path = root / audio
        if not audio_path.exists():
            skipped["missing"] += 1
            continue
        duration = sf.info(str(audio_path)).duration
        if not args.min_seconds <= duration <= args.max_seconds:
            skipped["duration"] += 1
            continue
        # a transcript paired with silence teaches the model to output silence
        if np.abs(sf.read(str(audio_path))[0]).max() < args.min_peak:
            skipped["silent"] += 1
            continue
        candidates.append((audio, text, duration))

    random.Random(args.seed).shuffle(candidates)
    target = args.minutes * 60
    subset, total = [], 0.0
    for audio, text, duration in candidates:
        if total >= target:
            break
        subset.append(f"{audio}|{text}")
        total += duration

    args.output.write_text("\n".join(subset) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(subset)} clips ({total / 60:.1f} min) to {args.output} "
        f"from {len(candidates)} usable of {len(lines)} lines; skipped {skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

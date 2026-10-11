"""Convert a legacy Coqui checkpoint into a weights-only checkpoint.

Older released Coqui models pickle Python objects (e.g. optimizer instances)
next to their weights, so PyTorch >= 2.6 refuses to load them by default. This
loads such a file once, with the safety check disabled, and saves only the model
weights so the training pipeline can warm start from it with the check enabled.

Only run this on checkpoints from a trusted source, such as Coqui's released
models: loading with ``weights_only=False`` can execute code in the file.

    uv run python scripts/convert_checkpoint.py \
        tmp/tts_models--en--ek1--tacotron2/model_file.pth tmp/ek1_tacotron2_weights.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="Trusted legacy checkpoint.")
    parser.add_argument("output", type=Path, help="Where to write the weights-only copy.")
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # step/epoch are reset by the trainer when fine-tuning, so only weights matter
    torch.save({"model": checkpoint["model"], "step": 0, "epoch": 0}, args.output)

    torch.load(args.output, map_location="cpu", weights_only=True)
    print(f"Wrote {len(checkpoint['model'])} weight tensors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

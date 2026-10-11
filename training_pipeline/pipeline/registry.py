"""Maps the ``backend`` name in a config file to the Trainer that implements it."""

from __future__ import annotations

import logging
from collections.abc import Callable

from pipeline.trainer import Trainer


TrainerFactory = Callable[[logging.Logger], Trainer]


# Backends are imported lazily so that selecting one backend never requires
# the dependencies of another (e.g. torch/Coqui) to be installed.
def _coqui(logger: logging.Logger) -> Trainer:
    from pipeline.coqui_trainer import CoquiTrainer

    return CoquiTrainer(logger)


BACKENDS: dict[str, TrainerFactory] = {
    "coqui": _coqui,
}


def get_trainer_factory(backend: str) -> TrainerFactory:
    """Return the factory registered for ``backend``."""
    try:
        return BACKENDS[backend]
    except KeyError:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(
            f"Unknown backend '{backend}'. Available backends: {available}"
        ) from None

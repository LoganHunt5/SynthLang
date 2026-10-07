from abc import ABC, abstractmethod
from typing import Any


class Trainer(ABC):
    """Interface implemented by every model training backend."""

    @abstractmethod
    def train(
        self,
        config: Any,
        *,
        warm_start_checkpoint: str | None = None,
    ) -> None:
        """Train the configured model, optionally restoring a warm-start checkpoint."""
        raise NotImplementedError("Subclasses must implement the train method.")
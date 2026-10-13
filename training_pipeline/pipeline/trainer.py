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
        resume_path: str | None = None,
    ) -> None:
        """Train the configured model.

        ``warm_start_checkpoint`` starts a new run from existing model weights.
        ``resume_path`` continues an interrupted run from its output folder,
        including its optimizer state and step/epoch counters.
        """
        raise NotImplementedError("Subclasses must implement the train method.")
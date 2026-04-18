from __future__ import annotations

from abc import ABC, abstractmethod

from nano_omni.types import StageConfig, StageInput, StageOutput


class StageEngine(ABC):
    """
    Base class for each inference stage.
    Exposes only: add_request / step / has_unfinished
    """

    def __init__(self, config: StageConfig):
        self.config = config

    @abstractmethod
    def add_request(self, inp: StageInput) -> None:
        """Add new request to processing queue."""

    @abstractmethod
    def step(self) -> list[StageOutput]:
        """
        Execute one scheduling step. Returns list of completed requests.
        Unfinished requests remain in internal queue for next step().
        """

    @abstractmethod
    def has_unfinished(self) -> bool:
        """Whether there are unfinished requests."""

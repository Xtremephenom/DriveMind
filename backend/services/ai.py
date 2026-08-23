from __future__ import annotations

from abc import ABC, abstractmethod

from backend.models.system import (
    AICase,
    AIResponse,
)


class AIProvider(ABC):
    """
    Interface for a local AI provider.

    Implementations must not directly modify the filesystem.
    """

    @abstractmethod
    def analyze(self, case: AICase) -> AIResponse:
        raise NotImplementedError
"""Native, offline decision execution. No GPU imports or network calls at module import time."""

from .core import DecisionEngine, EngineConfig

__all__ = ["DecisionEngine", "EngineConfig"]

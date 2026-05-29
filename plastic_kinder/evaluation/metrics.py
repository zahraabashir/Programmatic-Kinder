from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    length: int
    success: bool
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class EvaluationResult:
    mean_reward: float
    success_rate: float
    mean_length: float
    episodes: tuple[EpisodeResult, ...]
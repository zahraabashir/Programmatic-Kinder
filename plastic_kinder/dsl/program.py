from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from plastic_kinder.envs.kinder_env import KinderEnv


class Primitive(Protocol):
    """A DSL primitive maps state to an environment action."""

    name: str

    def __call__(self, obs: np.ndarray, env: KinderEnv) -> np.ndarray:
        ...


@dataclass(frozen=True)
class Program:
    """A simple sequential DSL program.

    For now, a program is a finite sequence of primitives. The policy executes
    primitive 0, then primitive 1, ..., and repeats the last primitive.
    """

    primitives: tuple[Primitive, ...]

    @property
    def name(self) -> str:
        return " ; ".join(p.name for p in self.primitives)

    def __len__(self) -> int:
        return len(self.primitives)

    def primitive_at(self, index: int) -> Primitive:
        if not self.primitives:
            raise ValueError("Program has no primitives.")
        return self.primitives[min(index, len(self.primitives) - 1)]
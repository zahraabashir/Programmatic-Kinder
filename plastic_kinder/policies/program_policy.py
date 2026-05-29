from __future__ import annotations

import numpy as np

from plastic_kinder.dsl.program import Program
from plastic_kinder.envs.kinder_env import KinderEnv
from plastic_kinder.policies.base import Policy


class ProgramPolicy(Policy):
    """Executes a DSL program as a policy."""

    def __init__(self, program: Program, steps_per_primitive: int = 10) -> None:
        if steps_per_primitive < 1:
            raise ValueError("steps_per_primitive must be >= 1.")
        self.program = program
        self.steps_per_primitive = steps_per_primitive
        self._timestep = 0

    def reset(self) -> None:
        self._timestep = 0

    def act(self, obs: np.ndarray, env: KinderEnv) -> np.ndarray:
        primitive_index = self._timestep // self.steps_per_primitive
        primitive = self.program.primitive_at(primitive_index)
        self._timestep += 1
        return primitive(obs, env)
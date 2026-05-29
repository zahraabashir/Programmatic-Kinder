from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from plastic_kinder.envs.kinder_env import KinderEnv


class Policy(ABC):
    """Base interface for all policies."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal policy state."""

    @abstractmethod
    def act(self, obs: np.ndarray, env: KinderEnv) -> np.ndarray:
        """Return an action for the current observation."""
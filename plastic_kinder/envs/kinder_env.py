from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import kinder
import numpy as np


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class KinderEnv:
    """Thin wrapper around a KinDER Gymnasium environment.

    This keeps KinDER-specific details out of the DSL and policy code.
    """

    def __init__(self, env_id: str, render_mode: str | None = None) -> None:
        kinder.register_all_environments()
        kwargs: dict[str, Any] = {}
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        self.env = kinder.make(env_id, **kwargs)
        self.env_id = env_id

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action: np.ndarray) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return StepResult(
            obs=obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )

    def render(self) -> np.ndarray:
        return self.env.render()

    def devectorize(self, obs: np.ndarray) -> Any:
        """Convert vector observation to KinDER object-centric state."""
        return self.env.observation_space.devectorize(obs)

    def zero_action(self) -> np.ndarray:
        """Return a neutral action if the action space is Box-like.

        This is a safe placeholder for the first DSL implementation.
        Later, we will replace this with meaningful KinDER skill parameters.
        """
        if not hasattr(self.action_space, "shape"):
            raise TypeError(f"Unsupported action space: {self.action_space}")
        return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)

    def sample_action(self) -> np.ndarray:
        return self.action_space.sample()

    def close(self) -> None:
        self.env.close()
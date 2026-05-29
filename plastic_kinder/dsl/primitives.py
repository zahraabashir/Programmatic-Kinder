from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from plastic_kinder.envs.kinder_env import KinderEnv
from plastic_kinder.envs.state_utils import (
    obstruction_xy,
    robot_xy,
    target_block_xy,
    target_surface_xy,
)


@dataclass(frozen=True)
class ActionPrimitive:
    """Primitive that maps an observation to a continuous KinDER action."""

    name: str
    fn: Callable[[np.ndarray, KinderEnv], np.ndarray]

    def __call__(self, obs: np.ndarray, env: KinderEnv) -> np.ndarray:
        return self.fn(obs, env)


def _clip_action(action: np.ndarray, env: KinderEnv) -> np.ndarray:
    return np.clip(action, env.action_space.low, env.action_space.high).astype(
        env.action_space.dtype
    )


def _move_toward_xy(
    obs: np.ndarray,
    env: KinderEnv,
    target_xy: tuple[float, float],
    vacuum: float = 0.0,
) -> np.ndarray:
    """Move robot base toward a target point.

    Action format inferred from inspect_env:
    [dx, dy, dtheta, darm_joint, vacuum]
    """
    state = env.devectorize(obs)
    rx, ry = robot_xy(state)
    tx, ty = target_xy

    direction = np.array([tx - rx, ty - ry], dtype=np.float32)
    norm = float(np.linalg.norm(direction))

    if norm > 1e-8:
        direction = direction / norm

    action = env.zero_action().astype(np.float32)
    action[0] = 0.05 * direction[0]
    action[1] = 0.05 * direction[1]
    action[2] = 0.0
    action[3] = 0.0
    action[4] = vacuum

    return _clip_action(action, env)


def move_to_target_block(obs: np.ndarray, env: KinderEnv) -> np.ndarray:
    state = env.devectorize(obs)
    return _move_toward_xy(obs, env, target_block_xy(state), vacuum=0.0)


def move_to_target_block_vacuum(obs: np.ndarray, env: KinderEnv) -> np.ndarray:
    state = env.devectorize(obs)
    return _move_toward_xy(obs, env, target_block_xy(state), vacuum=1.0)


def move_to_obstruction(obs: np.ndarray, env: KinderEnv) -> np.ndarray:
    state = env.devectorize(obs)
    return _move_toward_xy(obs, env, obstruction_xy(state, index=0), vacuum=0.0)


def move_to_obstruction_vacuum(obs: np.ndarray, env: KinderEnv) -> np.ndarray:
    state = env.devectorize(obs)
    return _move_toward_xy(obs, env, obstruction_xy(state, index=0), vacuum=1.0)


def move_to_target_surface(obs: np.ndarray, env: KinderEnv) -> np.ndarray:
    state = env.devectorize(obs)
    return _move_toward_xy(obs, env, target_surface_xy(state), vacuum=0.0)


def move_to_target_surface_vacuum(obs: np.ndarray, env: KinderEnv) -> np.ndarray:
    state = env.devectorize(obs)
    return _move_toward_xy(obs, env, target_surface_xy(state), vacuum=1.0)


def push_right(obs: np.ndarray, env: KinderEnv) -> np.ndarray:
    del obs
    action = env.zero_action().astype(np.float32)
    action[0] = 0.05
    action[4] = 1.0
    return _clip_action(action, env)


def push_left(obs: np.ndarray, env: KinderEnv) -> np.ndarray:
    del obs
    action = env.zero_action().astype(np.float32)
    action[0] = -0.05
    action[4] = 1.0
    return _clip_action(action, env)


def push_down(obs: np.ndarray, env: KinderEnv) -> np.ndarray:
    del obs
    action = env.zero_action().astype(np.float32)
    action[1] = -0.05
    action[4] = 1.0
    return _clip_action(action, env)


def push_up(obs: np.ndarray, env: KinderEnv) -> np.ndarray:
    del obs
    action = env.zero_action().astype(np.float32)
    action[1] = 0.05
    action[4] = 1.0
    return _clip_action(action, env)


BASE_PRIMITIVES: tuple[ActionPrimitive, ...] = (
    ActionPrimitive("MoveToTargetBlock", move_to_target_block),
    ActionPrimitive("MoveToTargetBlockVacuum", move_to_target_block_vacuum),
    ActionPrimitive("MoveToObstruction", move_to_obstruction),
    ActionPrimitive("MoveToObstructionVacuum", move_to_obstruction_vacuum),
    ActionPrimitive("MoveToTargetSurface", move_to_target_surface),
    ActionPrimitive("MoveToTargetSurfaceVacuum", move_to_target_surface_vacuum),
    ActionPrimitive("PushRight", push_right),
    ActionPrimitive("PushLeft", push_left),
    ActionPrimitive("PushUp", push_up),
    ActionPrimitive("PushDown", push_down),
)
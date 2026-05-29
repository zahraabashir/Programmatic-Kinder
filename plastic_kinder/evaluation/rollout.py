from __future__ import annotations

from statistics import mean

from plastic_kinder.envs.kinder_env import KinderEnv
from plastic_kinder.evaluation.metrics import EpisodeResult, EvaluationResult
from plastic_kinder.policies.base import Policy


def _infer_success(info: dict, reward: float, terminated: bool) -> bool:
    """Best-effort success inference.

    We will update this after inspecting KinDER info fields.
    """
    if "success" in info:
        return bool(info["success"])
    if "is_success" in info:
        return bool(info["is_success"])
    return terminated and reward > 0.0


def run_episode(
    env: KinderEnv,
    policy: Policy,
    seed: int,
    max_steps: int,
) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    policy.reset()

    total_reward = 0.0
    terminated = False
    truncated = False
    last_info: dict = {}

    for t in range(max_steps):
        action = policy.act(obs, env)
        step_result = env.step(action)

        obs = step_result.obs
        total_reward += step_result.reward
        terminated = step_result.terminated
        truncated = step_result.truncated
        last_info = step_result.info

        if terminated or truncated:
            return EpisodeResult(
                total_reward=total_reward,
                length=t + 1,
                success=_infer_success(last_info, step_result.reward, terminated),
                terminated=terminated,
                truncated=truncated,
            )

    return EpisodeResult(
        total_reward=total_reward,
        length=max_steps,
        success=False,
        terminated=terminated,
        truncated=True,
    )


def evaluate_policy(
    env: KinderEnv,
    policy: Policy,
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> EvaluationResult:
    episodes = tuple(
        run_episode(env=env, policy=policy, seed=seed + i, max_steps=max_steps)
        for i in range(num_episodes)
    )

    return EvaluationResult(
        mean_reward=mean(ep.total_reward for ep in episodes),
        success_rate=mean(float(ep.success) for ep in episodes),
        mean_length=mean(ep.length for ep in episodes),
        episodes=episodes,
    )
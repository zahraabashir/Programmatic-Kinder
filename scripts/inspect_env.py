from __future__ import annotations

import argparse

from plastic_kinder.envs.kinder_env import KinderEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="kinder/Obstruction2D-o1-v0")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = KinderEnv(args.env_id)

    obs, info = env.reset(seed=args.seed)
    state = env.devectorize(obs)

    print("Env:", args.env_id)
    print("Reset info:", info)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("\nObject-centric state:")
    print(state.pretty_str())

    action = env.sample_action()
    print("\nSample action:")
    print(action)
    print("Action shape:", getattr(action, "shape", None))
    print("Action dtype:", getattr(action, "dtype", None))

    step_result = env.step(action)
    print("\nAfter one sampled step:")
    print("reward:", step_result.reward)
    print("terminated:", step_result.terminated)
    print("truncated:", step_result.truncated)
    print("info:", step_result.info)

    env.close()


if __name__ == "__main__":
    main()
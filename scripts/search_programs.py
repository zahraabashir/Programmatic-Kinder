from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plastic_kinder.dsl.enumerate import enumerate_programs
from plastic_kinder.dsl.primitives import BASE_PRIMITIVES
from plastic_kinder.envs.kinder_env import KinderEnv
from plastic_kinder.evaluation.rollout import evaluate_policy
from plastic_kinder.policies.program_policy import ProgramPolicy


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/obstruction2d_o1.yaml"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    env = KinderEnv(cfg["env_id"])

    programs = enumerate_programs(
        primitives=BASE_PRIMITIVES,
        max_length=int(cfg["search"]["max_program_length"]),
    )

    print(f"Enumerated {len(programs)} programs.")

    scored = []
    for program in tqdm(programs):
        policy = ProgramPolicy(program, steps_per_primitive=20)
        result = evaluate_policy(
            env=env,
            policy=policy,
            num_episodes=int(cfg["eval"]["num_episodes"]),
            max_steps=int(cfg["eval"]["max_steps"]),
            seed=int(cfg["eval"]["seed"]),
        )
        scored.append((program, result))

    scored.sort(key=lambda x: (x[1].success_rate, x[1].mean_reward), reverse=True)

    print("\nTop programs:")
    for rank, (program, result) in enumerate(scored[:10], start=1):
        print(
            f"{rank:02d}. {program.name} | "
            f"success={result.success_rate:.2f} | "
            f"reward={result.mean_reward:.3f} | "
            f"len={result.mean_length:.1f}"
        )

    env.close()


if __name__ == "__main__":
    main()
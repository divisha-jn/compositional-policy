"""Run and print one episode so you can see Stage 1 (env) + Stage 2 (ego
agent) working together. The ego agent's weights are randomly initialized --
nothing is trained yet, so behavior is expected to look arbitrary. This is
just a shapes/wiring sanity check before we add any training (Stage 3+).

Usage:
    python demo.py                      # run against all three teammates
    python demo.py --teammate goal_a    # run against one teammate
    python demo.py --epsilon 0.3        # add exploration noise for variety
"""

import argparse

import numpy as np
import torch

from env import ACTION_NAMES, TEAMMATES, GridWorld
from ego_agent import EgoAgent, init_history, push_history


def run_episode(env, agent, epsilon, rng, verbose=True):
    obs = env.reset()
    history = init_history(agent.history_len)
    total_reward = 0.0

    if verbose:
        print(f"\n--- teammate: {env.teammate.name} ---")
        print(f"step 0 (E={env.ego_pos}, T={env.teammate_pos})")
        print(env.render())

    for step in range(1, env.max_steps + 1):
        history = push_history(history, obs)
        action, out = agent.act(obs, history, epsilon=epsilon, rng=rng)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if verbose:
            weights_str = ", ".join(f"{w:.2f}" for w in out["weights"].tolist())
            print(f"step {step}: E->{ACTION_NAMES[action]:5s}  "
                  f"T->{ACTION_NAMES[info['teammate_action']]:5s}  "
                  f"reward={reward:+.2f}  composer_weights=[{weights_str}]")
            print(env.render())

        if done:
            break

    if verbose:
        status = "SUCCESS" if info["success"] else "timed out"
        print(f"episode ended ({status}), total_reward={total_reward:.2f}, "
              f"steps={step}")
    return total_reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teammate", choices=list(TEAMMATES.keys()) + ["all"],
                         default="all")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.2,
                         help="exploration noise for the untrained agent, "
                              "purely so the demo isn't a static loop")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    agent = EgoAgent(num_components=2)

    teammate_names = list(TEAMMATES.keys()) if args.teammate == "all" else [args.teammate]
    for name in teammate_names:
        env = GridWorld(teammate=TEAMMATES[name], seed=args.seed)
        for ep in range(args.episodes):
            run_episode(env, agent, epsilon=args.epsilon, rng=rng)


if __name__ == "__main__":
    main()

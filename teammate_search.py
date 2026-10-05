"""Stage 7 (generalized): cheap teammate search.

Design change from the original proposal text: don't search only for high
*regret* (a basis-capacity failure). We now know -- from goal_c, see
diagnostic.py -- that an agent can fail completely (J(online) = -3.00, 0%
success) even when the basis has full capacity to solve the teammate (best
fixed composition nearly matches the oracle). That's a composer/routing
failure, and a regret-only search would never surface it, since regret is
computed against the best *fixed* composition, not the online composer.
So instead we rank candidates by J(online) itself -- how the actual agent,
with its actual adaptive composer, performs -- which catches both failure
modes.

Step 1: a data-driven teammate generator (corner x speed x wait-pattern),
not hand-picked one-offs, so new combinations are cheap to try and so
known bad teammates (like goal_c) can be rediscovered by the grid itself
rather than by being told where to look.

Step 2: a CHEAP pass over the pool -- just greedy eval episodes with the
already-trained agent, no oracle training, no retraining per candidate.
Ranking, not diagnosis; see teammate_diagnose.py (Stage D) for the
expensive follow-up on whichever candidates rank worst here.
"""

import argparse

import numpy as np
import torch

from env import GOALS, GridWorld, ScriptedTeammate, WaitThenGoTeammate
from ego_agent import init_history, push_history
from train import train_agent

CORNER_LABELS = ["A", "B", "C", "D"]  # matches GOALS index order
SPEEDS = {"normal": 1.0, "slow": 0.4}
WAIT_PATTERNS = ["none", "wait_then_go"]


def make_teammate(corner_idx, speed, wait_pattern, trigger_distance=2):
    """One teammate from the (corner, speed, wait_pattern) grid. Purely
    data-driven -- no special-casing for any particular corner/combo."""
    move_prob = SPEEDS[speed]
    name = f"{CORNER_LABELS[corner_idx]}_{speed}_{wait_pattern}"
    if wait_pattern == "wait_then_go":
        return WaitThenGoTeammate(goal_idx=corner_idx, trigger_distance=trigger_distance,
                                   move_prob=move_prob, name=name)
    return ScriptedTeammate(goal_idx=corner_idx, move_prob=move_prob, name=name)


def generate_pool():
    """Every corner x speed x wait-pattern combination. Includes corner D
    (the teammate's own spawn -- see env.py, a degenerate always-STAY
    target) on purpose: the grid shouldn't be hand-filtered to only the
    combinations we already suspect are interesting."""
    pool = {}
    for corner_idx in range(len(GOALS)):
        for speed in SPEEDS:
            for wait_pattern in WAIT_PATTERNS:
                teammate = make_teammate(corner_idx, speed, wait_pattern)
                pool[teammate.name] = teammate
    return pool


@torch.no_grad()
def evaluate_teammate(agent, teammate, num_episodes, seed):
    """Like train.py's evaluate(), but takes a teammate OBJECT directly
    instead of a name to look up in env.TEAMMATES -- generated candidates
    here aren't registered there."""
    env = GridWorld(teammate=teammate, seed=seed)
    returns, successes = [], []
    for _ in range(num_episodes):
        obs = env.reset()
        history = init_history(agent.history_len)
        total_reward = 0.0
        for _ in range(env.max_steps):
            history = push_history(history, obs)
            action, _ = agent.act(obs, history, epsilon=0.0)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        returns.append(total_reward)
        successes.append(float(info["success"]))
    return float(np.mean(returns)), float(np.mean(successes))


def rank_pool(agent, pool, num_episodes, seed):
    """Returns a list of (avg_return, success_rate, name, teammate) sorted
    worst-to-best by avg_return."""
    results = []
    for name, teammate in pool.items():
        avg_return, success_rate = evaluate_teammate(agent, teammate, num_episodes, seed)
        results.append((avg_return, success_rate, name, teammate))
    results.sort(key=lambda r: r[0])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-episodes", type=int, default=8000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eval-episodes", type=int, default=10,
                         help="cheap eval episodes per candidate (no oracle, no retraining)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=== training Stage-3 agent (the agent under search) ===")
    agent = train_agent(episodes=args.agent_episodes, gamma=args.gamma, lr=args.lr,
                         eps_start=args.eps_start, eps_end=args.eps_end,
                         seed=args.seed, verbose=True)
    agent.eval()

    pool = generate_pool()
    print(f"\n=== cheap search over {len(pool)} candidate teammates "
          f"({args.eval_episodes} eval episodes each, no oracle) ===")
    ranked = rank_pool(agent, pool, args.eval_episodes, seed=args.seed + 500)

    print(f"{'rank':<5s}{'teammate':<22s}{'J(online)':>12s}{'success':>10s}")
    for i, (avg_return, success_rate, name, _) in enumerate(ranked, 1):
        print(f"{i:<5d}{name:<22s}{avg_return:12.2f}{success_rate:10.2f}")


if __name__ == "__main__":
    main()

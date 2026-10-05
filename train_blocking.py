"""Train the same compositional EgoAgent architecture on the blocking/
guiding environment (env_blocking.py), using the exact same DQN-style
Q-learning as train.py (experience replay + periodic target network) --
reused directly, not reimplemented, since run_training_episode/ReplayBuffer/
compute_td_loss don't know or care which environment they're pointed at.

This is the first real test of whether the compositional agent learns
something other than "chase the teammate's current position" -- that
heuristic solved every env.py teammate, and (per the note in
env_blocking.py) it even still works there via the `chase` scripted policy.
Training is the actual test: does the learned policy anticipate the branch
point, or does it fall back to something chase-like and pay the long-path
penalty?

Trains across all three env_blocking.TEAMMATES_BLOCKING (early/mid/late
branch timing) -- no train/test split for this environment yet.
"""

import argparse
from collections import defaultdict, deque

import numpy as np
import torch

from env_blocking import BlockingGridWorld, NUM_ACTIONS, OBS_DIM, TEAMMATES_BLOCKING
from ego_agent import EgoAgent, init_history, push_history
from train import ReplayBuffer, linear_epsilon, run_training_episode


@torch.no_grad()
def evaluate(agent, teammate_name, num_episodes, seed):
    """Unlike train.py's evaluate() (binary success), this environment's
    natural metrics are continuous (steps to arrival) and a short/long path
    flag, so report those directly instead of forcing a success rate."""
    env = BlockingGridWorld(teammate=TEAMMATES_BLOCKING[teammate_name], seed=seed)
    returns, steps_list, short_path_flags, arrived_flags = [], [], [], []
    for _ in range(num_episodes):
        obs = env.reset()
        history = init_history(agent.history_len)
        total_reward = 0.0
        for step in range(1, env.max_steps + 1):
            history = push_history(history, obs)
            action, _ = agent.act(obs, history, epsilon=0.0)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        returns.append(total_reward)
        steps_list.append(step)
        short_path_flags.append(float(info["took_short_path"]))
        arrived_flags.append(float(info["arrived"]))
    return {
        "avg_return": float(np.mean(returns)),
        "avg_steps": float(np.mean(steps_list)),
        "short_path_rate": float(np.mean(short_path_flags)),
        "arrival_rate": float(np.mean(arrived_flags)),
    }


def train_agent(episodes=8000, gamma=0.95, lr=1e-3, eps_start=1.0, eps_end=0.05,
                 log_every=None, seed=0, verbose=True, num_components=2,
                 buffer_capacity=5000, batch_size=32, min_buffer_size=200,
                 target_update_every=50):
    """Mirrors train.py's train_agent(), but against BlockingGridWorld /
    TEAMMATES_BLOCKING, with obs_dim/num_actions passed explicitly (they
    happen to equal env.py's defaults, 8 and 5, but that's env_blocking's
    own OBS_DIM/NUM_ACTIONS, not a coincidence we should rely on silently)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    agent = EgoAgent(num_components=num_components, obs_dim=OBS_DIM, num_actions=NUM_ACTIONS)
    target_agent = EgoAgent(num_components=num_components, obs_dim=OBS_DIM, num_actions=NUM_ACTIONS)
    target_agent.load_state_dict(agent.state_dict())
    for param in target_agent.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity, seed=seed)

    teammate_names = list(TEAMMATES_BLOCKING.keys())
    envs = {
        name: BlockingGridWorld(teammate=TEAMMATES_BLOCKING[name], seed=seed + i)
        for i, name in enumerate(teammate_names)
    }

    log_every = log_every or max(1, episodes // 10)
    recent_returns = defaultdict(lambda: deque(maxlen=log_every))
    recent_short_path = defaultdict(lambda: deque(maxlen=log_every))
    recent_steps = defaultdict(lambda: deque(maxlen=log_every))

    for ep in range(1, episodes + 1):
        epsilon = linear_epsilon(ep, episodes, eps_start, eps_end)
        name = teammate_names[rng.integers(len(teammate_names))]
        env = envs[name]
        ep_return, _ = run_training_episode(
            env, agent, target_agent, optimizer, buffer, gamma, epsilon, rng,
            batch_size, min_buffer_size)
        recent_returns[name].append(ep_return)
        recent_short_path[name].append(float(env.teammate.took_short_path))
        recent_steps[name].append(env.t)

        if ep % target_update_every == 0:
            target_agent.load_state_dict(agent.state_dict())

        if verbose and ep % log_every == 0:
            print(f"[episode {ep:5d}] epsilon={epsilon:.2f}  " + "  ".join(
                f"{n}: return={np.mean(recent_returns[n]):+.2f} "
                f"steps={np.mean(recent_steps[n]):.1f} "
                f"short_path={np.mean(recent_short_path[n]):.2f}"
                for n in teammate_names if recent_returns[n]
            ))

    return agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-components", type=int, default=2,
                         help="K, the number of basis components")
    parser.add_argument("--buffer-capacity", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-buffer-size", type=int, default=200)
    parser.add_argument("--target-update-every", type=int, default=50)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    agent = train_agent(episodes=args.episodes, gamma=args.gamma, lr=args.lr,
                         eps_start=args.eps_start, eps_end=args.eps_end,
                         log_every=args.log_every, seed=args.seed, verbose=True,
                         num_components=args.num_components,
                         buffer_capacity=args.buffer_capacity, batch_size=args.batch_size,
                         min_buffer_size=args.min_buffer_size,
                         target_update_every=args.target_update_every)

    print("\n--- final greedy evaluation (epsilon=0) ---")
    for name in TEAMMATES_BLOCKING:
        m = evaluate(agent, name, args.eval_episodes, seed=args.seed + 100)
        print(f"  {name:14s}  avg_return={m['avg_return']:+.2f}  avg_steps={m['avg_steps']:.1f}  "
              f"short_path_rate={m['short_path_rate']:.2f}  arrival_rate={m['arrival_rate']:.2f}")

    if args.save:
        torch.save(agent.state_dict(), args.save)
        print(f"\nsaved trained weights to {args.save}")


if __name__ == "__main__":
    main()

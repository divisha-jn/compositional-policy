"""Stage 3: baseline end-to-end training (encoder + basis + composer),
trained jointly with plain online Q-learning, no regret objective yet.

Each episode is played against a randomly chosen scripted teammate from
env.TEAMMATES. The point of training against a mix (rather than one
teammate at a time) is that it gives the composer a reason to exist: the
same basis has to be recombined differently depending on who it's paired
with. This stage only asks "does it learn something reasonable?" -- returns
should climb and success rate should rise well above a random policy.
"""

import argparse
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn

from env import GridWorld, TEAMMATES
from ego_agent import EgoAgent, epsilon_greedy_action, init_history, push_history


def linear_epsilon(episode, total_episodes, eps_start, eps_end, decay_frac=0.8):
    decay_episodes = max(1, int(total_episodes * decay_frac))
    frac = min(1.0, episode / decay_episodes)
    return eps_start + frac * (eps_end - eps_start)


def run_training_episode(env, agent, optimizer, gamma, epsilon, rng):
    """Plays one episode, doing an online TD(0) Q-learning update every step.
    Returns the total (undiscounted) reward for the episode."""
    obs = env.reset()
    history = init_history(agent.history_len)
    total_reward = 0.0

    for _ in range(env.max_steps):
        history = push_history(history, obs)
        out = agent.compute(obs, history)
        action = epsilon_greedy_action(out["q_combined"], epsilon, rng)

        next_obs, reward, done, info = env.step(action)
        next_history = push_history(history, next_obs)

        with torch.no_grad():
            if done:
                target = torch.tensor(reward, dtype=torch.float32)
            else:
                next_out = agent.compute(next_obs, next_history)
                target = reward + gamma * next_out["q_combined"].max()

        pred = out["q_combined"][action]
        loss = (pred - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs
        history = next_history
        total_reward += reward
        if done:
            break

    return total_reward, info["success"]


@torch.no_grad()
def evaluate(agent, teammate_name, num_episodes, seed):
    env = GridWorld(teammate=TEAMMATES[teammate_name], seed=seed)
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


def train_agent(episodes=8000, gamma=0.95, lr=1e-3, eps_start=1.0, eps_end=0.05,
                 log_every=None, seed=0, verbose=True):
    """Trains a fresh compositional EgoAgent against a random mix of the
    scripted teammates. Returns the trained agent. Used by both this
    script's CLI and by regret.py (Stage 4), which needs a trained
    compositional agent to freeze the basis of."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    agent = EgoAgent(num_components=2)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    teammate_names = list(TEAMMATES.keys())
    envs = {
        name: GridWorld(teammate=TEAMMATES[name], seed=seed + i)
        for i, name in enumerate(teammate_names)
    }

    log_every = log_every or max(1, episodes // 10)
    recent_returns = defaultdict(lambda: deque(maxlen=log_every))
    recent_success = defaultdict(lambda: deque(maxlen=log_every))

    for ep in range(1, episodes + 1):
        epsilon = linear_epsilon(ep, episodes, eps_start, eps_end)
        name = teammate_names[rng.integers(len(teammate_names))]
        ep_return, success = run_training_episode(
            envs[name], agent, optimizer, gamma, epsilon, rng)
        recent_returns[name].append(ep_return)
        recent_success[name].append(float(success))

        if verbose and ep % log_every == 0:
            print(f"[episode {ep:5d}] epsilon={epsilon:.2f}  " + "  ".join(
                f"{n}: return={np.mean(recent_returns[n]):+.2f} "
                f"success={np.mean(recent_success[n]):.2f}"
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
    parser.add_argument("--save", type=str, default=None,
                         help="optional path to save trained weights, e.g. ego_agent.pt")
    args = parser.parse_args()

    agent = train_agent(episodes=args.episodes, gamma=args.gamma, lr=args.lr,
                         eps_start=args.eps_start, eps_end=args.eps_end,
                         log_every=args.log_every, seed=args.seed, verbose=True)

    print("\n--- final greedy evaluation (epsilon=0) ---")
    for name in TEAMMATES:
        avg_return, success_rate = evaluate(agent, name, args.eval_episodes, seed=args.seed + 100)
        print(f"{name:12s}  avg_return={avg_return:+.2f}  success_rate={success_rate:.2f}")

    if args.save:
        torch.save(agent.state_dict(), args.save)
        print(f"\nsaved trained weights to {args.save}")


if __name__ == "__main__":
    main()

"""Stage 3: baseline end-to-end training (encoder + basis + composer),
trained jointly with DQN-style Q-learning (experience replay + a periodic
target network), no regret objective yet.

Each episode is played against a randomly chosen scripted teammate from
env.TEAMMATES. The point of training against a mix (rather than one
teammate at a time) is that it gives the composer a reason to exist: the
same basis has to be recombined differently depending on who it's paired
with. This stage only asks "does it learn something reasonable?" -- returns
should climb and success rate should rise well above a random policy.

The replay buffer + target network (standard DQN stabilizers) replace an
earlier pure-online version of this loop, which -- on a harder teammate --
got stuck in a 2-step action oscillation: bootstrapping every step off the
same network being updated let the composer's shifting mixture weights
flip the greedy action back and forth forever. Sampling random past
transitions and bootstrapping off a slower-moving target copy removes that
feedback loop.
"""

import argparse
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn

from env import GridWorld, TEAMMATES
from ego_agent import EgoAgent, init_history, push_history


def linear_epsilon(episode, total_episodes, eps_start, eps_end, decay_frac=0.8):
    decay_episodes = max(1, int(total_episodes * decay_frac))
    frac = min(1.0, episode / decay_episodes)
    return eps_start + frac * (eps_end - eps_start)


class ReplayBuffer:
    """Fixed-capacity ring buffer of (obs, history, action, reward,
    next_obs, next_history, done) transitions, sampled uniformly."""

    def __init__(self, capacity, seed=0):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idx = self.rng.integers(0, len(self.buffer), size=batch_size)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


def compute_td_loss(agent, target_agent, batch, gamma):
    obs_b = torch.as_tensor(np.stack([t[0] for t in batch]), dtype=torch.float32)
    history_b = torch.as_tensor(np.stack([t[1] for t in batch]), dtype=torch.float32)
    action_b = torch.as_tensor([t[2] for t in batch], dtype=torch.long)
    reward_b = torch.as_tensor([t[3] for t in batch], dtype=torch.float32)
    next_obs_b = torch.as_tensor(np.stack([t[4] for t in batch]), dtype=torch.float32)
    next_history_b = torch.as_tensor(np.stack([t[5] for t in batch]), dtype=torch.float32)
    done_b = torch.as_tensor([t[6] for t in batch], dtype=torch.float32)

    q_combined = agent.compute_batch(obs_b, history_b)  # (B, A)
    pred = q_combined.gather(1, action_b.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_agent.compute_batch(next_obs_b, next_history_b)  # (B, A)
        target = reward_b + gamma * (1.0 - done_b) * next_q.max(dim=1).values

    return nn.functional.mse_loss(pred, target)


def run_training_episode(env, agent, target_agent, optimizer, buffer, gamma, epsilon, rng,
                          batch_size, min_buffer_size):
    """Plays one episode, pushing each transition into the replay buffer and
    (once there's enough data) taking one minibatch TD update per step
    against the frozen target network. Returns the total (undiscounted)
    reward for the episode."""
    obs = env.reset()
    history = init_history(agent.history_len)
    total_reward = 0.0

    for _ in range(env.max_steps):
        history = push_history(history, obs)
        action, _ = agent.act(obs, history, epsilon=epsilon, rng=rng)

        next_obs, reward, done, info = env.step(action)
        next_history = push_history(history, next_obs)

        buffer.push((obs, history, action, reward, next_obs, next_history, float(done)))

        if len(buffer) >= min_buffer_size:
            batch = buffer.sample(batch_size)
            loss = compute_td_loss(agent, target_agent, batch, gamma)
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
                 log_every=None, seed=0, verbose=True, num_components=2,
                 buffer_capacity=5000, batch_size=32, min_buffer_size=200,
                 target_update_every=50):
    """Trains a fresh compositional EgoAgent against a random mix of the
    scripted teammates, using experience replay + a periodic target network.
    Returns the trained agent. Used by both this script's CLI and by
    regret.py (Stage 4), which needs a trained compositional agent to freeze
    the basis of."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    agent = EgoAgent(num_components=num_components)
    target_agent = EgoAgent(num_components=num_components)
    target_agent.load_state_dict(agent.state_dict())
    for param in target_agent.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity, seed=seed)

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
            envs[name], agent, target_agent, optimizer, buffer, gamma, epsilon, rng,
            batch_size, min_buffer_size)
        recent_returns[name].append(ep_return)
        recent_success[name].append(float(success))

        if ep % target_update_every == 0:
            target_agent.load_state_dict(agent.state_dict())

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
    parser.add_argument("--num-components", type=int, default=2,
                         help="K, the number of basis components")
    parser.add_argument("--buffer-capacity", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-buffer-size", type=int, default=200,
                         help="transitions to collect before learning starts")
    parser.add_argument("--target-update-every", type=int, default=50,
                         help="episodes between target network hard updates")
    parser.add_argument("--save", type=str, default=None,
                         help="optional path to save trained weights, e.g. ego_agent.pt")
    args = parser.parse_args()

    agent = train_agent(episodes=args.episodes, gamma=args.gamma, lr=args.lr,
                         eps_start=args.eps_start, eps_end=args.eps_end,
                         log_every=args.log_every, seed=args.seed, verbose=True,
                         num_components=args.num_components,
                         buffer_capacity=args.buffer_capacity, batch_size=args.batch_size,
                         min_buffer_size=args.min_buffer_size,
                         target_update_every=args.target_update_every)

    print("\n--- final greedy evaluation (epsilon=0) ---")
    for name in TEAMMATES:
        avg_return, success_rate = evaluate(agent, name, args.eval_episodes, seed=args.seed + 100)
        print(f"{name:12s}  avg_return={avg_return:+.2f}  success_rate={success_rate:.2f}")

    if args.save:
        torch.save(agent.state_dict(), args.save)
        print(f"\nsaved trained weights to {args.save}")


if __name__ == "__main__":
    main()

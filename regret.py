"""Stage 4: regret metric (evaluation only -- no change to how anything is
trained).

For each scripted teammate we compare two things against a FIXED, already
end-to-end-trained compositional basis (from train.py's Stage 3 agent):

  1. J(best_response, teammate): train a plain, dedicated Q-network from
     scratch against just that one teammate (it never sees the others).
     This is the oracle -- the best a specialized policy could do.

  2. J(best_composition, teammate): freeze the Stage-3 agent's K basis
     Q-networks (don't retrain them) and grid-search over a single fixed
     mixture weight vector w to find the best possible constant recombination
     of that basis against that teammate. This asks "if you had to pick one
     fixed blend of the existing basis, how good is the best blend?" --
     deliberately ignoring the online composer, which picks a different w
     every timestep.

regret = J(best_response) - J(best_composition), logged per teammate. A
large regret means the fixed basis (no matter how you mix it) can't match a
specialist -- i.e. the basis is missing something a specialist has. This
script only measures; it doesn't feed back into training.

Reports this separately for env.TRAIN_TEAMMATES (what the Stage-3 agent
trained on) and env.TEST_TEAMMATES (held out entirely from training) so
in-distribution regret can be compared against generalization regret.
"""

import argparse

import numpy as np
import torch

from env import GridWorld, TEAMMATES, TEST_TEAMMATES, TRAIN_TEAMMATES
from ego_agent import QComponent, epsilon_greedy_action
from train import linear_epsilon, train_agent


def evaluate_policy(policy_fn, env, num_episodes):
    """policy_fn: obs (numpy array) -> action (int). Returns (avg_return, success_rate)."""
    returns, successes = [], []
    for _ in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        for _ in range(env.max_steps):
            action = policy_fn(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        returns.append(total_reward)
        successes.append(float(info["success"]))
    return float(np.mean(returns)), float(np.mean(successes))


def train_oracle(teammate, episodes, gamma, lr, eps_start, eps_end, seed):
    """Trains a plain Q-network (no history, no composer) from scratch
    against a single scripted teammate object -- the best-response oracle."""
    env = GridWorld(teammate=teammate, seed=seed)
    net = QComponent()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    for ep in range(1, episodes + 1):
        epsilon = linear_epsilon(ep, episodes, eps_start, eps_end)
        obs = env.reset()
        for _ in range(env.max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q = net(obs_t).squeeze(0)
            action = epsilon_greedy_action(q, epsilon, rng)

            next_obs, reward, done, info = env.step(action)
            with torch.no_grad():
                if done:
                    target = torch.tensor(reward, dtype=torch.float32)
                else:
                    next_q = net(torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                    target = reward + gamma * next_q.max()

            loss = (q[action] - target) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs
            if done:
                break

    return net


@torch.no_grad()
def fixed_weight_policy(basis, weights):
    """Builds a policy_fn for evaluate_policy() out of a frozen basis and a
    single constant mixture weight vector (no encoder/composer involved)."""
    weights_t = torch.as_tensor(weights, dtype=torch.float32)

    def policy_fn(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_per_component = torch.stack([comp(obs_t) for comp in basis], dim=1).squeeze(0)  # (K, A)
        q_combined = (weights_t.unsqueeze(-1) * q_per_component).sum(dim=0)
        return int(torch.argmax(q_combined).item())

    return policy_fn


def search_best_composition(basis, teammate, num_episodes, seed, resolution=21):
    """Grid/random-searches fixed mixture weights w (on the simplex) for the
    frozen basis, and returns the best return found plus the winning w."""
    num_components = len(basis)
    if num_components == 2:
        candidates = [np.array([w, 1.0 - w]) for w in np.linspace(0.0, 1.0, resolution)]
    else:
        rng = np.random.default_rng(seed)
        candidates = list(rng.dirichlet(np.ones(num_components), size=resolution * 10))

    env = GridWorld(teammate=teammate, seed=seed)
    best_return, best_weights, best_success = -np.inf, None, None
    for w in candidates:
        avg_return, success_rate = evaluate_policy(fixed_weight_policy(basis, w), env, num_episodes)
        if avg_return > best_return:
            best_return, best_weights, best_success = avg_return, w, success_rate

    return best_return, best_weights, best_success


def report_regret(agent, teammate_names, oracle_episodes, gamma, lr, eps_start, eps_end,
                   eval_episodes, search_resolution, seed):
    header = f"{'teammate':14s} {'J(best_response)':>18s} {'J(best_composition)':>20s} {'regret':>10s} {'best_w':>16s}"
    print(header)
    print("-" * len(header))

    for name in teammate_names:
        teammate = TEAMMATES[name]
        oracle_net = train_oracle(teammate, oracle_episodes, gamma, lr, eps_start, eps_end, seed=seed + 1)
        oracle_env = GridWorld(teammate=teammate, seed=seed + 100)
        oracle_policy = lambda obs, net=oracle_net: int(torch.argmax(
            net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)).item())
        j_best_response, br_success = evaluate_policy(oracle_policy, oracle_env, eval_episodes)

        j_best_composition, best_w, bc_success = search_best_composition(
            agent.basis, teammate, eval_episodes, seed=seed + 200, resolution=search_resolution)

        regret = j_best_response - j_best_composition
        w_str = "[" + ", ".join(f"{x:.2f}" for x in best_w) + "]"
        print(f"{name:14s} {j_best_response:18.2f} {j_best_composition:20.2f} {regret:10.2f} {w_str:>16s}")
        print(f"{'':14s} (success={br_success:.2f}){'':7s}(success={bc_success:.2f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-episodes", type=int, default=8000,
                         help="training episodes for the Stage-3 compositional agent")
    parser.add_argument("--oracle-episodes", type=int, default=3000,
                         help="training episodes per best-response oracle")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eval-episodes", type=int, default=30,
                         help="episodes averaged for each J(.) estimate")
    parser.add_argument("--search-resolution", type=int, default=21,
                         help="number of fixed-weight candidates to try (K=2 grid steps)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=== training Stage-3 compositional agent (basis to be frozen) ===")
    print(f"(trained only on: {list(TRAIN_TEAMMATES)})")
    agent = train_agent(episodes=args.agent_episodes, gamma=args.gamma, lr=args.lr,
                         eps_start=args.eps_start, eps_end=args.eps_end,
                         seed=args.seed, verbose=True)
    agent.eval()
    for param in agent.basis.parameters():
        param.requires_grad_(False)

    print("\n=== regret: TRAIN teammates ===")
    report_regret(agent, list(TRAIN_TEAMMATES), args.oracle_episodes, args.gamma, args.lr,
                  args.eps_start, args.eps_end, args.eval_episodes, args.search_resolution, args.seed)

    print("\n=== regret: HELD-OUT TEST teammates ===")
    report_regret(agent, list(TEST_TEAMMATES), args.oracle_episodes, args.gamma, args.lr,
                  args.eps_start, args.eps_end, args.eval_episodes, args.search_resolution, args.seed)


if __name__ == "__main__":
    main()

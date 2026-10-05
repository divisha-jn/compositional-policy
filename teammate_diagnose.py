"""Stage D: the expensive diagnosis, run only on whichever candidate(s)
teammate_search.py's cheap ranking flagged as worst.

Reuses the exact Stage 4/5 machinery (regret.py's train_oracle /
search_best_composition, diagnostic.py's diagnose()) against a teammate
pulled from teammate_search.py's generated pool instead of env.TEAMMATES --
regret.py was generalized to take a teammate object directly for this
reason (it used to take a name and look it up in env.TEAMMATES, which
doesn't know about generated candidates).

J(online) is recomputed here with the same seed/eval-episode settings
teammate_search.py used, rather than hardcoding its reported number, as a
cheap consistency check that the two scripts agree.
"""

import argparse

import torch

from env import GridWorld
from regret import evaluate_policy, search_best_composition, train_oracle
from diagnostic import diagnose
from teammate_search import evaluate_teammate, make_teammate
from train import train_agent


def diagnose_candidate(agent, teammate, oracle_episodes, gamma, lr, eps_start, eps_end,
                        eval_episodes, search_resolution, seed):
    j_online, online_success = evaluate_teammate(agent, teammate, eval_episodes, seed=seed + 500)

    j_best_comp, best_w, bc_success = search_best_composition(
        agent.basis, teammate, eval_episodes, seed=seed + 200, resolution=search_resolution)

    oracle_net = train_oracle(teammate, oracle_episodes, gamma, lr, eps_start, eps_end, seed=seed + 1)
    oracle_env = GridWorld(teammate=teammate, seed=seed + 100)
    oracle_policy = lambda obs, net=oracle_net: int(torch.argmax(
        net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)).item())
    j_oracle, oracle_success = evaluate_policy(oracle_policy, oracle_env, eval_episodes)

    gap = j_best_comp - j_online
    regret = j_oracle - j_best_comp
    return {
        "j_online": j_online, "online_success": online_success,
        "j_best_comp": j_best_comp, "best_w": best_w, "bc_success": bc_success,
        "j_oracle": j_oracle, "oracle_success": oracle_success,
        "gap": gap, "regret": regret,
    }


def print_row(name, r, gap_threshold, regret_threshold):
    diagnosis = diagnose(r["gap"], r["regret"], gap_threshold, regret_threshold)
    print(f"{name:14s}{r['j_online']:11.2f}{r['j_best_comp']:11.2f}{r['gap']:7.2f}"
          f"{r['j_oracle']:11.2f}{r['regret']:8.2f}  {diagnosis}")
    print(f"{'':14s}(succ={r['online_success']:.2f}) (succ={r['bc_success']:.2f})"
          f"{'':11s}(succ={r['oracle_success']:.2f})")
    w_str = "[" + ", ".join(f"{x:.2f}" for x in r["best_w"]) + "]"
    print(f"{'':14s}best fixed w = {w_str}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-episodes", type=int, default=8000)
    parser.add_argument("--oracle-episodes", type=int, default=3000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--search-resolution", type=int, default=21)
    parser.add_argument("--gap-threshold", type=float, default=1.0)
    parser.add_argument("--regret-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=== training Stage-3 agent (same as teammate_search.py's default seed) ===")
    agent = train_agent(episodes=args.agent_episodes, gamma=args.gamma, lr=args.lr,
                         eps_start=args.eps_start, eps_end=args.eps_end,
                         seed=args.seed, verbose=True)
    agent.eval()
    for param in agent.basis.parameters():
        param.requires_grad_(False)

    # corner index 2 = C, per teammate_search.CORNER_LABELS
    candidate = make_teammate(corner_idx=2, speed="normal", wait_pattern="none")
    assert candidate.name == "C_normal_none"

    print(f"\n=== Stage D diagnosis: {candidate.name} (rank #1 worst from teammate_search.py) ===")
    header = (f"{'teammate':14s}{'J(online)':>11s}{'J(best_w)':>11s}{'gap':>7s}"
              f"{'J(oracle)':>11s}{'regret':>8s}  diagnosis")
    print(header)
    print("-" * len(header))

    result = diagnose_candidate(agent, candidate, args.oracle_episodes, args.gamma, args.lr,
                                 args.eps_start, args.eps_end, args.eval_episodes,
                                 args.search_resolution, args.seed)
    print_row(candidate.name, result, args.gap_threshold, args.regret_threshold)


if __name__ == "__main__":
    main()

"""Stage 5: oracle-composition diagnostic (evaluation only).

Extends Stage 4's regret metric with one more comparison, per teammate:

  (a) J(online composer)     -- the actual trained agent, using the
                                 composer's real per-timestep weights
  (b) J(best composition)    -- same frozen basis, but the single best FIXED
                                 weight vector found by search (Stage 4)
  (c) J(best response)       -- a dedicated oracle trained against just this
                                 teammate (Stage 4)

Two derived numbers per teammate:
  gap    = (b) - (a)   how much return is left on the table by the online
                        composer's routing, given a basis that could already
                        do better with the right (fixed) weights
  regret = (c) - (b)   how much return is left on the table by the basis
                        itself, even with the best possible fixed blend

Large gap, small regret  -> composer/routing problem: the basis is fine,
                             the composer isn't finding/using it well.
Small gap, large regret  -> basis problem: no blend of the current K
                             components can reach oracle performance, so the
                             basis is missing a component.

Reports this separately for env.TRAIN_TEAMMATES (what the Stage-3 agent
trained on) and env.TEST_TEAMMATES (held out entirely from training), so a
generalization failure (high gap/regret only on the held-out teammate) can
be told apart from an in-distribution one.
"""

import argparse

import torch

from env import GridWorld, TEAMMATES, TEST_TEAMMATES, TRAIN_TEAMMATES
from train import evaluate, train_agent
from regret import evaluate_policy, search_best_composition, train_oracle


def diagnose(gap, regret, gap_threshold, regret_threshold):
    if gap > gap_threshold:
        return "COMPOSER: routing leaves basis return on the table"
    if regret > regret_threshold:
        return "BASIS: missing a component (best blend still << oracle)"
    return "no significant issue"


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
    parser.add_argument("--gap-threshold", type=float, default=1.0,
                         help="flag as a composer/routing problem above this gap")
    parser.add_argument("--regret-threshold", type=float, default=1.0,
                         help="flag as a basis problem above this regret")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-components", type=int, default=2,
                         help="K, the number of basis components (try 1 to sanity-check "
                              "that the diagnostic correctly blames the basis, not the composer)")
    args = parser.parse_args()

    print("=== training Stage-3 compositional agent (basis to be frozen) ===")
    print(f"(trained only on: {list(TRAIN_TEAMMATES)})")
    agent = train_agent(episodes=args.agent_episodes, gamma=args.gamma, lr=args.lr,
                         eps_start=args.eps_start, eps_end=args.eps_end,
                         seed=args.seed, verbose=True, num_components=args.num_components)
    agent.eval()
    for param in agent.basis.parameters():
        param.requires_grad_(False)

    for group_label, teammate_names in [("TRAIN", list(TRAIN_TEAMMATES)),
                                         ("HELD-OUT TEST", list(TEST_TEAMMATES))]:
        print(f"\n=== Stage 5 ({group_label} teammates): "
              f"online composer vs. best fixed composition vs. oracle ===")
        header = (f"{'teammate':14s}{'J(online)':>11s}{'J(best_w)':>11s}{'gap':>7s}"
                  f"{'J(oracle)':>11s}{'regret':>8s}  diagnosis")
        print(header)
        print("-" * len(header))

        for name in teammate_names:
            j_online, online_success = evaluate(agent, name, args.eval_episodes, seed=args.seed + 300)

            j_best_comp, best_w, bc_success = search_best_composition(
                agent.basis, name, args.eval_episodes, seed=args.seed + 200,
                resolution=args.search_resolution)

            oracle_net = train_oracle(name, args.oracle_episodes, args.gamma, args.lr,
                                       args.eps_start, args.eps_end, seed=args.seed + 1)
            oracle_env = GridWorld(teammate=TEAMMATES[name], seed=args.seed + 100)
            oracle_policy = lambda obs, net=oracle_net: int(torch.argmax(
                net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)).item())
            j_oracle, oracle_success = evaluate_policy(oracle_policy, oracle_env, args.eval_episodes)

            gap = j_best_comp - j_online
            regret = j_oracle - j_best_comp
            diagnosis = diagnose(gap, regret, args.gap_threshold, args.regret_threshold)

            print(f"{name:14s}{j_online:11.2f}{j_best_comp:11.2f}{gap:7.2f}"
                  f"{j_oracle:11.2f}{regret:8.2f}  {diagnosis}")
            print(f"{'':14s}(succ={online_success:.2f}) (succ={bc_success:.2f})"
                  f"{'':11s}(succ={oracle_success:.2f})")
            w_str = "[" + ", ".join(f"{x:.2f}" for x in best_w) + "]"
            print(f"{'':14s}best fixed w = {w_str}")


if __name__ == "__main__":
    main()

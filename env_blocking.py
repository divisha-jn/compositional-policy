"""Stage B: a blocking/guiding environment variant.

A new, standalone environment (env.py is left untouched for comparison).
The key behavioral change from env.py:

  - The teammate pursues its OWN fixed goal, autonomously, every step. It
    is NOT trying to reach the ego agent anymore -- it never even looks at
    where the ego is, except at one specific moment (see below).

  - The teammate's route passes through a fixed "branch point" cell,
    checked once: if the ego is NOT adjacent to the teammate at the exact
    step the teammate reaches that cell, the teammate gets stuck there for
    a few steps (the "long way"). If the ego IS adjacent at that moment,
    the teammate proceeds immediately (the "short way").

  - Reward depends only on how fast the teammate reaches its own goal
    (-1 per step, plus a bonus if the short path was taken). The ego's own
    position is otherwise irrelevant to the reward -- there's no "meet on
    the same cell" condition like in env.py.

This is designed so "walk toward wherever the teammate currently is" --
which trivially solved every env.py teammate -- does NOT work here: the
teammate moves under its own steam at the same speed as the ego, so if the
ego always heads for the teammate's live position, the gap to a teammate
moving away never closes (a tailing chase at equal speed doesn't gain
ground). The ego has to anticipate the branch point -- get there in
advance -- rather than react to where the teammate is right now.

Same grid size and ego action space as env.py (reused directly below) so
the existing ego_agent.py architecture still applies unchanged.
"""

import numpy as np

from env import ACTION_NAMES, ACTIONS, GRID_SIZE, NUM_ACTIONS

STEP_PENALTY = -1.0
SHORT_PATH_BONUS = 5.0
MAX_STEPS = 30
DEFAULT_DETOUR_LEN = 3


def _clip(pos):
    r, c = pos
    return (max(0, min(GRID_SIZE - 1, r)), max(0, min(GRID_SIZE - 1, c)))


def _move(pos, action_idx):
    dr, dc = ACTIONS[action_idx]
    return _clip((pos[0] + dr, pos[1] + dc))


def _greedy_action_towards(pos, target):
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    if dr == 0 and dc == 0:
        return 4  # STAY
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 1  # UP / DOWN
    return 2 if dc < 0 else 3  # LEFT / RIGHT


class BlockingTeammate:
    """Walks greedily toward its own fixed goal, independent of the ego.

    At `branch_point`, checked once (the first time the teammate's
    position equals it): if the ego is NOT adjacent (Manhattan distance
    <= 1) at that exact step, the teammate is stuck (STAYs) for
    `detour_len` extra steps before resuming -- the long way. If the ego
    IS adjacent, it resumes immediately -- the short way. `took_short_path`
    records which happened, for the reward and for sanity-checking.
    """

    def __init__(self, start, goal, branch_point, detour_len=DEFAULT_DETOUR_LEN, name=None):
        self.start = start
        self.goal = goal
        self.branch_point = branch_point
        self.detour_len = detour_len
        self.name = name or f"branch_at_{branch_point}"
        self.resolved = False
        self.detour_remaining = 0
        self.took_short_path = True

    def reset(self):
        self.resolved = False
        self.detour_remaining = 0
        self.took_short_path = True

    def act(self, teammate_pos, ego_pos, rng):
        if teammate_pos == self.branch_point and not self.resolved:
            self.resolved = True
            dist = abs(teammate_pos[0] - ego_pos[0]) + abs(teammate_pos[1] - ego_pos[1])
            if dist > 1:
                self.detour_remaining = self.detour_len
                self.took_short_path = False

        if self.detour_remaining > 0:
            self.detour_remaining -= 1
            return 4  # STAY -- stuck, taking the long way

        return _greedy_action_towards(teammate_pos, self.goal)


# Three scripted teammates, each with its own start/goal, whose branch point
# falls early / mid / late along its (otherwise fixed, ~8-step) route -- so
# the ego needs a different anticipatory positioning/timing strategy for
# each, not one fixed reflex.
TEAMMATES_BLOCKING = {
    # Branch points below are NOT hand-traced -- the greedy policy zigzags
    # between row/column moves whenever the remaining gaps are close, so an
    # "elbow" cell chosen by eye is often not even on the real path (a bug
    # caught during testing: see the retained comment in git history).
    # Instead each branch_point was taken directly from simulating
    # _greedy_action_towards() start->goal, and chosen to be reachable in
    # time from EGO_START (distance to branch_point <= its step index + 1).

    # real route: (0,4)->(1,4)->(1,3)->(2,3)->(2,2)->(3,2)->(3,1)->(4,1)->(4,0)
    # branch point (1,3) hit at step 2 (of 8) -- early.
    "early_branch": BlockingTeammate(start=(0, 4), goal=(4, 0), branch_point=(1, 3),
                                      name="early_branch"),
    # real route: (4,4)->(3,4)->(3,3)->(2,3)->(2,2)->(1,2)->(1,1)->(0,1)->(0,0)
    # branch point (1,2) hit at step 5 (of 8) -- mid.
    "mid_branch": BlockingTeammate(start=(4, 4), goal=(0, 0), branch_point=(1, 2),
                                    name="mid_branch"),
    # real route: (0,0)->(1,0)->(1,1)->(2,1)->(2,2)->(3,2)->(3,3)->(4,3)->(4,4)
    # branch point (4,3) hit at step 7 (of 8) -- late.
    "late_branch": BlockingTeammate(start=(0, 0), goal=(4, 4), branch_point=(4, 3),
                                     name="late_branch"),
}

EGO_START = (2, 4)  # chosen so all three branch points above are reachable in time

OBS_DIM = 8  # ego(2) + teammate(2) + teammate_goal(2) + branch_point(2)


class BlockingGridWorld:
    """5x5 grid where the teammate pursues its own goal independent of the
    ego. Reward depends only on how quickly the teammate arrives, and
    whether the ego got it there the short way. Call reset(), then
    step(ego_action)."""

    def __init__(self, teammate, max_steps=MAX_STEPS, seed=None):
        self.teammate = teammate
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.ego_pos = None
        self.teammate_pos = None
        self.t = 0

    def reset(self):
        self.ego_pos = EGO_START
        self.teammate_pos = self.teammate.start
        self.t = 0
        self.teammate.reset()
        return self._obs()

    def _obs(self):
        """Full observation: ego, teammate, the teammate's goal, and the
        branch point -- all fixed and known in advance, so an ego capable of
        planning has everything it needs to anticipate the branch point."""
        vals = [*self.ego_pos, *self.teammate_pos, *self.teammate.goal, *self.teammate.branch_point]
        return np.array(vals, dtype=np.float32) / (GRID_SIZE - 1)

    def step(self, ego_action):
        teammate_action = self.teammate.act(self.teammate_pos, self.ego_pos, self.rng)
        self.ego_pos = _move(self.ego_pos, ego_action)
        self.teammate_pos = _move(self.teammate_pos, teammate_action)
        self.t += 1

        arrived = self.teammate_pos == self.teammate.goal
        done = arrived or self.t >= self.max_steps

        reward = STEP_PENALTY
        if arrived and self.teammate.took_short_path:
            reward += SHORT_PATH_BONUS

        info = {
            "teammate_action": teammate_action,
            "arrived": arrived,
            "took_short_path": self.teammate.took_short_path,
            "branch_resolved": self.teammate.resolved,
        }
        return self._obs(), reward, done, info

    def render(self):
        grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        gr, gc = self.teammate.goal
        grid[gr][gc] = "G"
        br, bc = self.teammate.branch_point
        if grid[br][bc] == ".":
            grid[br][bc] = "B"
        er, ec = self.ego_pos
        tr, tc = self.teammate_pos
        if (er, ec) == (tr, tc):
            grid[er][ec] = "*"
        else:
            grid[er][ec] = "E"
            grid[tr][tc] = "T"
        return "\n".join(" ".join(row) for row in grid)


def _anticipate_policy(env):
    """Sanity-check ego policy: heads straight for the branch point and
    then stays there. Known in advance from the observation, so this is
    "cheating" on purpose -- it exists only to prove the short path CAN be
    triggered, not as a real agent policy."""
    def policy_fn(obs):
        if env.ego_pos == env.teammate.branch_point:
            return 4  # STAY -- hold position
        return _greedy_action_towards(env.ego_pos, env.teammate.branch_point)
    return policy_fn


def _chase_policy(env):
    """Sanity-check ego policy: the OLD env.py heuristic, walk toward the
    teammate's current position. NOTE: on this particular small grid and
    these three teammates, this heuristic happens to still deliver the
    short path (see the note in run_episode's docstring) -- kept here as
    an honest comparison point, not as proof the mechanic forces a
    different strategy. "ignore" below is the real negative control."""
    def policy_fn(obs):
        return _greedy_action_towards(env.ego_pos, env.teammate_pos)
    return policy_fn


def _ignore_policy(env):
    """Negative-control ego policy: never moves. Demonstrates the OTHER
    half of the branch logic -- if the ego isn't adjacent when the
    teammate reaches the branch point, the detour must trigger."""
    def policy_fn(obs):
        return 4  # STAY
    return policy_fn


def run_episode(teammate_name, policy_name, seed=0, verbose=True):
    teammate = TEAMMATES_BLOCKING[teammate_name]
    env = BlockingGridWorld(teammate=teammate, seed=seed)
    obs = env.reset()
    policy_fn = {"anticipate": _anticipate_policy, "chase": _chase_policy,
                 "ignore": _ignore_policy}[policy_name](env)

    total_reward = 0.0
    if verbose:
        print(f"\n--- teammate: {teammate_name}  policy: {policy_name} ---")
        print(f"step 0 (E={env.ego_pos}, T={env.teammate_pos}, "
              f"goal={teammate.goal}, branch={teammate.branch_point})")
        print(env.render())

    for step in range(1, env.max_steps + 1):
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if verbose:
            print(f"step {step}: E->{ACTION_NAMES[action]:5s}  "
                  f"T->{ACTION_NAMES[info['teammate_action']]:5s}  reward={reward:+.1f}  "
                  f"branch_resolved={info['branch_resolved']}  short_path={info['took_short_path']}")
            print(env.render())
        if done:
            break

    if verbose:
        status = "ARRIVED (short path)" if info["took_short_path"] and info["arrived"] else \
            "ARRIVED (long path)" if info["arrived"] else "timed out"
        print(f"episode ended ({status}), total_reward={total_reward:.1f}, steps={step}")
    return total_reward, info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teammate", choices=list(TEAMMATES_BLOCKING.keys()) + ["all"], default="all")
    parser.add_argument("--policy", choices=["anticipate", "chase", "ignore"], default="anticipate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = list(TEAMMATES_BLOCKING.keys()) if args.teammate == "all" else [args.teammate]
    for name in names:
        run_episode(name, args.policy, seed=args.seed)

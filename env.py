"""Stage 1: a tiny cooperative gridworld for ad-hoc teamwork.

Ego agent E and teammate agent T share a 5x5 grid with two goal cells.
Both agents get a shared reward when they end up on the SAME goal cell at
the same time. Which goal the teammate heads for (and how fast) depends on
its scripted policy -- this is the "teammate type" the ego agent must adapt
to.
"""

import numpy as np

GRID_SIZE = 5
GOALS = [(0, 0), (4, 4)]  # goal A, goal B

# actions: up, down, left, right, stay
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
NUM_ACTIONS = len(ACTIONS)

STEP_PENALTY = -0.1
SUCCESS_REWARD = 10.0
MAX_STEPS = 30


def _clip(pos):
    r, c = pos
    return (max(0, min(GRID_SIZE - 1, r)), max(0, min(GRID_SIZE - 1, c)))


def _move(pos, action_idx):
    dr, dc = ACTIONS[action_idx]
    return _clip((pos[0] + dr, pos[1] + dc))


def _greedy_action_towards(pos, target):
    """Return the action index that greedily reduces distance to target.

    Prefers reducing whichever axis has the larger gap so movement looks
    natural on a grid (no diagonal moves available).
    """
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    if dr == 0 and dc == 0:
        return 4  # STAY
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 1  # UP / DOWN
    return 2 if dc < 0 else 3  # LEFT / RIGHT


class ScriptedTeammate:
    """Greedily walks toward a preferred goal, optionally moving slowly.

    `goal_idx` sets which goal cell it prefers (behavioral axis: preference).
    `move_prob` is the chance it actually moves each step, otherwise it
    stays put (behavioral axis: speed).
    """

    def __init__(self, goal_idx, move_prob=1.0, name=None):
        self.goal_idx = goal_idx
        self.move_prob = move_prob
        self.name = name or f"goal{goal_idx}_p{move_prob}"

    def act(self, teammate_pos, rng):
        if rng.random() > self.move_prob:
            return 4  # STAY
        target = GOALS[self.goal_idx]
        return _greedy_action_towards(teammate_pos, target)


# Three scripted teammates differing along goal-preference and speed axes.
TEAMMATES = {
    "goal_a": ScriptedTeammate(goal_idx=0, move_prob=1.0, name="goal_a"),
    "goal_b": ScriptedTeammate(goal_idx=1, move_prob=1.0, name="goal_b"),
    "slow_goal_a": ScriptedTeammate(goal_idx=0, move_prob=0.4, name="slow_goal_a"),
}


class GridWorld:
    """5x5 cooperative gridworld. Call reset(), then step(ego_action)."""

    def __init__(self, teammate, max_steps=MAX_STEPS, seed=None):
        self.teammate = teammate
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.ego_pos = None
        self.teammate_pos = None
        self.t = 0

    def reset(self):
        self.ego_pos = (GRID_SIZE - 1, 0)
        self.teammate_pos = (0, GRID_SIZE - 1)
        self.t = 0
        return self._obs()

    def _obs(self):
        """Full observation: normalized positions of both agents + both goals."""
        vals = [
            *self.ego_pos,
            *self.teammate_pos,
            *GOALS[0],
            *GOALS[1],
        ]
        return np.array(vals, dtype=np.float32) / (GRID_SIZE - 1)

    def step(self, ego_action):
        teammate_action = self.teammate.act(self.teammate_pos, self.rng)
        self.ego_pos = _move(self.ego_pos, ego_action)
        self.teammate_pos = _move(self.teammate_pos, teammate_action)
        self.t += 1

        success = self.ego_pos == self.teammate_pos and self.ego_pos in GOALS
        reward = SUCCESS_REWARD if success else STEP_PENALTY
        done = success or self.t >= self.max_steps

        info = {"teammate_action": teammate_action, "success": success}
        return self._obs(), reward, done, info

    def render(self):
        grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for gr, gc in GOALS:
            grid[gr][gc] = "G"
        er, ec = self.ego_pos
        tr, tc = self.teammate_pos
        if (er, ec) == (tr, tc):
            grid[er][ec] = "*"  # both agents on the same cell
        else:
            grid[er][ec] = "E"
            grid[tr][tc] = "T"
        lines = [" ".join(row) for row in grid]
        return "\n".join(lines)


OBS_DIM = 8  # ego(2) + teammate(2) + goalA(2) + goalB(2)

"""Stage 1: a tiny cooperative gridworld for ad-hoc teamwork.

Ego agent E and teammate agent T share a 5x5 grid with goal cells (see
GOALS). Both agents get a shared reward when they end up on the SAME goal
cell at the same time. Which goal the teammate heads for (and how fast, and
under what conditions) depends on its scripted policy -- this is the
"teammate type" the ego agent must adapt to.

All 4 corners are goals: (0,0)=A, (4,4)=B, (4,0)=C (the ego's own spawn
point), (0,4)=D (the teammate's own spawn point). D is a degenerate target
for any teammate -- greedy-towards-self is always STAY, so a teammate
aimed at D never moves -- kept in anyway (as of Stage 7's teammate_search.py)
so the generator's corner x speed x wait-pattern grid is genuinely
data-driven rather than hand-filtered to "the corners that work out nicely".
"""

import numpy as np

GRID_SIZE = 5
GOALS = [(0, 0), (4, 4), (4, 0), (0, 4)]  # goal A, goal B, goal C, goal D

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

    def reset(self):
        pass  # stateless -- nothing to reset between episodes

    def act(self, teammate_pos, ego_pos, rng):
        if rng.random() > self.move_prob:
            return 4  # STAY
        target = GOALS[self.goal_idx]
        return _greedy_action_towards(teammate_pos, target)


class WaitThenGoTeammate:
    """Stands still until the ego agent comes within `trigger_distance`,
    then greedily walks to its preferred goal -- and keeps going even if the
    ego later moves away again.

    This is a qualitatively different behavioral axis from ScriptedTeammate:
    it isn't just "walk to a fixed point," it's conditional on the ego's
    behavior. In particular, whether a given (teammate stationary) snapshot
    means "hasn't been triggered yet" or "was triggered, but move_prob-like
    luck" is NOT recoverable from a single observation -- telling them apart
    needs history. Since the basis components in ego_agent.py are memoryless
    (obs -> Q, no history), this is meant to probe whether a fixed/adaptively
    mixed basis can still cope, or whether the missing memory shows up as
    real regret.
    """

    def __init__(self, goal_idx, trigger_distance=2, move_prob=1.0, name=None):
        self.goal_idx = goal_idx
        self.trigger_distance = trigger_distance
        self.move_prob = move_prob
        self.name = name or f"wait_then_goal{goal_idx}"
        self.triggered = False

    def reset(self):
        self.triggered = False

    def act(self, teammate_pos, ego_pos, rng):
        if not self.triggered:
            dist = abs(teammate_pos[0] - ego_pos[0]) + abs(teammate_pos[1] - ego_pos[1])
            if dist <= self.trigger_distance:
                self.triggered = True
            else:
                return 4  # STAY -- waiting for the ego to approach
        if rng.random() > self.move_prob:
            return 4  # STAY -- triggered, but slow (same speed axis as ScriptedTeammate)
        target = GOALS[self.goal_idx]
        return _greedy_action_towards(teammate_pos, target)


# Stage-3 training set: goal preference (A vs B) and the "waits for the ego
# to approach before committing to a goal" behavior.
TRAIN_TEAMMATES = {
    "goal_a": ScriptedTeammate(goal_idx=0, move_prob=1.0, name="goal_a"),
    "goal_b": ScriptedTeammate(goal_idx=1, move_prob=1.0, name="goal_b"),
    "wait_then_b": WaitThenGoTeammate(goal_idx=1, trigger_distance=2, name="wait_then_b"),
}

# Held out of training entirely, for the Stage 4/5 generalization check.
# slow_goal_a shares a trait with a training teammate (goal_idx=0, same
# target as goal_a) but isn't identical to anything the agent trained
# against (the move_prob=0.4 slowness is new) -- a deliberately "close but
# not seen" probe, not a wildly different teammate.
#
# goal_c is the real corner-generalization test: it heads for goal C, a
# corner neither goal_a, goal_b, nor wait_then_b ever uses -- entirely
# unseen during training, not just a speed variant.
TEST_TEAMMATES = {
    "slow_goal_a": ScriptedTeammate(goal_idx=0, move_prob=0.4, name="slow_goal_a"),
    "goal_c": ScriptedTeammate(goal_idx=2, move_prob=1.0, name="goal_c"),
}

# Convenience union, for anything that looks a teammate up by name without
# caring whether it's a training or held-out one (e.g. training a
# best-response oracle, which is always teammate-specific regardless).
TEAMMATES = {**TRAIN_TEAMMATES, **TEST_TEAMMATES}


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
        self.teammate.reset()
        return self._obs()

    def _obs(self):
        """Full observation: normalized positions of both agents + all goals."""
        vals = [*self.ego_pos, *self.teammate_pos]
        for goal in GOALS:
            vals.extend(goal)
        return np.array(vals, dtype=np.float32) / (GRID_SIZE - 1)

    def step(self, ego_action):
        teammate_action = self.teammate.act(self.teammate_pos, self.ego_pos, self.rng)
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


OBS_DIM = 4 + 2 * len(GOALS)  # ego(2) + teammate(2) + 2 per goal

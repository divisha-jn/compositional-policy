"""Stage 2: minimal ego-agent architecture.

Three modules, combined into one acting policy:
  1. TeammateEncoder  - small GRU over recent relative teammate positions
  2. Basis            - K small Q-network components
  3. Composer          - MLP(encoder output + current obs) -> softmax weights

The acting Q-function is the weighted sum of the K component Q-functions,
Q(s, a) = sum_k w_k(s, history) * Q_k(s, a).

Nothing here is trained yet -- weights are randomly initialized so we can
sanity-check shapes and see the (currently arbitrary) behavior it produces.
"""

import numpy as np
import torch
import torch.nn as nn

from env import GRID_SIZE, NUM_ACTIONS, OBS_DIM

HISTORY_LEN = 5
ENCODER_HIDDEN = 8
QNET_HIDDEN = 16
COMPOSER_HIDDEN = 16


class TeammateEncoder(nn.Module):
    """GRU over the last HISTORY_LEN relative teammate positions (dx, dy)."""

    def __init__(self, hidden_size=ENCODER_HIDDEN):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)

    def forward(self, history):
        # history: (batch, T, 2)
        _, h_n = self.gru(history)
        return h_n.squeeze(0)  # (batch, hidden_size)


class QComponent(nn.Module):
    """One basis component: a small MLP mapping observation -> Q-values."""

    def __init__(self, obs_dim=OBS_DIM, hidden=QNET_HIDDEN, num_actions=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs):
        return self.net(obs)  # (batch, num_actions)


class Composer(nn.Module):
    """Maps [encoder_output, obs] -> softmax mixture weights over K components."""

    def __init__(self, encoder_dim=ENCODER_HIDDEN, obs_dim=OBS_DIM, num_components=2,
                 hidden=COMPOSER_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoder_dim + obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_components),
        )

    def forward(self, encoder_out, obs):
        x = torch.cat([encoder_out, obs], dim=-1)
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)  # (batch, K)


class EgoAgent(nn.Module):
    """Wires encoder + basis + composer into one acting policy.

    Call reset() at the start of each episode, then act(obs) each step.
    act() maintains its own history buffer of relative teammate positions.
    """

    def __init__(self, num_components=2, obs_dim=OBS_DIM, num_actions=NUM_ACTIONS,
                 history_len=HISTORY_LEN):
        super().__init__()
        self.num_components = num_components
        self.num_actions = num_actions
        self.history_len = history_len

        self.encoder = TeammateEncoder()
        self.basis = nn.ModuleList([QComponent(obs_dim, QNET_HIDDEN, num_actions)
                                     for _ in range(num_components)])
        self.composer = Composer(encoder_dim=self.encoder.hidden_size, obs_dim=obs_dim,
                                  num_components=num_components)

        self._history = None  # deque-like list of (dx, dy), filled on reset()

    def reset(self):
        self._history = [(0.0, 0.0)] * self.history_len

    def _push_history(self, obs):
        # obs layout: ego(2), teammate(2), goalA(2), goalB(2), all in [0, 1]
        ego = obs[0:2]
        teammate = obs[2:4]
        rel = (teammate[0] - ego[0], teammate[1] - ego[1])
        self._history.append(rel)
        self._history = self._history[-self.history_len:]

    def compute(self, obs):
        """Run a forward pass for a single observation (numpy array, shape (OBS_DIM,)).

        Returns a dict with combined Q-values, per-component Q-values, and
        the composer's mixture weights -- useful for both acting and for the
        Stage 4/5 diagnostics later.
        """
        self._push_history(obs)

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, obs_dim)
        history_t = torch.as_tensor(self._history, dtype=torch.float32).unsqueeze(0)  # (1, T, 2)

        encoder_out = self.encoder(history_t)  # (1, hidden)
        weights = self.composer(encoder_out, obs_t)  # (1, K)

        q_per_component = torch.stack([comp(obs_t) for comp in self.basis], dim=1)
        # q_per_component: (1, K, num_actions)
        q_combined = (weights.unsqueeze(-1) * q_per_component).sum(dim=1)  # (1, num_actions)

        return {
            "q_combined": q_combined.squeeze(0),
            "q_per_component": q_per_component.squeeze(0),
            "weights": weights.squeeze(0),
        }

    def act(self, obs, epsilon=0.0, rng=None):
        """Pick an action for the current observation. epsilon>0 adds random
        exploration (unused for now, since nothing is trained yet)."""
        out = self.compute(obs)
        if epsilon > 0.0:
            rng = rng or np.random.default_rng()
            if rng.random() < epsilon:
                action = int(rng.integers(self.num_actions))
                return action, out
        action = int(torch.argmax(out["q_combined"]).item())
        return action, out

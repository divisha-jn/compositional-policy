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


def init_history(history_len=HISTORY_LEN):
    """A fresh, empty history buffer for the start of an episode."""
    return [(0.0, 0.0)] * history_len


def push_history(history, obs):
    """Return a new history buffer with obs's relative teammate position
    appended (oldest entry dropped). Pure -- does not mutate `history`."""
    # obs layout: ego(2), teammate(2), goalA(2), goalB(2), all in [0, 1]
    ego = obs[0:2]
    teammate = obs[2:4]
    rel = (float(teammate[0] - ego[0]), float(teammate[1] - ego[1]))
    return (history + [rel])[-len(history):]


def epsilon_greedy_action(q_values, epsilon, rng):
    """epsilon-greedy over a 1D tensor of Q-values. rng is a numpy Generator."""
    if rng.random() < epsilon:
        return int(rng.integers(q_values.shape[0]))
    return int(torch.argmax(q_values).item())


class EgoAgent(nn.Module):
    """Wires encoder + basis + composer into one acting policy.

    Stateless: the caller owns the history buffer (see init_history /
    push_history above) and passes it into compute()/act() each step. This
    keeps the module a pure function of (obs, history), which is what a
    Q-learning update needs -- it must query Q(s_t) and Q(s_{t+1}) with two
    different, precisely-known histories.
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

    def compute(self, obs, history):
        """Run a forward pass for a single (obs, history) pair.

        obs: numpy array, shape (OBS_DIM,). history: list of history_len
        (dx, dy) tuples, as produced/maintained via push_history().

        Returns a dict with combined Q-values, per-component Q-values, and
        the composer's mixture weights -- useful for both acting and for the
        Stage 4/5 diagnostics later.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, obs_dim)
        history_t = torch.as_tensor(history, dtype=torch.float32).unsqueeze(0)  # (1, T, 2)

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

    def compute_batch(self, obs_batch, history_batch):
        """Batched version of compute(), for minibatch training. obs_batch:
        tensor (B, obs_dim). history_batch: tensor (B, T, 2). Returns just
        q_combined (B, num_actions) -- all that a TD loss needs."""
        encoder_out = self.encoder(history_batch)  # (B, hidden)
        weights = self.composer(encoder_out, obs_batch)  # (B, K)
        q_per_component = torch.stack([comp(obs_batch) for comp in self.basis], dim=1)  # (B, K, A)
        return (weights.unsqueeze(-1) * q_per_component).sum(dim=1)  # (B, A)

    def act(self, obs, history, epsilon=0.0, rng=None):
        """Pick an action for the current (obs, history). epsilon>0 adds
        random exploration."""
        out = self.compute(obs, history)
        if epsilon > 0.0:
            rng = rng if rng is not None else np.random.default_rng()
            action = epsilon_greedy_action(out["q_combined"], epsilon, rng)
        else:
            action = int(torch.argmax(out["q_combined"]).item())
        return action, out

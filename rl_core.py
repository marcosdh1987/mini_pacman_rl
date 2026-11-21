"""Small and deliberately transparent grid-world RL demonstration.
No gym or external training service required.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
import random

import numpy as np
import torch
from torch import nn

# Run very small models efficiently on laptops.
torch.set_num_threads(1)

SIZE = 5
START = (0, 0)
GOAL = (4, 4)
WALLS = {(1, 1), (2, 1), (3, 3)}
GHOST = (3, 2)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROWS = ["↑", "↓", "←", "→"]
HORIZON = 30
DEFAULT_REWARDS = {"goal": 10.0, "ghost": -15.0, "step": -0.05, "wall": -0.55, "progress": 0.10}


def to_id(pos):
    return pos[0] * SIZE + pos[1]


def from_id(i):
    return (int(i) // SIZE, int(i) % SIZE)


VALID_STARTS = tuple((r, c) for r in range(SIZE) for c in range(SIZE)
                     if (r, c) not in WALLS and (r, c) not in (GOAL, GHOST))


class Maze:
    def __init__(self, seed=42, rewards=None):
        self.rewards = dict(DEFAULT_REWARDS if rewards is None else rewards)
        self.rng = random.Random(seed)
        self.pos = START
        self.steps = 0

    def reset(self, random_start=False):
        self.pos = self.rng.choice(VALID_STARTS) if random_start else START
        self.steps = 0
        return to_id(self.pos)

    def step(self, action):
        old = self.pos
        dr, dc = MOVES[int(action)]
        nxt = (old[0] + dr, old[1] + dc)
        self.steps += 1
        reward = self.rewards["step"]
        if not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE) or nxt in WALLS:
            nxt = old
            reward += self.rewards["wall"]
        else:
            # Tiny positive shaping on progress to the coin.
            previous_dist = abs(old[0] - GOAL[0]) + abs(old[1] - GOAL[1])
            next_dist = abs(nxt[0] - GOAL[0]) + abs(nxt[1] - GOAL[1])
            reward += self.rewards["progress"] * (previous_dist - next_dist)
        self.pos = nxt
        success = nxt == GOAL
        collision = nxt == GHOST
        if success:
            reward += self.rewards["goal"]
        if collision:
            reward += self.rewards["ghost"]
        done = bool(success or collision or self.steps >= HORIZON)
        return to_id(nxt), float(reward), done, {"success": success, "ghost": collision}


class SmallDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(SIZE * SIZE, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def onehot(ids):
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    x = np.zeros((len(ids), SIZE * SIZE), dtype=np.float32)
    x[np.arange(len(ids)), ids] = 1.0
    return torch.from_numpy(x)


def snapshot(trainer):
    if trainer.mode == "Q-learning":
        return {"episode": trainer.episode, "q": trainer.q.copy(),
                "visits": trainer.visits.copy()}
    return {"episode": trainer.episode,
            "w1": trainer.network.fc1.weight.detach().numpy().copy(),
            "b1": trainer.network.fc1.bias.detach().numpy().copy(),
            "w2": trainer.network.fc2.weight.detach().numpy().copy(),
            "b2": trainer.network.fc2.bias.detach().numpy().copy(),
            "visits": trainer.visits.copy()}


def forward_details(snap, state):
    """Forward pass of the DQN broken into pieces, for explanation.

    Returns the hidden pre-activations, the ReLU activations and the
    contribution matrix ``contrib[a, j] = W2[a, j] * h[j]`` so that
    ``Q(s, a) = sum_j contrib[a, j] + b2[a]``.
    """
    pre = snap["w1"][:, state] + snap["b1"]  # one-hot input selects a column of W1
    hidden = np.maximum(0, pre)
    contrib = snap["w2"] * hidden[None, :]
    return {"pre": pre, "hidden": hidden, "contrib": contrib,
            "q": contrib.sum(axis=1) + snap["b2"]}


def snapshot_values(snap, state):
    if "q" in snap:
        return snap["q"][state]
    x = np.zeros(SIZE * SIZE, dtype=np.float32)
    x[state] = 1.0
    hidden = np.maximum(0, snap["w1"] @ x + snap["b1"])
    return snap["w2"] @ hidden + snap["b2"]


def choose_snapshot_action(snap, state):
    return int(np.argmax(snapshot_values(snap, state)))


@dataclass
class EpisodeResult:
    reward: float
    success: bool
    steps: int


class Trainer:
    def __init__(self, mode="Q-learning", seed=42, rewards=None):
        self.rewards = dict(DEFAULT_REWARDS if rewards is None else rewards)
        self.mode = mode
        self.seed = seed
        self.episode = 0
        self.rng = random.Random(seed)
        self.environment = Maze(seed, self.rewards)
        self.q = np.zeros((SIZE * SIZE, 4), dtype=np.float32)
        torch.manual_seed(seed)
        self.network = SmallDQN()
        self.target = SmallDQN()
        self.target.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.002)
        self.memory = deque(maxlen=5000)
        self.steps = 0
        self.visits = np.zeros(SIZE * SIZE, dtype=np.int64)
        self.history = []
        self.checkpoint_every = 25
        self.live = None  # episode in progress (used by step-by-step mode)
        self.checkpoints = [snapshot(self)]

    def epsilon(self):
        return 0.04 + 0.91 * math.exp(-self.episode / 350)

    def greedy_values(self, state):
        if self.mode == "Q-learning":
            return self.q[state].copy()
        with torch.no_grad():
            return self.network(onehot([state]))[0].numpy().copy()

    def act(self, state, eps):
        return self.act_explained(state, eps)[0]

    def act_explained(self, state, eps):
        """Return (action, explored): explored=True if the move was random."""
        if self.rng.random() < eps:
            return self.rng.randrange(4), True
        return int(np.argmax(self.greedy_values(state))), False

    def parameter_vector(self):
        if self.mode == "Q-learning":
            return self.q.reshape(-1).copy()
        return torch.cat([p.detach().reshape(-1) for p in self.network.parameters()]).numpy().copy()

    def replay_update(self, gamma, detail=False):
        """One gradient step on a random minibatch of 64 stored transitions.

        Returns ``None`` while the memory is too small, otherwise a dict with
        the loss (and, if ``detail``, gradient norms and weight deltas).
        """
        if len(self.memory) < 64:
            return None
        batch = self.rng.sample(self.memory, 64)
        states, actions, rewards, next_states, dones = zip(*batch)
        values = self.network(onehot(states)).gather(
            1, torch.tensor(actions, dtype=torch.long).view(-1, 1)).squeeze(1)
        with torch.no_grad():
            target_q = self.target(onehot(next_states)).max(dim=1).values
            expected = torch.tensor(rewards, dtype=torch.float32) + \
                gamma * (1 - torch.tensor(dones, dtype=torch.float32)) * target_q
        loss = nn.functional.smooth_l1_loss(values, expected)
        self.optimizer.zero_grad()
        loss.backward()
        info = {"loss": float(loss.item())}
        if detail:
            info["grad_w1"] = float(self.network.fc1.weight.grad.norm().item())
            info["grad_w2"] = float(self.network.fc2.weight.grad.norm().item())
            before_w1 = self.network.fc1.weight.detach().numpy().copy()
            before_w2 = self.network.fc2.weight.detach().numpy().copy()
            info["batch_states"] = sorted(set(int(s) for s in states))
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.steps += 1
        info["target_sync"] = self.steps % 125 == 0
        if info["target_sync"]:
            self.target.load_state_dict(self.network.state_dict())
        if detail:
            info["delta_w1"] = self.network.fc1.weight.detach().numpy() - before_w1
            info["delta_w2"] = self.network.fc2.weight.detach().numpy() - before_w2
        return info

    # --- episode bookkeeping, shared by batch training and step-by-step mode ---
    def _begin_episode(self):
        s = self.environment.reset(random_start=True)
        self.live = {"state": s, "start": s, "total": 0.0, "t": 0, "eps": self.epsilon(),
                     "errors": [], "path": [s], "params_before": self.parameter_vector()}

    def _end_episode(self, success):
        live = self.live
        self.episode += 1
        errors = live["errors"]
        self.history.append({
            "episode": self.episode, "reward": live["total"], "success": int(success),
            "epsilon": live["eps"], "steps": live["t"],
            # Q-learning: mean |TD error|; DQN: mean minibatch loss (NaN before replay starts).
            "error": float(np.mean(errors)) if errors else float("nan"),
            # How far the learnable numbers moved during this episode (L2 norm).
            "weight_change": float(np.linalg.norm(self.parameter_vector() - live["params_before"])),
        })
        self.live = None
        if self.episode % self.checkpoint_every == 0:
            self.checkpoints.append(snapshot(self))

    def learn_step(self, alpha=0.2, gamma=0.95, detail=False):
        """Play one move of the current training episode and learn from it.

        Returns a trace dict that explains the update (used by the UI).
        """
        if self.live is None:
            self._begin_episode()
        live = self.live
        s = live["state"]
        values_before = self.greedy_values(s) if detail else None
        a, explored = self.act_explained(s, live["eps"])
        sp, reward, done, info = self.environment.step(a)
        self.visits[s] += 1
        trace = {"state": s, "action": a, "explored": explored, "eps": live["eps"],
                 "reward": reward, "next_state": sp, "done": done, "info": info,
                 "values_before": values_before, "mode": self.mode}
        if self.mode == "Q-learning":
            old = float(self.q[s, a])
            next_max = 0.0 if done else float(np.max(self.q[sp]))
            target = reward + gamma * next_max
            td = target - old
            self.q[s, a] += alpha * td
            live["errors"].append(abs(td))
            trace.update(old=old, next_max=next_max, target=target, td=td,
                         new=float(self.q[s, a]), alpha=alpha, gamma=gamma)
        else:
            if detail:
                with torch.no_grad():
                    next_max = 0.0 if done else float(self.target(onehot([sp])).max().item())
                trace.update(old=float(values_before[a]), next_max=next_max,
                             target=reward + gamma * next_max, gamma=gamma)
            self.memory.append((s, a, reward, sp, float(done)))
            upd = self.replay_update(gamma, detail=detail)
            if upd is not None:
                live["errors"].append(upd["loss"])
            trace.update(update=upd, memory=len(self.memory))
            if detail:
                trace["new"] = float(self.greedy_values(s)[a])
        if detail:
            trace["values_after"] = self.greedy_values(s)
        live["total"] += reward
        live["t"] += 1
        live["state"] = sp
        live["path"].append(sp)
        trace["episode_total"] = live["total"]
        trace["episode_step"] = live["t"]
        trace["path"] = list(live["path"])
        if done:
            self._end_episode(info["success"])
            trace["episode_finished"] = True
        return trace

    def train(self, episodes=100, alpha=0.2, gamma=0.95, checkpoint_every=25):
        """Train ``episodes`` full episodes (an unfinished step-by-step episode
        is completed first and counts as one of them)."""
        self.checkpoint_every = checkpoint_every
        for _ in range(episodes):
            while True:
                trace = self.learn_step(alpha, gamma)
                if trace["done"]:
                    break
        if self.checkpoints[-1]["episode"] != self.episode:
            self.checkpoints.append(snapshot(self))


def evaluate_snapshot(snap, starts=VALID_STARTS):
    wins = 0
    travel = []
    ghosts = 0
    for initial in starts:
        e = Maze()
        e.pos = initial
        for _ in range(HORIZON):
            sp, reward, done, info = e.step(choose_snapshot_action(snap, to_id(e.pos)))
            if done:
                wins += int(info["success"])
                ghosts += int(info["ghost"])
                if info["success"]:
                    travel.append(e.steps)
                break
    return {"success_rate": wins / len(starts), "wins": wins,
            "attempts": len(starts), "ghosts": ghosts,
            "mean_steps": float(np.mean(travel)) if travel else None}


def evaluate_random(seed=123, rounds=300):
    wins = 0
    rng = random.Random(seed)
    e = Maze(seed)
    for _ in range(rounds):
        e.reset(random_start=True)
        for _ in range(HORIZON):
            _, _, done, info = e.step(rng.randrange(4))
            if done:
                wins += int(info["success"])
                break
    return wins / rounds

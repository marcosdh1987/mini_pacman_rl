"""Tests for the configurable, cyclic ghost movement."""
from __future__ import annotations

import pytest

from rl_core import (
    GHOST, GHOST_PATTERNS, N_CELLS, START, Maze, Trainer, decode_state, encode_state,
    evaluate_snapshot, free_cells, parse_route,
)


@pytest.mark.parametrize("name", list(GHOST_PATTERNS))
def test_builtin_patterns_are_valid_cycles(name):
    ghost = GHOST_PATTERNS[name]
    assert ghost.position(0) == ghost.position(ghost.period)


def test_static_pattern_matches_the_original_ghost():
    ghost = GHOST_PATTERNS["static"]
    assert ghost.is_static and ghost.route == (GHOST,)


def test_state_encoding_round_trip():
    for phase in range(4):
        for cell in [(0, 0), (2, 3), (4, 4)]:
            assert decode_state(encode_state(cell, phase)) == (cell, phase)
    assert encode_state((0, 0), 1) == N_CELLS


def test_ghost_advances_one_cell_per_step_and_wraps():
    ghost = GHOST_PATTERNS["vertical_patrol"]
    env = Maze(ghost=ghost)
    env.reset()
    seen = [env.ghost_pos]
    for _ in range(ghost.period):
        env.step(2)  # bump into the left wall: the agent stays at START
        seen.append(env.ghost_pos)
    assert seen == list(ghost.route) + [ghost.route[0]]


def test_agent_walking_into_the_ghost_path_collides():
    ghost = parse_route("0,2 0,1")
    env = Maze(ghost=ghost)
    env.reset()
    assert env.pos == START
    _, _, done, info = env.step(3)  # (0,0) -> (0,1), where the ghost is heading
    assert done and info["ghost"]


def test_agent_cannot_swap_places_through_the_ghost():
    ghost = parse_route("1,2 0,2")
    env = Maze(ghost=ghost)
    env.pos, env.phase = (0, 2), 0
    _, _, done, info = env.step(1)  # agent moves down while the ghost moves up
    assert done and info["ghost"]


def test_random_starts_avoid_an_unavoidable_first_hit():
    ghost = GHOST_PATTERNS["loop"]
    for phase in range(ghost.period):
        cells = free_cells(ghost, phase)
        assert ghost.position(phase) not in cells
        assert ghost.position(phase + 1) not in cells


@pytest.mark.parametrize("text", ["", "4,4", "0,0 1,0", "1,1", "2,2 2,4", "9,9", "a,b", "1"])
def test_parse_route_rejects_invalid_routes(text):
    with pytest.raises(ValueError):
        parse_route(text)


@pytest.mark.parametrize("mode", ["Q-learning", "DQN (PyTorch)"])
def test_state_space_grows_with_the_cycle_length(mode):
    ghost = GHOST_PATTERNS["loop"]
    trainer = Trainer(mode, seed=1, ghost=ghost)
    assert trainer.n_states == N_CELLS * ghost.period
    assert trainer.greedy_values(encode_state((0, 0), ghost.period - 1)).shape == (4,)
    trainer.train(episodes=5)
    assert trainer.episode == 5


@pytest.mark.parametrize("name", ["vertical_patrol", "horizontal_patrol", "loop"])
def test_q_learning_learns_a_moving_pattern(name):
    trainer = Trainer("Q-learning", seed=1, ghost=GHOST_PATTERNS[name])
    untrained = evaluate_snapshot(trainer.checkpoints[0])
    trainer.train(episodes=1500)
    trained = evaluate_snapshot(trainer.checkpoints[-1])
    assert trained["success_rate"] >= 0.9 > untrained["success_rate"]

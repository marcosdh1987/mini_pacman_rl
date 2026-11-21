# Moving ghost: configurable fixed patterns

## Goal

The ghost used to sit on `(3,2)`. It can now follow a **fixed, cyclic route**,
chosen from a selector in the sidebar, so the agent has something to learn
beyond "avoid one cell": the timing of the ghost.

## Model (`rl_core.py`)

- `GhostPattern(route)`: immutable cyclic route. The ghost is at
  `route[phase]` and advances one entry per environment step, wrapping around.
  A one-cell route is the static ghost.
- `GHOST_PATTERNS`: built-in presets.

  | key | route | period |
  |---|---|---|
  | `static` | `(3,2)` | 1 |
  | `vertical_patrol` | `(1,4) (2,4) (3,4) (2,4)` | 4 |
  | `horizontal_patrol` | `(3,0) (3,1) (3,2) (3,1)` | 4 |
  | `loop` | `(1,2) (1,3) (1,4) (2,4) (2,3) (2,2)` | 6 |

- `parse_route("r,c r,c ...")` builds a custom pattern. `validate_route`
  rejects: empty or longer than `MAX_ROUTE_LENGTH` (12), out of the grid,
  walls, the coin `(4,4)`, the start `(0,0)`, and non-adjacent consecutive
  cells (including last → first, because the route is cyclic).

### State: why the pattern is learnable

The agent's state is `(agent cell, ghost phase)`, encoded as
`phase * 25 + cell` (`encode_state` / `decode_state`). Without the phase the
environment would not be Markovian for the agent: the same cell would be safe
or deadly depending on an invisible clock, and the agent could not learn the
pattern. With it:

- Q-learning: the table has `25 × period` rows.
- DQN: the input is the one-hot cell (25) concatenated with the one-hot phase
  (`period`), so `W1` has `25 + period` columns (`f0`, `f1`, …).

For the static pattern (`period = 1`) everything is identical to the original
implementation: same state ids, same network size and the same RNG usage, so
training with a given seed reproduces the previous results exactly.

### Dynamics of one step

1. The agent moves (walls/borders block it as before).
2. The ghost advances to the next phase.
3. Collision if the agent ends on the ghost's cell **before or after** its
   move — the agent cannot swap places through the ghost.
4. Reaching the coin only counts if there was no collision.

### Episode starts

- Training: random phase and random free cell. Free cells exclude walls, the
  coin, the ghost and the cell the ghost moves into next (an unavoidable hit).
  The static pattern skips the phase draw to keep the original RNG sequence.
- Evaluation and the visual game: the ghost starts at phase `f0`, so results
  are deterministic and comparable across checkpoints.

## UI (`app.py`)

- Sidebar *Movimiento del fantasma*: pattern selector, custom route input
  with validation errors, a mini board showing the route (dashed cells) and
  the resulting state-space size. Changing it shows the usual "restart
  training" warning; the trainer keeps its own pattern until restarted.
- All boards draw the ghost at its current position and the route outline.
- Checkpoint section: a *Fase del fantasma* slider (only for moving ghosts)
  selects which slice of the state space the policy, value, visits, Q-table
  and DQN forward-pass views show.

## Tests

`tests/test_ghost_patterns.py` (`make test-unit`) covers route validation,
state encoding, ghost stepping and wrap-around, collision rules, safe starts,
state-space size for both methods, and that Q-learning learns each moving
preset (≥90 % greedy success after 1500 episodes). Equivalence of the static
pattern with the original code was checked by training both versions with the
same seed and comparing histories and parameters.

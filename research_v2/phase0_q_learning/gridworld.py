"""A tiny deterministic environment for learning the MDP interface."""

from __future__ import annotations

from dataclasses import dataclass


UP, RIGHT, DOWN, LEFT = range(4)
ACTION_NAMES = ("up", "right", "down", "left")
ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class GridConfig:
    size: int = 4
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (3, 3)
    traps: frozenset[tuple[int, int]] = frozenset({(1, 1), (2, 2)})
    step_reward: float = -0.02
    goal_reward: float = 1.0
    trap_reward: float = -1.0


class GridWorld:
    """4x4 grid: reach the goal while avoiding terminal trap cells."""

    def __init__(self, config: GridConfig | None = None):
        self.config = config or GridConfig()
        self.n_states = self.config.size**2
        self.n_actions = len(ACTION_NAMES)
        self.state = self.position_to_state(self.config.start)
        self.done = False

    def position_to_state(self, position: tuple[int, int]) -> int:
        row, col = position
        return row * self.config.size + col

    def state_to_position(self, state: int) -> tuple[int, int]:
        return divmod(state, self.config.size)

    @property
    def goal_state(self) -> int:
        return self.position_to_state(self.config.goal)

    @property
    def trap_states(self) -> frozenset[int]:
        return frozenset(self.position_to_state(p) for p in self.config.traps)

    def reset(self) -> int:
        self.state = self.position_to_state(self.config.start)
        self.done = False
        return self.state

    def transition(self, state: int, action: int) -> tuple[int, float, bool]:
        """Return the deterministic (next_state, reward, done) transition."""
        if not 0 <= state < self.n_states:
            raise ValueError(f"invalid state: {state}")
        if not 0 <= action < self.n_actions:
            raise ValueError(f"invalid action: {action}")

        row, col = self.state_to_position(state)
        dr, dc = ACTION_DELTAS[action]
        size = self.config.size
        next_position = (
            min(max(row + dr, 0), size - 1),
            min(max(col + dc, 0), size - 1),
        )
        next_state = self.position_to_state(next_position)
        if next_position == self.config.goal:
            return next_state, self.config.goal_reward, True
        if next_position in self.config.traps:
            return next_state, self.config.trap_reward, True
        return next_state, self.config.step_reward, False

    def step(self, action: int) -> tuple[int, float, bool]:
        if self.done:
            raise RuntimeError("episode is done; call reset() before step()")
        self.state, reward, self.done = self.transition(self.state, action)
        return self.state, reward, self.done

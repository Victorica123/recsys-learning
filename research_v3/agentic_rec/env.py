"""A minimal recommendation agent that learns when a costly tool is useful.

The wrapped ``RecSimEnv`` exposes 200 item actions.  This environment turns the
decision into two *meta actions*:

``BASELINE_ACTION``
    Use the existing short-sighted SASRec scorer (the highest raw score).

``TOOL_ACTION``
    Invoke a fatigue-aware reranking tool.  It discounts candidates whose genre
    repeats the current run and chooses the best adjusted click probability.

The training reward borrows one narrow idea from DeepEyes: tool use receives a
bonus only when it is both useful according to the simulator and followed by a
successful outcome.  We additionally charge every invocation, so calling the
tool unconditionally is not free.  Evaluation keeps raw user reward, net reward
after cost, and shaped training reward separate.

This is a small agentic-RL experiment over a frozen SASRec simulator.  It is not
a DeepEyes reproduction and the resulting numbers are not online evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from train_dqn_rec import SESSION, STATE_DIM


BASELINE_ACTION = 0
TOOL_ACTION = 1
N_META_ACTIONS = 2
AGENT_FEATURE_DIM = 4
AGENT_STATE_DIM = STATE_DIM + AGENT_FEATURE_DIM


@dataclass(frozen=True)
class ToolPlan:
    """The two candidate actions and their simulator click probabilities."""

    baseline_action: int
    tool_action: int
    baseline_probability: float
    tool_probability: float

    @property
    def expected_gain(self) -> float:
        return self.tool_probability - self.baseline_probability

    @property
    def changes_item(self) -> bool:
        return self.tool_action != self.baseline_action


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class AgenticRecEnv:
    """Two-action tool-selection layer over ``train_dqn_rec.RecSimEnv``."""

    def __init__(
        self,
        base_env,
        *,
        tool_cost: float = 0.03,
        tool_bonus: float = 0.05,
        min_gain: float = 0.01,
    ) -> None:
        if tool_cost < 0.0 or tool_bonus < 0.0 or min_gain < 0.0:
            raise ValueError("tool_cost, tool_bonus, and min_gain must be non-negative")
        self.base = base_env
        self.tool_cost = float(tool_cost)
        self.tool_bonus = float(tool_bonus)
        self.min_gain = float(min_gain)
        self.current_plan: ToolPlan | None = None
        self._current_state: np.ndarray | None = None

    def reset(self) -> np.ndarray:
        # RecSimEnv.reset() initializes the user/session fields.  We then score
        # once more so the state and the action plan share the same context.
        self.base.reset()
        ctx, scores = self.base._score(self.base.hist)
        base_state = self.base._state(ctx)
        self.current_plan = self._make_plan(scores)
        self._current_state = self._augment(base_state, self.current_plan)
        return self._current_state.copy()

    def _make_plan(self, scores: np.ndarray) -> ToolPlan:
        raw_probabilities = _sigmoid(scores)
        baseline_action = int(np.argmax(scores))

        fatigue = np.ones(len(scores), dtype=np.float64)
        if self.base.last_genre >= 0 and self.base.run_len > 0:
            repeated_multiplier = 0.75 ** self.base.run_len
            for action, item in enumerate(self.base.top_items):
                genre = self.base.genre.get(int(item), 0)
                if genre == self.base.last_genre:
                    fatigue[action] = repeated_multiplier

        adjusted = raw_probabilities * fatigue
        tool_action = int(np.argmax(adjusted))
        return ToolPlan(
            baseline_action=baseline_action,
            tool_action=tool_action,
            baseline_probability=float(adjusted[baseline_action]),
            tool_probability=float(adjusted[tool_action]),
        )

    def _augment(self, base_state: np.ndarray, plan: ToolPlan) -> np.ndarray:
        features = np.array(
            [
                plan.baseline_probability,
                plan.tool_probability,
                plan.expected_gain,
                min(1.0, float(self.base.run_len) / float(SESSION)),
            ],
            dtype=np.float32,
        )
        state = np.concatenate([base_state.astype(np.float32), features])
        if state.shape != (AGENT_STATE_DIM,):
            raise RuntimeError(
                f"agent state shape {state.shape} != {(AGENT_STATE_DIM,)}"
            )
        return state

    def step(self, meta_action: int):
        if meta_action not in (BASELINE_ACTION, TOOL_ACTION):
            raise ValueError(f"meta_action must be 0 or 1, got {meta_action}")
        if self.current_plan is None or self._current_state is None:
            raise RuntimeError("reset() must be called before step()")

        plan = self.current_plan
        tool_used = meta_action == TOOL_ACTION
        item_action = plan.tool_action if tool_used else plan.baseline_action
        predicted_probability = (
            plan.tool_probability if tool_used else plan.baseline_probability
        )

        item = int(self.base.top_items[item_action])
        genre = self.base.genre.get(item, 0)
        if genre == self.base.last_genre:
            self.base.run_len += 1
        else:
            self.base.last_genre, self.base.run_len = genre, 1

        like = self.base.rng.random() < predicted_probability
        self.base.recent.append(item)
        if like:
            self.base.hist.append(item)
            self.base.consec_dislike = 0
            user_reward = 0.9
        else:
            self.base.consec_dislike += 1
            user_reward = -0.1

        done = (
            len(self.base.recent) >= SESSION
            or self.base.consec_dislike >= 3
        )
        beneficial_tool = bool(
            tool_used
            and plan.changes_item
            and plan.expected_gain >= self.min_gain
        )
        bonus_earned = bool(beneficial_tool and like)
        cost_paid = self.tool_cost if tool_used else 0.0
        bonus_paid = self.tool_bonus if bonus_earned else 0.0
        net_reward = user_reward - cost_paid
        objective_reward = net_reward + bonus_paid

        info = {
            "p": predicted_probability,
            "genre": genre,
            "like": like,
            "item_action": item_action,
            "movie_item_id": item,
            "meta_action": meta_action,
            "tool_used": tool_used,
            "tool_changed_item": bool(tool_used and plan.changes_item),
            "beneficial_tool": beneficial_tool,
            "bonus_earned": bonus_earned,
            "expected_gain": plan.expected_gain,
            "baseline_probability": plan.baseline_probability,
            "tool_probability": plan.tool_probability,
            "user_reward": user_reward,
            "tool_cost_paid": cost_paid,
            "tool_bonus_paid": bonus_paid,
            "net_reward": net_reward,
            "objective_reward": objective_reward,
        }

        if done:
            self.current_plan = None
            self._current_state = None
            next_state = np.zeros(AGENT_STATE_DIM, dtype=np.float32)
        else:
            next_ctx, next_scores = self.base._score(self.base.hist)
            next_base_state = self.base._state(next_ctx)
            self.current_plan = self._make_plan(next_scores)
            self._current_state = self._augment(next_base_state, self.current_plan)
            next_state = self._current_state.copy()
        return next_state, objective_reward, done, info

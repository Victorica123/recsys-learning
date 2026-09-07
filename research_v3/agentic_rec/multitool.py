"""Budgeted multi-tool recommendation trajectories over the frozen simulator.

The previous experiment chose between ``baseline`` and one reranking tool in a
single step.  This module turns each recommendation into a short agent loop:

    inspect state -> call zero/one/two tools -> STOP_AND_SERVE -> user feedback

Three tools expose complementary information at different costs.  Action masks,
session budgets, repeated-call errors and a forced-stop recovery make invalid
tool use observable rather than silently ignored.  This remains a controlled
RecSim experiment, not an LLM agent or online evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import GENRES
from train_dqn_rec import SESSION, STATE_DIM

from .policy import clipped_surrogate_loss, group_relative_advantages


STOP_AND_SERVE = 0
FATIGUE_TOOL = 1
DIVERSITY_TOOL = 2
PREFERENCE_TOOL = 3
N_MULTITOOL_ACTIONS = 4
TOOL_ACTIONS = (FATIGUE_TOOL, DIVERSITY_TOOL, PREFERENCE_TOOL)
TOOL_NAMES = {
    FATIGUE_TOOL: "fatigue_scan",
    DIVERSITY_TOOL: "diversity_search",
    PREFERENCE_TOOL: "preference_probe",
}

# baseline p, best revealed gain, fatigue, recent diversity, budget, slots,
# decision progress, three called flags, three tool gains, revealed genre one-hot.
MULTITOOL_FEATURE_DIM = 7 + 3 + 3 + len(GENRES)
MULTITOOL_STATE_DIM = STATE_DIM + MULTITOOL_FEATURE_DIM


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class ToolProposal:
    tool_action: int
    item_action: int
    estimated_probability: float
    estimated_gain: float


class MultiToolRecEnv:
    """A multi-action tool loop wrapped around one RecSim user session."""

    def __init__(
        self,
        base_env,
        *,
        session_budget: float = 0.60,
        fatigue_cost: float = 0.05,
        diversity_cost: float = 0.03,
        preference_cost: float = 0.12,
        max_tools_per_decision: int = 2,
        invalid_action_penalty: float = 0.05,
        preferred_genre_multiplier: float = 1.12,
        other_genre_multiplier: float = 0.97,
    ) -> None:
        costs = {
            FATIGUE_TOOL: float(fatigue_cost),
            DIVERSITY_TOOL: float(diversity_cost),
            PREFERENCE_TOOL: float(preference_cost),
        }
        if session_budget < 0.0 or any(value < 0.0 for value in costs.values()):
            raise ValueError("budget and tool costs must be non-negative")
        if max_tools_per_decision < 1:
            raise ValueError("max_tools_per_decision must be positive")
        if invalid_action_penalty < 0.0:
            raise ValueError("invalid_action_penalty must be non-negative")
        if preferred_genre_multiplier <= 0.0 or other_genre_multiplier <= 0.0:
            raise ValueError("genre multipliers must be positive")
        self.base = base_env
        self.initial_budget = float(session_budget)
        self.remaining_budget = float(session_budget)
        self.tool_costs = costs
        self.max_tools_per_decision = int(max_tools_per_decision)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.preferred_genre_multiplier = float(preferred_genre_multiplier)
        self.other_genre_multiplier = float(other_genre_multiplier)

        self.hidden_preferred_genre = 0
        self.called_tools: set[int] = set()
        self.tool_gains = np.full(3, -1.0, dtype=np.float32)
        self.proposals: dict[int, ToolProposal] = {}
        self.baseline_action = 0
        self.baseline_raw_probability = 0.0
        self._scores = np.zeros(1, dtype=np.float32)
        self._base_state: np.ndarray | None = None
        self._current_state: np.ndarray | None = None
        self._force_serve = False
        self.total_tool_calls = 0
        self.total_invalid_actions = 0
        self.total_decisions = 0
        self.total_multi_tool_decisions = 0

    def reset(self) -> np.ndarray:
        self.base.reset()
        history_genres = [
            self.base.genre.get(int(item), 0) for item in self.base.hist]
        self.hidden_preferred_genre = int(self.base.rng.choice(history_genres))
        self.remaining_budget = self.initial_budget
        self.total_tool_calls = 0
        self.total_invalid_actions = 0
        self.total_decisions = 0
        self.total_multi_tool_decisions = 0
        self._start_decision()
        return self._current_state.copy()

    def _start_decision(self) -> None:
        context, scores = self.base._score(self.base.hist)
        self._scores = np.asarray(scores, dtype=np.float64)
        probabilities = _sigmoid(self._scores)
        self.baseline_action = int(np.argmax(self._scores))
        self.baseline_raw_probability = float(probabilities[self.baseline_action])
        self.called_tools = set()
        self.tool_gains = np.full(3, -1.0, dtype=np.float32)
        self.proposals = {}
        self._force_serve = False
        self._base_state = self.base._state(context).astype(np.float32)
        self._current_state = self._augment()

    def _prospective_fatigue(self, item_action: int) -> float:
        item = int(self.base.top_items[item_action])
        genre = self.base.genre.get(item, 0)
        if genre == self.base.last_genre and self.base.run_len > 0:
            return float(0.75 ** self.base.run_len)
        return 1.0

    def _preference_multiplier(self, item_action: int) -> float:
        item = int(self.base.top_items[item_action])
        genre = self.base.genre.get(item, 0)
        return (
            self.preferred_genre_multiplier
            if genre == self.hidden_preferred_genre
            else self.other_genre_multiplier)

    def true_probability(self, item_action: int) -> float:
        raw = float(_sigmoid(self._scores)[item_action])
        return float(np.clip(
            raw * self._prospective_fatigue(item_action)
            * self._preference_multiplier(item_action), 0.0, 0.995))

    def _tool_proposal(self, action: int) -> ToolProposal:
        raw = _sigmoid(self._scores)
        fatigue = np.asarray([
            self._prospective_fatigue(index) for index in range(len(raw))])
        if action == FATIGUE_TOOL:
            estimate = raw * fatigue
        elif action == DIVERSITY_TOOL:
            recent_genres = {
                self.base.genre.get(int(item), 0) for item in self.base.recent[-5:]}
            eligible = np.asarray([
                self.base.genre.get(int(item), 0) not in recent_genres
                for item in self.base.top_items], dtype=bool)
            estimate = raw * fatigue
            if eligible.any():
                estimate = np.where(eligible, estimate, -np.inf)
        elif action == PREFERENCE_TOOL:
            estimate = np.asarray([
                self.true_probability(index) for index in range(len(raw))])
        else:
            raise ValueError(f"unknown tool action {action}")
        item_action = int(np.argmax(estimate))
        if action == PREFERENCE_TOOL:
            baseline_estimate = self.true_probability(self.baseline_action)
        else:
            baseline_estimate = float(
                raw[self.baseline_action]
                * self._prospective_fatigue(self.baseline_action))
        selected_estimate = float(estimate[item_action])
        return ToolProposal(
            tool_action=action,
            item_action=item_action,
            estimated_probability=selected_estimate,
            estimated_gain=selected_estimate - baseline_estimate,
        )

    def _best_proposal(self) -> ToolProposal | None:
        if not self.proposals:
            return None
        return max(
            self.proposals.values(),
            key=lambda proposal: (proposal.estimated_gain, -proposal.tool_action))

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(N_MULTITOOL_ACTIONS, dtype=bool)
        mask[STOP_AND_SERVE] = True
        if self._force_serve or len(self.called_tools) >= self.max_tools_per_decision:
            return mask
        for action in TOOL_ACTIONS:
            mask[action] = (
                action not in self.called_tools
                and self.remaining_budget + 1e-12 >= self.tool_costs[action])
        return mask

    def _augment(self) -> np.ndarray:
        best = self._best_proposal()
        best_gain = max(0.0, best.estimated_gain) if best is not None else 0.0
        recent_genres = {
            self.base.genre.get(int(item), 0) for item in self.base.recent[-5:]}
        preference = np.zeros(len(GENRES), dtype=np.float32)
        if PREFERENCE_TOOL in self.called_tools:
            preference[self.hidden_preferred_genre] = 1.0
        called = np.asarray([
            float(action in self.called_tools) for action in TOOL_ACTIONS],
            dtype=np.float32)
        features = np.concatenate([
            np.asarray([
                self.baseline_raw_probability,
                best_gain,
                min(1.0, float(self.base.run_len) / 5.0),
                len(recent_genres) / 5.0,
                (self.remaining_budget / self.initial_budget
                 if self.initial_budget > 0.0 else 0.0),
                (self.max_tools_per_decision - len(self.called_tools))
                / self.max_tools_per_decision,
                len(self.called_tools) / self.max_tools_per_decision,
            ], dtype=np.float32),
            called,
            self.tool_gains,
            preference,
        ])
        state = np.concatenate([self._base_state, features]).astype(np.float32)
        if state.shape != (MULTITOOL_STATE_DIM,):
            raise RuntimeError(
                f"multi-tool state shape {state.shape} != {(MULTITOOL_STATE_DIM,)}")
        return state

    def oracle_action(self) -> int:
        """One-step expected-net-reward ceiling with access to hidden preference."""
        mask = self.action_mask()
        current = self.baseline_action
        best = self._best_proposal()
        if best is not None and best.estimated_gain > 0.0:
            current = best.item_action
        current_probability = self.true_probability(current)
        best_action = STOP_AND_SERVE
        best_increment = 0.0
        for action in TOOL_ACTIONS:
            if not mask[action]:
                continue
            proposal = self._tool_proposal(action)
            increment = (
                self.true_probability(proposal.item_action)
                - current_probability - self.tool_costs[action])
            if increment > best_increment:
                best_increment = increment
                best_action = action
        return best_action

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if not 0 <= int(action) < N_MULTITOOL_ACTIONS:
            raise ValueError(f"action must be in [0, {N_MULTITOOL_ACTIONS})")
        if self._current_state is None:
            raise RuntimeError("reset() must be called before step()")
        action = int(action)
        valid = bool(self.action_mask()[action])
        if not valid:
            self.total_invalid_actions += 1
            self._force_serve = True
            self._current_state = self._augment()
            return self._current_state.copy(), -self.invalid_action_penalty, False, {
                "decision_complete": False,
                "tool_used": False,
                "tool_action": action,
                "tool_name": TOOL_NAMES.get(action, "invalid_stop"),
                "invalid_action": True,
                "forced_serve_next": True,
                "tool_cost_paid": 0.0,
                "invalid_penalty": self.invalid_action_penalty,
                "user_reward": 0.0,
                "net_reward": -self.invalid_action_penalty,
            }

        if action in TOOL_ACTIONS:
            proposal = self._tool_proposal(action)
            cost = self.tool_costs[action]
            self.remaining_budget = max(0.0, self.remaining_budget - cost)
            self.called_tools.add(action)
            self.tool_gains[action - 1] = float(proposal.estimated_gain)
            self.proposals[action] = proposal
            self.total_tool_calls += 1
            self._current_state = self._augment()
            return self._current_state.copy(), -cost, False, {
                "decision_complete": False,
                "tool_used": True,
                "tool_action": action,
                "tool_name": TOOL_NAMES[action],
                "invalid_action": False,
                "forced_serve_next": False,
                "tool_cost_paid": cost,
                "invalid_penalty": 0.0,
                "estimated_gain": proposal.estimated_gain,
                "user_reward": 0.0,
                "net_reward": -cost,
            }

        best = self._best_proposal()
        item_action = (
            best.item_action if best is not None and best.estimated_gain > 0.0
            else self.baseline_action)
        source = (
            TOOL_NAMES[best.tool_action]
            if best is not None and best.estimated_gain > 0.0 else "baseline")
        item = int(self.base.top_items[item_action])
        genre = self.base.genre.get(item, 0)
        probability = self.true_probability(item_action)
        if genre == self.base.last_genre:
            self.base.run_len += 1
        else:
            self.base.last_genre, self.base.run_len = genre, 1
        like = self.base.rng.random() < probability
        self.base.recent.append(item)
        if like:
            self.base.hist.append(item)
            self.base.consec_dislike = 0
            user_reward = 0.9
        else:
            self.base.consec_dislike += 1
            user_reward = -0.1
        done = (
            len(self.base.recent) >= SESSION or self.base.consec_dislike >= 3)
        decision_calls = len(self.called_tools)
        self.total_decisions += 1
        self.total_multi_tool_decisions += int(decision_calls >= 2)
        info = {
            "decision_complete": True,
            "tool_used": False,
            "tool_action": STOP_AND_SERVE,
            "tool_name": "stop_and_serve",
            "invalid_action": False,
            "forced_serve_next": False,
            "tool_cost_paid": 0.0,
            "invalid_penalty": 0.0,
            "user_reward": user_reward,
            "net_reward": user_reward,
            "like": like,
            "p": probability,
            "genre": genre,
            "movie_item_id": item,
            "item_action": item_action,
            "selected_source": source,
            "decision_tool_calls": decision_calls,
            "multi_tool_decision": decision_calls >= 2,
            "remaining_budget": self.remaining_budget,
        }
        if done:
            self._current_state = None
            next_state = np.zeros(MULTITOOL_STATE_DIM, dtype=np.float32)
        else:
            self._start_decision()
            next_state = self._current_state.copy()
        return next_state, user_reward, done, info


class MultiToolPolicy(nn.Module):
    def __init__(self, feature_dim: int = MULTITOOL_FEATURE_DIM) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, 48),
            nn.LayerNorm(48),
            nn.Tanh(),
            nn.Linear(48, 24),
            nn.Tanh(),
            nn.Linear(24, N_MULTITOOL_ACTIONS),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] == MULTITOOL_STATE_DIM:
            states = states[..., -self.feature_dim:]
        if states.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected {self.feature_dim} decision features, got {states.shape[-1]}")
        return self.network(states)


def masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if logits.shape != masks.shape:
        raise ValueError("logits and action masks must have identical shape")
    if not torch.all(masks.any(dim=-1)):
        raise ValueError("every state must allow at least one action")
    return logits.masked_fill(~masks, torch.finfo(logits.dtype).min)


@dataclass
class MultiToolTrajectory:
    states: list[np.ndarray]
    masks: list[np.ndarray]
    actions: list[int]
    log_probs: list[float]
    net_return: float
    user_return: float
    tool_calls: int
    recommendations: int
    invalid_actions: int


def rollout_multitool(
    policy: MultiToolPolicy,
    env: MultiToolRecEnv,
    *,
    device: str,
    sample: bool,
) -> MultiToolTrajectory:
    state = env.reset()
    states: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    net_return = 0.0
    user_return = 0.0
    tool_calls = 0
    recommendations = 0
    invalid_actions = 0
    while True:
        mask = env.action_mask()
        tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = masked_logits(policy(tensor), mask_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            action_tensor = distribution.sample() if sample else logits.argmax(dim=1)
            action = int(action_tensor.item())
            log_probability = float(distribution.log_prob(action_tensor).item())
        next_state, reward, done, info = env.step(action)
        states.append(state)
        masks.append(mask)
        actions.append(action)
        log_probs.append(log_probability)
        net_return += float(reward)
        user_return += float(info["user_reward"])
        tool_calls += int(info["tool_used"])
        recommendations += int(info["decision_complete"])
        invalid_actions += int(info["invalid_action"])
        state = next_state
        if done:
            break
    return MultiToolTrajectory(
        states, masks, actions, log_probs, net_return, user_return,
        tool_calls, recommendations, invalid_actions)


def train_multitool_policy(
    *,
    make_env: Callable[[int], MultiToolRecEnv],
    seed: int,
    updates: int,
    groups_per_update: int,
    group_size: int,
    lr: float,
    entropy_coef: float,
    clip_eps: float,
    kl_coef: float,
    epochs: int,
    device: str,
    verbose: bool = True,
) -> tuple[MultiToolPolicy, list[dict]]:
    if group_size < 2 or updates < 1 or groups_per_update < 1 or epochs < 1:
        raise ValueError("training counts must be positive and group_size >= 2")
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = MultiToolPolicy().to(device)
    reference = MultiToolPolicy().to(device)
    reference.load_state_dict(policy.state_dict())
    reference.eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history: list[dict] = []

    for update in range(1, updates + 1):
        weighted: list[tuple[MultiToolTrajectory, float]] = []
        all_trajectories: list[MultiToolTrajectory] = []
        for group in range(groups_per_update):
            context_seed = seed + update * 100_000 + group
            trajectories = [
                rollout_multitool(
                    policy, make_env(context_seed), device=device, sample=True)
                for _ in range(group_size)
            ]
            advantages = group_relative_advantages(
                [trajectory.net_return for trajectory in trajectories])
            weighted.extend(zip(trajectories, advantages.tolist()))
            all_trajectories.extend(trajectories)

        for _ in range(epochs):
            losses = []
            entropies = []
            kls = []
            for trajectory, advantage in weighted:
                states = torch.from_numpy(np.stack(trajectory.states)).float().to(device)
                masks = torch.from_numpy(np.stack(trajectory.masks)).bool().to(device)
                actions = torch.tensor(trajectory.actions, dtype=torch.long, device=device)
                logits = masked_logits(policy(states), masks)
                distribution = torch.distributions.Categorical(logits=logits)
                logp_new = distribution.log_prob(actions)
                logp_old = torch.tensor(
                    trajectory.log_probs, dtype=torch.float32, device=device)
                advantage_tensor = torch.full_like(logp_new, float(advantage))
                losses.append(clipped_surrogate_loss(
                    logp_new, logp_old, advantage_tensor, clip_eps))
                entropies.append(distribution.entropy().mean())
                with torch.no_grad():
                    reference_logits = masked_logits(reference(states), masks)
                policy_probs = F.softmax(logits, dim=-1)
                kls.append((policy_probs * (
                    F.log_softmax(logits, dim=-1)
                    - F.log_softmax(reference_logits, dim=-1))).sum(-1).mean())
            policy_loss = torch.stack(losses).mean()
            entropy = torch.stack(entropies).mean()
            kl = torch.stack(kls).mean()
            loss = policy_loss + kl_coef * kl - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()

        calls = sum(item.tool_calls for item in all_trajectories)
        decisions = sum(item.recommendations for item in all_trajectories)
        row = {
            "update": update,
            "mean_net_return": round(float(np.mean([
                item.net_return for item in all_trajectories])), 6),
            "mean_user_return": round(float(np.mean([
                item.user_return for item in all_trajectories])), 6),
            "tool_calls_per_decision": round(calls / max(1, decisions), 6),
            "entropy": round(float(entropy.item()), 6),
            "kl": round(float(kl.item()), 6),
            "loss": round(float(loss.item()), 6),
        }
        history.append(row)
        if verbose and (update == 1 or update == updates
                        or update % max(1, updates // 4) == 0):
            print(
                f"    update={update}/{updates} net={row['mean_net_return']:.3f} "
                f"tools/decision={row['tool_calls_per_decision']:.2f} "
                f"entropy={row['entropy']:.3f}", flush=True)
    return policy, history


def learned_multitool_action(
    policy: MultiToolPolicy,
    device: str,
    *,
    sample: bool,
    seed: int,
):
    rng = np.random.default_rng(seed)

    def act(state: np.ndarray, mask: np.ndarray, env: MultiToolRecEnv) -> int:
        del env
        with torch.no_grad():
            states = torch.from_numpy(state).float().unsqueeze(0).to(device)
            masks = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
            logits = masked_logits(policy(states), masks)
            if not sample:
                return int(logits.argmax(dim=1).item())
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        return int(rng.choice(len(probabilities), p=probabilities))

    return act


def budgeted_rule_action(
    state: np.ndarray, mask: np.ndarray, env: MultiToolRecEnv,
) -> int:
    del state
    best = env._best_proposal()
    if best is not None and best.estimated_gain >= 0.02:
        return STOP_AND_SERVE
    if (env.base.run_len >= 2 and mask[FATIGUE_TOOL]
            and FATIGUE_TOOL not in env.called_tools):
        return FATIGUE_TOOL
    if (env.baseline_raw_probability < 0.68 and mask[PREFERENCE_TOOL]
            and PREFERENCE_TOOL not in env.called_tools):
        return PREFERENCE_TOOL
    recent_genres = {
        env.base.genre.get(int(item), 0) for item in env.base.recent[-5:]}
    if (len(recent_genres) <= 2 and mask[DIVERSITY_TOOL]
            and DIVERSITY_TOOL not in env.called_tools):
        return DIVERSITY_TOOL
    return STOP_AND_SERVE


def evaluate_multitool_policy(
    name: str,
    action_fn,
    *,
    make_env: Callable[[int], MultiToolRecEnv],
    seed: int,
    sessions: int,
) -> dict:
    if sessions < 1:
        raise ValueError("sessions must be positive")
    user_returns = []
    net_returns = []
    lengths = []
    diversities = []
    budget_spent = []
    tool_counts = {name: 0 for name in TOOL_NAMES.values()}
    tool_calls = 0
    invalid_actions = 0
    agent_actions = 0
    recommendations = 0
    multi_tool_decisions = 0
    source_counts: dict[str, int] = {}
    for session in range(sessions):
        env = make_env(seed + session)
        state = env.reset()
        user_return = 0.0
        net_return = 0.0
        genres = []
        while True:
            mask = env.action_mask()
            action = int(action_fn(state, mask, env))
            state, reward, done, info = env.step(action)
            agent_actions += 1
            net_return += float(reward)
            user_return += float(info["user_reward"])
            invalid_actions += int(info["invalid_action"])
            if info["tool_used"]:
                tool_calls += 1
                tool_counts[info["tool_name"]] += 1
            if info["decision_complete"]:
                recommendations += 1
                genres.append(int(info["genre"]))
                multi_tool_decisions += int(info["multi_tool_decision"])
                source = str(info["selected_source"])
                source_counts[source] = source_counts.get(source, 0) + 1
            if done:
                break
        user_returns.append(user_return)
        net_returns.append(net_return)
        lengths.append(len(genres))
        diversities.append(len(set(genres)))
        budget_spent.append(env.initial_budget - env.remaining_budget)
    return {
        "policy": name,
        "sessions": sessions,
        "avg_user_reward": round(float(np.mean(user_returns)), 6),
        "avg_net_reward": round(float(np.mean(net_returns)), 6),
        "avg_session_length": round(float(np.mean(lengths)), 6),
        "avg_genre_diversity": round(float(np.mean(diversities)), 6),
        "avg_budget_spent": round(float(np.mean(budget_spent)), 6),
        "tool_calls_per_decision": round(tool_calls / max(1, recommendations), 6),
        "multi_tool_decision_rate": round(
            multi_tool_decisions / max(1, recommendations), 6),
        "invalid_action_rate": round(invalid_actions / max(1, agent_actions), 6),
        "agent_actions_per_decision": round(
            agent_actions / max(1, recommendations), 6),
        "tool_mix": {
            tool: round(count / max(1, tool_calls), 6)
            for tool, count in tool_counts.items()},
        "selected_source_share": {
            source: round(count / max(1, recommendations), 6)
            for source, count in sorted(source_counts.items())},
    }

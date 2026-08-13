# AI Handoff

## Role

Codex is taking over the implementation and research work previously produced by Kimi K3. Start every task with the startup harness and use the project-local skill `recsys-learning-maintainer` when available.

## Handoff Contract

- State the target surface before editing: data, model, evaluation, experiment, or demo.
- Reuse existing checkpoints and logs before retraining.
- Keep evaluation protocol explicit (ML-1M vs ML-25M, leave-one-out, negative sampling, sampled users).
- Add or update a focused note when a change introduces a new algorithmic finding or a failure diagnosis.
- End with the exact verification command and any runtime/dependency limitation.

## Likely Next Tasks

1. ~~Improve reproducibility: seed control, config capture, and a single experiment manifest.~~ **DONE (2026-07-22)**
2. ~~Harden the Streamlit demo for missing/corrupt checkpoints and empty candidate sets.~~ **DONE (2026-07-23)**
3. ~~Add automated smoke tests for data loaders, metric functions, and RL environment transitions without GPU/external services.~~ **DONE (2026-07-23)**
4. ~~Extend the research track toward offline-RL or a real feedback/log replay protocol while preserving the current simulator baseline.~~ **DONE (2026-07-24)**
5. ~~Package a serving entrypoint (e.g. a thin API wrapper around the two-tower→DeepFM pipeline) reusing `app.py`'s loading path.~~ **DONE (2026-07-23)**

## Offline RL / Post-Training (Completed 2026-07-24)

- `research_v2/phase2_offline_rl/`: the recsys analogue of LLM post-training (SFT → RLHF) under the real constraint that production can't explore online — you only have logged data.
  - `dataset.py`: freeze a behavior policy's rollouts into a fixed `*.npz` (with discounted return-to-go). Training never touches the env after this.
  - `behavior_cloning.py`: SFT = plain BC + reward-weighted BC (the "rejection-sampling SFT" analogue).
  - `cql.py`: offline RL = Conservative Q-Learning (Double DQN + CQL(H) term); `cql_alpha=0` gives naive offline DQN.
  - `train.py` (orchestrator) + `ablation_alpha.py` (α sweep, reuses one dataset).
- Honest findings (RecSimEnv, 200-session eval): **SFT reproduces the behavior policy exactly (BC fidelity 99.7% on greedy logs → reward 2.87, cannot exceed it); naive offline DQN improves past it (reward 8–13.6, learns to interleave genres); CQL's conservatism monotonically hurts here (α↑ → collapses toward the myopic behavior policy).** Root cause: bounded returns (churn cap + LayerNorm) and all-actions-valid mean the OOD-overestimation catastrophe CQL guards against never arises in this simulator. CQL's mechanism is verified in isolation by a unit test. Lesson: **diagnose data coverage / return structure before reaching for conservatism.**
- Verify: `RECSYS_INTEGRATION` not needed; `.venv/Scripts/python.exe -m unittest tests.test_offline_rl`. Full run: `python research_v2/phase2_offline_rl/train.py --smoke`. See `notes/12-推荐系统后训练-离线RL与SFT.md`.

## Serving (Completed 2026-07-23)

- `src/serving.py`: streamlit-agnostic `Recommender` (two-tower recall → DeepFM rank) — single source of truth reused by `app.py` and the API. `check_artifacts()`, `UnknownUserError`, and empty-candidate handling live here.
- `serve.py`: Starlette + uvicorn REST API (no new deps; both already installed). Routes: `GET /health`, `/users`, `/users/{uid}`, `/recommend?user_id=&k=`. Model is a lazily-loaded process singleton.
- Measured (ML-1M, CPU): model load ~8s once; `recommend(user=1, k=10)` returns 10 unseen items in ~9 ms; `/recommend` HTTP round-trip ~4.5 ms. Error paths return 400/404/503, never a 500 traceback.
- Run: `.venv/Scripts/python.exe serve.py --port 8000 --preload`; verify `GET /health`. See `notes/11-工程化落地-测试与服务化.md`.

## Production Observability & Load Test (Completed 2026-07-26)

- `src/observability.py`: dependency-free, thread-safe process metrics with bounded latency samples, stable route labels, status/error taxonomy, JSON access logs, `X-Request-ID`, and `Server-Timing`.
- `serve.py`: adds `GET /metrics` without triggering model load or counting the scrape itself. Model loading now uses a lock, so concurrent cold-start requests cannot build duplicate Torch/Faiss models. `/health` reports load duration and uptime.
- `scripts/load_test_api.py`: standard-library keep-alive load generator. Every named run refuses overwrite and saves config, client latency/throughput, correlation failures, and before/after service-metric deltas under `experiments/serve_load_<tag>.json`.
- Measured on localhost, one Uvicorn worker, CPU, `GET /recommend?user_id=1&k=10`:

  | Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
  |---:|---:|---:|---:|---:|---:|---:|
  | 1 | 200 | 285.845 RPS | 3.436 ms | 4.301 ms | 4.476 ms | 0 |
  | 16 | 500 | 291.192 RPS | 53.048 ms | 59.966 ms | 96.839 ms | 0 |
  | 64 | 1000 | 293.935 RPS | 139.137 ms | 185.898 ms | 1733.937 ms | 0 |

- Interpretation: throughput plateaus near 290 RPS; higher concurrency creates queueing instead of capacity. Keep a single worker below roughly 16 concurrent in-flight requests if a 100 ms p99 is desired, then validate multi-worker scaling. The server middleware measures application time after ASGI entry, so the client report—not the lower server-side app latency—is authoritative for queue-inclusive SLOs.
- Evidence: `experiments/serve_load_baseline-c1-20260726.json`, `experiments/serve_load_contention-c16-20260726.json`, `experiments/serve_load_saturation-c64-20260726.json`. Service counter deltas exactly matched 200/500/1000 successful requests; live 400/404 probes populated `invalid_request` and `not_found`.
- Verify: `.venv/Scripts/python.exe -m unittest tests.test_observability -v`; `.venv/Scripts/python.exe scripts/ai_startup_harness.py --test`; start `serve.py --preload`, then run `scripts/load_test_api.py` with a unique `--tag`.

## Overload Protection & Worker Scaling (Completed 2026-07-26)

- `serve.py`: synchronous Torch/Faiss work now runs in Starlette's thread pool, keeping the event loop responsive. `AdmissionControlMiddleware` caps each worker's in-flight business requests and returns `429` plus `Retry-After` instead of building an unbounded queue; `/health` and `/metrics` are exempt. CLI flags: `--max-in-flight` (default calibrated to 8) and `--workers`.
- `src/observability.py`: classifies 429 as `overloaded`, exports admission totals/peak/in-flight plus the current worker PID. The hot `/health` path no longer waits for a saturated inference thread-pool token.
- `scripts/load_test_api.py`: reports successful RPS and latency split by status, so fast 429 responses cannot hide the p99 of admitted HTTP 200 requests.
- `scripts/scaling_experiment.py`: starts/stops exact Uvicorn process trees, waits until every worker PID reports a loaded model, records client-side scaling data and manager+worker RSS, retries transient timeouts, and can run a focused gate sweep.
- Verified localhost results on Windows, 12 logical CPUs, ML-1M, 600 requests per point:

  | Workers | Concurrency | Successful RPS | HTTP 200 p99 | Tracked RSS |
  |---:|---:|---:|---:|---:|
  | 1 | 16 | 159.8 | 148.8 ms | 887.6 MiB |
  | 2 | 16 | 270.1 | 72.4 ms | 1778.2 MiB |
  | 4 | 16 | 385.5 | 51.9 ms | 3547.8 MiB |
  | 4 | 32 | 475.4 | 81.9 ms | 3547.8 MiB |

- Scaling is useful but not linear: at concurrency 16, 2/4 workers deliver 1.69x/2.41x the single-worker throughput while memory grows 2.00x/4.00x. A 4-worker, concurrency-4 point reached 630.6 RPS, but keep-alive connection placement and CPU scheduling make individual localhost points noisy; use the same-concurrency rows above for conservative decisions.
- Gate sweep at concurrency 64: gate 8 preserved 132.4 successful RPS versus 148.6 with a wide gate, while reducing admitted HTTP-200 p99 from 869.6 ms to 340.8 ms. It shed 533/600 requests, so callers must honor `Retry-After`. Gate 32 had 806.1 ms admitted p99 and was rejected as the default. No single-worker gate met a 100 ms p99 under this extreme burst.
- Recommended starting deployment for this machine: **2 workers × max-in-flight 8** (~1.78 GiB measured RSS), then tune from real traffic/SLOs. For more throughput, 4 workers reached 385–475 successful RPS at concurrency 16–32 but consumed ~3.55 GiB.
- Evidence: `experiments/serve_scaling_verified-20260726.json`, `experiments/serve_scaling_overload-gates-verified-20260726.json`, `figures/serve_worker_scaling_verified-20260726.png`, `figures/serve_overload_gates_verified-20260726.png`.
- The previously listed local production gaps—worker aggregation, event-loop/process metrics, and ingress rate limiting—are completed in the section below. Current capacity numbers remain localhost evidence, not a networked production SLO.

## Production Operations Closure (Completed 2026-07-26)

- `src/metrics_registry.py`: workers atomically publish bounded snapshots into an instance-specific runtime directory. Any worker can serve exact recent-sample aggregation through `GET /metrics/aggregate` or Prometheus text through `GET /metrics/prometheus`. Files older than the freshness window and abandoned temporary writes are removed; normal lifespan shutdown unregisters immediately.
- `src/observability.py`: adds bounded 10/60-second QPS, per-status route latency, process CPU/current RSS, event-loop lag, and a per-worker token bucket. Rate-limit 429 (`rate_limited`) is distinct from concurrency shedding (`overloaded`).
- `serve.py`: adds non-loading `GET /live` and `/ready`, runtime sampling/publication, `--rate-limit`, `--rate-burst`, `--metrics-dir`, `--metrics-interval`, `--graceful-timeout`, and `--keep-alive`. Probe/metrics routes bypass both traffic gates.
- `monitoring/prometheus_rules.yml`: starter alerts for target/model failure, HTTP-200 recommendation p99, loop lag, admission shedding, rate rejection, and RSS. `docs/PRODUCTION_SERVING.md` is the deployment/runbook and documents security/single-host boundaries.
- `scripts/production_smoke.py`: real-checkpoint two-worker validation with paced traffic, a concurrency-32 burst, exact aggregate reconciliation, per-worker traffic proof, Prometheus checks, runtime signals, and forced-crash stale-file cleanup. `tests/test_serve_lifecycle.py` separately exercises normal graceful lifespan cleanup and guards the late-`.tmp` race found during verification.
- Final evidence (`experiments/serve_production_smoke_final-v2-20260726.json`, Windows, 12 logical CPUs):
  - all **13/13** production checks passed; both workers loaded and served traffic;
  - paced phase: 40/40 HTTP 200, client p99 **7.151 ms**;
  - burst phase: 300 requests, 27 HTTP 200 / 273 HTTP 429, admitted client p99 **88.740 ms**, no transport or request-correlation errors;
  - aggregate count exactly 340; error classes split into rate limiting and concurrency overload;
  - RSS/CPU/event-loop lag and Prometheus output present; registry empty after shutdown/crash recovery.
- Recommended starting command: `.venv/Scripts/python.exe serve.py --port 8000 --preload --workers 2 --max-in-flight 8 --rate-limit 80 --rate-burst 8`. Put authentication/TLS and cross-host rate limiting at the ingress.

## Testing (Completed 2026-07-23)

- `tests/` holds a stdlib `unittest` smoke suite — no pytest, no GPU, no network:
  - `test_metrics.py`: pins `train_deepfm.auc_score` against a brute-force pairwise AUC reference.
  - `test_observability.py`: route cardinality, error taxonomy, request headers/metrics, exception accounting, and admission-control shedding/exemptions.
  - `test_load_testing.py`: successful-throughput and status-specific latency summaries, including 429 vs HTTP 200 separation.
  - `test_feedback.py`: slate propensity math, SQLite integrity/idempotency,
    replay export, IPS/SNIPS, and explicit zero-support rejection.
  - `test_sasrec.py`: `pad_left` behavior + the SASRec causal-attention invariant (future tokens never leak into earlier positions) on a tiny CPU model.
  - `test_recsim_env.py`: `RecSimEnv` fatigue (`0.75 ** (run_len-1)`) and 3-dislike churn exit, driven by a stub user model instead of the SASRec checkpoint.
- Latest run: 61 discovered, 59 passed, 2 real-checkpoint serving tests skipped by their `RECSYS_INTEGRATION=1` gate. The feedback API/UI smokes and prior production smoke loaded the real checkpoints.
- Run: `.venv/Scripts/python.exe scripts/ai_startup_harness.py --test` (or `-m unittest discover -t . -s tests -v`).

## Demo Hardening (Completed 2026-07-23)

- `app.py` now self-checks required artifacts (`check_artifacts()`), degrades load failures to `st.error`, and guards the empty-candidate / missing-metadata paths that previously crashed `max()` on an empty sequence.

## Reproducibility (Completed)

- `src/config.py`: unified `set_seed(seed)` (Python/NumPy/PyTorch/CUDA determinism) and `ExperimentConfig` dataclass with auto-save to `experiments/`.
- All 8 runnable scripts (`train_mf`, `train_deepfm`, `train_two_tower`, `train_sasrec`, `train_dqn_rec`, `train_pg_rec`, `train_ppo_lander`, `run_bandit`) accept `--seed` and save config snapshots.
- `scripts/experiment_manifest.py`: prints table/JSON/CSV of all `*_config.json` and `*_log.csv` / `*_eval.csv` files.
- Verify: `.venv/Scripts/python.exe scripts/experiment_manifest.py`

## Decision Notes

- Keep the two-stage serving boundary: retrieval is fast/coarse; DeepFM is slow/precise.
- For long-term RL, sampling deployment is intentional in this simulator; argmax is not an equivalent deployment.
- When a result changes, update the relevant CSV/figure/report together rather than silently changing a headline metric.

## Real Feedback Loop V1 (Completed 2026-07-27)

- `src/feedback.py`: SQLite/WAL store for recommendation candidate pools,
  served ranks, exact item inclusion propensities, impressions, and idempotent
  feedback. Successful recommendations fail closed with 503 if logging fails.
- `serve.py`: `POST /events/impression`, `POST /events/feedback`, and
  `GET /events/stats`; `/recommend` returns tracking/model/policy metadata.
  Exploration is opt-in (`--exploration-rate`, default 0).
- `scripts/feedback_replay.py`: refuses artifact overwrite, exports all
  candidates (including unobserved ones), hashes the JSONL, and writes
  item-level IPS/SNIPS, ESS, and support diagnostics.
- Real-checkpoint smoke evidence:
  `experiments/feedback_smoke-v1-20260727.sqlite3`,
  `experiments/feedback_smoke-v1-20260727.jsonl`, and
  `experiments/feedback_smoke-v1-20260727_ope.json`. One recommendation logged
  20 candidates / 5 impressions / 1 click; Top-5 support passed. This validates
  plumbing only, not policy uplift.
- Synthetic scaling/production-smoke traffic now uses tag-isolated feedback
  databases instead of polluting the default event store.
- Next optimization: collect a meaningful window, add uncertainty/confidence
  intervals and candidate-policy probabilities, then define shadow/A-B gates.

## Human Feedback Experiment V2 (Completed 2026-07-27)

- `feedback_app.py`: dedicated Streamlit collection surface. It logs an
  anonymous actor id, a 20-item candidate pool, 10 shown items, exact
  propensities, impressions, and one immutable preference per displayed item.
  Default local experiment exploration is 10%.
- Recommendation items now snapshot title/genres; SQLite initialization
  migrates existing V1 databases in place. API clients may provide
  `X-Actor-ID`; personal identifiers should not be used.
- Candidate policy: greedy `DeepFM score + 0.15 * uncovered-genre fraction`.
  It is evaluated against deterministic DeepFM Top-k with paired
  item-inclusion IPS by recommendation.
- Formal gate: at least 200 recommendations from 5 anonymous actors, 95%
  impression coverage, ESS 100, full propensity support, and a 95%
  actor-cluster percentile-bootstrap interval. Decisions
  are `promote_candidate`, `keep_baseline`, `continue_experiment`, or
  `collect_more_data`.
- Isolated browser QA:
  `experiments/feedback_ui-smoke-v1-20260727.sqlite3` contains 1 recommendation,
  20 candidates, 10 impressions, and 1 UI preference; no browser console
  errors. This database is QA evidence and must never be merged with
  `runtime/feedback/events.sqlite3`.
- The real collector was started at `http://127.0.0.1:8502` with
  `.venv/Scripts/python.exe -m streamlit run feedback_app.py --server.port 8502`.
  Opening the handed-off page created the first real exposure batch in
  `runtime/feedback/events.sqlite3`; feedback remains human-controlled.
  A formal verdict is intentionally unavailable until the on-page gate passes.

## Low-friction Personal Pilot (Completed 2026-07-27)

- `feedback_app.py` now defaults to a low-friction personal pilot: 10 candidates,
  5 displayed, 20% exploration, and one “most want to watch” click per batch.
  The click atomically records one click plus four explicit skips and advances
  to the next batch; collection stops after 20 complete batches (20 total user
  clicks).
- Pilot evaluation is isolated by `epsilon_slate_personal_pilot_v1` and current
  anonymous actor. It uses 20 complete batches, one actor, ESS 5, and a 90%
  actor-cluster bootstrap interval. Its wording is deliberately a personal
  signal, never a production decision.
- The original per-item workflow remains under “多人正式评估” with the existing
  200-batch/5-actor/95%-coverage/ESS-100/95%-confidence gate.
- Isolated end-to-end browser evidence:
  `experiments/feedback_quick-ui-oneclick-v1-20260727.sqlite3` contains exactly
  20 recommendations, 200 candidates, 100 impressions, and 100 feedback events
  (20 clicks + 80 skips). Browser automation used exactly one movie-button
  click per batch; the UI stopped at 20/20, created no 21st batch, and emitted
  no browser console errors. The isolated QA run did not touch the real
  database. Applying the updated page to the user's existing tab created the
  first legitimate but incomplete pilot exposure; at that point the real
  database held 3 recommendations / 25 impressions / 10 feedback events
  (2 legacy formal batches plus 1 pilot batch awaiting the user's choice).
- Verification: `.venv/Scripts/python.exe scripts/ai_startup_harness.py --test`
  passed 61 tests (59 passed, 2 real-checkpoint-gated skips).

## External Survey Validation (Completed 2026-07-27)

- Official GroupLens Personality 2018 was downloaded for local research use
  from `https://grouplens.org/datasets/personality-2018/`. The archive SHA-256
  is recorded in `data/external/personality-isf2018/SOURCE.md`; raw files are
  excluded by `.gitignore` because the upstream license prohibits
  redistribution without permission.
- `scripts/validate_external_survey.py` compares default vs low/medium/high
  diversity assignments on perceived personalization and expected enjoyment,
  using 10,000 bootstrap draws, 10,000 two-sided permutations, and Holm
  correction across six predeclared contrasts.
- Evidence:
  `experiments/external_survey_personality-2018-diversity-v1-20260727.json`.
  The four relevant groups contain 728 users. Medium diversity's enjoyment
  difference is +0.113/5 but its 95% interval crosses zero. High diversity
  reduces perceived personalization by 0.332/5
  (95% bootstrap interval [-0.542, -0.121], Holm-adjusted p=0.0144).
- Decision: external evidence does not support a global diversity rollout.
  Combined with the completed local pilot (+11.1% IPS direction), it supports
  keeping the 0.15 reranker as a personal/experimental candidate.

## Interview Packaging (Completed 2026-07-27)

- `README.md` now opens with a recruiter-oriented 30-second overview and routes
  to the interview materials before retaining the full learning path.
- `docs/RESUME_PROJECT.md` contains concise and detailed Chinese bullets, an
  English version, and role-positioning guidance.
- `docs/INTERVIEW_PLAYBOOK.md` contains 60-second and 5-minute narratives,
  technical/business deep dives, likely questions, honest boundaries, and a
  pre-interview checklist.
- `docs/PROJECT_EVIDENCE.md` maps every interview metric to its exact protocol
  and repository evidence, and explicitly lists metrics that must not be mixed.
- `docs/interview/Recsys_Interview_Deck_CN.pptx` is a nine-slide editable deck
  with speaker notes and per-slide `[Sources]` blocks. Every slide was rendered
  and individually inspected; `slides_test.py` reported no overflow.
- `项目总结.md` was corrected from the obsolete 29-test count; the current
  suite has 73 discovered / 71 passed / 2 integration-gated skips.

## Agentic Recommendation P0 (Completed 2026-07-27)

- `research_v3/agentic_rec/` adds a DeepEyes-inspired minimal tool-selection
  layer over the frozen SASRec `RecSimEnv`: baseline raw scoring versus a
  fatigue-aware reranking tool.
- Reward accounting separates raw user reward, tool-cost net reward, and a
  conditional tool bonus. Training is group-relative REINFORCE, explicitly not
  full GRPO because it has no old-policy ratio or clipping.
- Verified experiment:
  `experiments/agentic_rec_deepeyes-minloop-v2-20260727_report.json` (ML-1M,
  frozen SASRec, 300 same-seed simulator sessions). The rule gain gate preserved
  user reward 17.947, used the tool on 87.2% rather than 100% of steps, and
  improved net reward from 15.947 to 16.203.
- Learned argmax collapsed to always-tool; sampled deployment lost raw reward;
  removing the conditional bonus made almost no difference. Decision: deploy
  neither learned variant in this simulator; retain the interpretable rule gate.
- Focused verification: `.venv/Scripts/python.exe -m unittest tests.test_agentic_rec -v`.
- Research note: `notes/13-Agentic推荐-主动工具调用.md`.

## SASRec Ablation Matrix (Completed 2026-08-04)

- Completes the W2 milestone from `research/课题设计书.md` that `notes/08` still
  listed as TODO: the one-at-a-time ablation matrix (d / blocks / maxlen /
  dropout). Previous partial runs stopped at epoch 30 with no checkpoints and no
  test-set verdicts.
- Protocol: ML-1M, leave-one-out + 100 negatives, 100 epochs, eval_every 10,
  seed 42, best-valid selection then full-user test evaluation. Baseline is
  `sasrec_main.pt` (d64 / 2 blocks / maxlen 200 / dropout 0.2); every variant
  changes exactly one axis.
- Test HR@10 / NDCG@10: baseline 0.7909 / 0.5428; d=128 0.7788 / 0.5156;
  blocks=3 **0.7972 / 0.5489** (best); maxlen=50 0.7659 / 0.5107;
  dropout=0.5 0.7791 / 0.5174. Verdict: deeper helps modestly; wider embedding,
  shorter context, and higher dropout all hurt on the small ML-1M corpus.
- Artifacts: `experiments/sasrec_ablation_matrix.csv`,
  `figures/sasrec_ablation_valid_ndcg-20260804.png`,
  `figures/sasrec_ablation_test-20260804.png`, checkpoints
  `sasrec_ablation_{d128,b3,L50,do05}.pt`.
- Reproduce: `.venv/Scripts/python.exe scripts/sasrec_ablation_summary.py`
  (reloads the five checkpoints and re-evaluates on the full test set; baseline
  numbers reproduce exactly).
- Still open on this track: full ML-25M training and a GRU4Rec baseline
  (`notes/08` 待办).

## Frontier RL Survey + Hand-written PPO (Completed 2026-08-05)

- `notes/14-前沿RL论文与落地路线.md` maps the 2023-2025 high-impact RL papers
  (PPO/GAE, GRPO/DeepSeekMath, DeepSeek-R1, DPO/SimPO/Softmax-DPO, Decision
  Transformer, DAPO, DreamerV3, RL/Offline-RL-for-RecSys surveys) onto this
  project's existing assets, with a feasibility score and a three-phase
  roadmap (PPO done; GRPO for the agentic tool gate; DPO rerank on the real
  feedback loop).
- `src/train_ppo_rec.py`: hand-written PPO on `RecSimEnv` (GAE λ=0.95, clip
  ε=0.2, K=4 epochs, advantage normalization, entropy bonus), reusing the
  same ActorCritic network, environment, seeds, and 300-session evaluation
  protocol as PG/DQN.
- Result (300 same-seed sessions): PPO(sampled) 15.43 reward / 19.2 length /
  6.87 genre diversity vs PG(sampled) 15.86 / 19.5 / 5.42, Double DQN 7.63,
  greedy 2.81. Verdict: PPO matches the best known policy with ~27% higher
  diversity; argmax still collapses (sampled deployment required).
- Tests: `tests/test_ppo_rec.py` pins three GAE invariants (λ=1⇒MC returns,
  terminal bootstrap reset, λ=0⇒one-step TD).
- Reproduce: `.venv/Scripts/python.exe src/train_ppo_rec.py --updates 40
  --resume`; smoke: `--updates 1 --no_eval`.
- Next candidates on the roadmap: DPO/S-DPO (or KTO) rerank against the real
  feedback loop; GRPO itself is now done (below).

## Literature-Driven IQL Trial (Completed 2026-08-05)

- The 50-paper survey (notes/14) was used to change an actual direction: after
  CQL's conservatism hurt in RecSimEnv, the offline-RL track pivoted from more
  CQL alpha sweeps to testing IQL (Kostrikov et al. 2021, arXiv:2110.06169),
  which avoids the max bootstrap entirely (expectile V + advantage-weighted BC).
- `research_v2/phase2_offline_rl/iql.py` (Q + expectile V + AWR policy) and
  `train_iql.py` (reuses the frozen phase2 npz, 15000 steps, 300-session eval
  against SFT/naive/CQL).
- Honest negative result: IQL reaches 3.35-3.40 reward (tau/beta swept), same
  side as CQL (3.63), while naive offline DQN reaches 9.43. In this simulator
  the max bootstrap is exactly what discovers the interleaving policy, so
  "swap CQL for another safety mechanism" does not help; the fix is data
  coverage + a real OOD scenario. Conclusion refined in notes/12 §3.5.
- Tests: `tests/test_iql.py` (5 cases, CPU, synthetic). Run:
  `.venv/Scripts/python.exe research_v2/phase2_offline_rl/train_iql.py
  --steps 15000`.

## Full GRPO on the Agentic Tool Gate (Completed 2026-08-05)

- `research_v3/agentic_rec/train.py` now supports `--algo grpo`: group-relative
  advantage + clipped importance ratio (eps=0.2) + KL to a frozen initial-policy
  reference (beta=0.04) + `--epochs K` reuse (DeepSeekMath/R1 recipe).
- Implementation lesson recorded in notes/13: a first version using an online
  per-update reference with a single gradient step degenerated bit-for-bit to
  REINFORCE (ratio always 1, KL gradient zero at the reference); switching to a
  frozen reference and multi-epoch reuse is what makes clip/KL act.
- Same-budget comparison (40x8x4, 300 same-seed sessions, tool_cost=0.03):
  sampled net reward REINFORCE 15.905 -> GRPO K=1 15.899 -> **GRPO K=4 16.668**
  (+4.8%), tool rate 70.4% -> 85.8%; rule gain gate remains the deployment
  choice (17.423); learned argmax still collapses to always-tool.
- Artifacts: `experiments/agentic_rec_grpo-compare-{reinforce,k1,k4}_*`.
- Tests: `tests/test_agentic_rec.py` now 14 cases (5 new GRPO loss invariants).

## Beginner Course + Core-Concept Visual Rewrite (Completed 2026-08-07)

- `docs/从零到懂-课程浓缩指南.md`: a self-contained ~50k-character curriculum
  covering math basics -> ML -> deep learning -> recommender systems -> RL
  (REINFORCE/DQN/PPO/GRPO/offline RL) -> LLM alignment (DPO/KTO/GRPO) ->
  interview prep, with 11 numbered chapters, per-chapter self-tests, hands-on
  exercises mapped to this repo's scripts, and a 60+ term plain-language
  glossary.
- The 2026-08-07 pass rewrote the main knowledge path around
  intuition -> mechanism -> formula -> project evidence -> failure boundary,
  adding 15 Mermaid diagrams plus matrix, curve, timeline, and comparison-table
  views. A second pass connected the foundations to primary 2023-2026 work on
  TIGER/Semantic IDs, OneRec, DAPO, GSPO, and Agentic Recommendation without
  presenting preprints or single-system gains as production guarantees. It also
  added four dense, accessible SVG explainers under `figures/`: the industrial
  offline/online/feedback loop, classical-vs-generative recommendation, the
  PPO/GRPO/DAPO/GSPO update path, and the agentic memory/tool/verification loop.
  It corrected beginner-facing traps including normalized dot product vs
  cosine similarity, confidence-interval coverage, AUC/GAUC limits, FM's
  factorized interaction, scaled attention and causal visibility, logQ's
  correction direction, Bellman expectation vs optimality, advantage-sign
  semantics, PPO clipping limits, `gamma=0` versus a concrete greedy policy,
  recipe-dependent KL in GRPO variants, and exposure bias in KTO-style feedback.
- `src/train_two_tower.py` received a comment-only logQ clarification so source
  and course no longer teach opposite intuitions; model behavior is unchanged.
- Verification: Markdown fences are balanced (82 fence markers); all four SVGs parse as XML, have
  unique element IDs plus accessible `<title>`/`<desc>`, stay inside their
  declared viewboxes, and are referenced by existing relative paths. Browser
  rendering could not be used because the in-app browser blocks local `file:`
  URLs, so no browser-level visual claim is recorded. Run
  `.venv/Scripts/python.exe scripts/ai_startup_harness.py --check` after edits.

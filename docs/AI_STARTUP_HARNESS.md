# AI Startup Harness

Use this page as the compact project cache for a new coding model (including Kimi K3).
Run ` .venv/Scripts/python.exe scripts/ai_startup_harness.py --check` first; it prints a shorter machine-readable summary.

## Project In One Sentence

An educational but runnable MovieLens recommendation stack: offline data exploration and matrix factorization, CTR ranking (LR/FM/DeepFM), two-tower + Faiss retrieval, contextual bandits, SASRec sequence recommendation, and simulated-session RL for long-term satisfaction, exposed through a Streamlit demo.

## Architecture

```text
MovieLens-1M/25M
  -> preprocessing and leave-one-out/negative-sampling protocols
  -> MF baseline | DeepFM ranking | TwoTower retrieval + Faiss | Bandit
  -> SASRec user model
  -> RecSimEnv (fatigue + churn) -> Double DQN / REINFORCE Actor-Critic
  -> checkpoints/, experiments/, figures/, REPORT.md, app.py
```

## Current Completed State

- Phase 1 complete: MF Recall@10 0.062; DeepFM test AUC 0.754 (temporal split); two-tower corrected Recall@10 0.097; LinUCB online-simulation CTR 0.704.
- Phase 2 complete: SASRec ML-1M test HR@10 0.7909 / NDCG@10 0.5428; ML-25M 20k-user scale check; DQN v2 stable; policy-gradient sampling reward 15.86 vs naive greedy 2.81 in the simulated session environment. A fair greedy-dedup baseline (skip the previous genre) scores 17.93 — above DQN 7.63 and PG 15.86 — so the hand-written fatigue reward makes "just alternate genres" near-optimal and RL adds little here.
- Agentic recommendation P0 complete: a DeepEyes-inspired two-action tool gate
  chooses between raw SASRec and fatigue-aware reranking under explicit cost and
  conditional reward. On 300 frozen-SASRec simulator sessions, an analytic gain
  gate preserved user reward 17.947 while reducing tool calls 12.8% and improving
  net reward to 16.203 versus 15.947 for always-tool. Group-relative REINFORCE
  collapsed to always-tool under argmax; decision: keep the rule gate, do not
  promote the learned gate. See `notes/13-Agentic推荐-主动工具调用.md`.
- Product artifact complete: `app.py` demonstrates two-tower recall (50 candidates) followed by DeepFM Top-10 ranking.
- Production observability complete: `serve.py` exposes `/metrics`, bounded request latency/error metrics, correlation headers, structured access logs, and thread-safe single model loading; `scripts/load_test_api.py` writes reproducible load reports.
- Capacity hardening complete: synchronous inference is offloaded from the event loop, per-worker admission control sheds excess work with 429, and `serve.py` supports 1/2/4+ Uvicorn workers. Verified same-concurrency scaling and RSS evidence live in `experiments/serve_scaling_verified-20260726.json`.
- Production operations complete: live/ready probes, token-bucket rate limiting, rolling QPS, CPU/RSS/event-loop lag, same-host cross-worker JSON aggregation, Prometheus exposition/rules, bounded runtime registry cleanup, and a real-checkpoint production smoke harness.
- Feedback loop V1 complete: SQLite/WAL recommendation, candidate, impression, and feedback events; explicit behavior propensities; immutable JSONL replay export; item-level IPS/SNIPS with support checks. See `docs/FEEDBACK_LOOP.md`.
- Human feedback experiment V2 complete: `feedback_app.py` collects anonymous
  real-human preferences under explicit epsilon exploration; replay compares
  DeepFM Top-k with diversity reranking using paired IPS, bootstrap confidence
  intervals, minimum sample/coverage/ESS gates, and an explicit decision.
- Feedback UX closure complete: the collector defaults to a 20-batch personal
  pilot with one five-way choice per batch and automatic stop; the original
  200-batch/5-actor formal protocol remains available as a separate mode.
- External survey validation complete: GroupLens Personality 2018 provides a
  licensed 1,834-user population check. Moderate diversity has an uncertain
  positive enjoyment direction; high diversity significantly reduces perceived
  personalization. The current candidate remains personal/experimental rather
  than globally promoted.
- SASRec ablation matrix complete (2026-08-04): on ML-1M, 3 blocks slightly
  beats the 2-block baseline (test HR@10 0.7972 / NDCG@10 0.5489 vs
  0.7909 / 0.5428); d=128, maxlen=50, and dropout=0.5 all hurt. Evidence:
  `experiments/sasrec_ablation_matrix.csv`,
  `figures/sasrec_ablation_*-20260804.png`; rerun via
  `.venv/Scripts/python.exe scripts/sasrec_ablation_summary.py`.
- Hand-written PPO on RecSimEnv complete (2026-08-05): matches the best
  policy-gradient baseline (sampled reward 15.43 vs PG 15.86) with ~27% higher
  genre diversity (6.87 vs 5.42); argmax still collapses. Frontier-RL survey
  and roadmap live in `notes/14-前沿RL论文与落地路线.md`.
- Literature-driven offline-RL trial complete (2026-08-05): IQL
  (expectile V + advantage-weighted BC, arXiv:2110.06169) was tested against
  the frozen phase2 dataset as the paper-suggested follow-up to CQL's negative
  result; it also stays near the behavior policy (3.35-3.40 vs CQL 3.63,
  naive DQN 9.43). Conclusion refined in `notes/12` §3.5: the lever is data
  coverage, not swapping safety mechanisms. See
  `research_v2/phase2_offline_rl/iql.py` and `train_iql.py`.
- Full GRPO on the agentic tool gate complete (2026-08-05): `--algo grpo` adds
  clipped ratio + KL + K-epoch reuse (DeepSeekMath/R1 recipe). Sampled net
  reward 15.905 -> 16.668 (+4.8%) at K=4; the rule gain gate remains the
  deployment choice. See `notes/13` and
  `research_v3/agentic_rec/train.py`.
- Interview package complete: recruiter-first README, Chinese/English resume
  bullets, evidence index, 60-second/5-minute talk tracks, defensible Q&A, and a
  rendered/verified nine-slide PowerPoint live under `docs/`.
- Beginner curriculum visual rewrite complete (2026-08-07):
  `docs/从零到懂-课程浓缩指南.md` now teaches each core concept through
  intuition, mechanism, formula, project evidence, and failure boundaries,
  with Mermaid flows, matrices, curves, timelines, comparison tables, and four
  dense SVG explainers for the industrial feedback loop, generative
  recommendation, policy optimization, and agentic recommendation. Primary
  2023-2026 papers are connected to the established concepts with explicit
  evidence levels and non-transferability cautions.
  Definitions for normalized dot products, AUC/GAUC, confidence intervals,
  logQ, Bellman equations, PPO clipping, gamma=0 versus concrete greedy behavior,
  recipe-dependent KL, and feedback exposure bias were tightened for
  interview-safe accuracy.
- Existing artifacts are authoritative evidence: `checkpoints/`, `experiments/`, `figures/`, `REPORT.md`, and `research/研究报告.md`.

## Fast Orientation

| Need | Read/run |
|---|---|
| Explain the project | `项目导学.md`, then `REPORT.md` |
| Learn the stack from scratch (beginner) | `docs/从零到懂-课程浓缩指南.md`（数学→ML→DL→推荐→RL→对齐→面试，含 60+ 术语词典） |
| Reproduce sequence model | `src/train_sasrec.py`; `research/研究报告.md` |
| Debug RL | `src/train_dqn_rec.py`, `src/train_pg_rec.py`, `notes/09-手写DQN与策略梯度.md` |
| Offline RL / post-training | `research_v2/phase2_offline_rl/`, `notes/12-推荐系统后训练-离线RL与SFT.md` |
| Active tool gate | `research_v3/agentic_rec/`, `notes/13-Agentic推荐-主动工具调用.md` |
| Understand serving path | `app.py`, `src/serving.py`, `serve.py`, `src/train_two_tower.py`, `src/train_deepfm.py` |
| Inspect API capacity | `src/observability.py`, `scripts/load_test_api.py`, `experiments/serve_load_*.json` |
| Re-run worker/overload sweep | `scripts/scaling_experiment.py --tag <unique-tag>` |
| Deploy/monitor the API | `docs/PRODUCTION_SERVING.md`, `monitoring/prometheus_rules.yml` |
| Verify full production surface | `scripts/production_smoke.py --tag <unique-tag>` |
| Export feedback and run OPE | `scripts/feedback_replay.py --tag <unique-tag> --target-k 10` |
| Collect real human feedback | `streamlit run feedback_app.py --server.port 8502` |
| Check current files/dependencies | `scripts/ai_startup_harness.py --check`, `requirements.txt` |

## Safe Commands

```powershell
.venv/Scripts/python.exe scripts/ai_startup_harness.py --check
.venv/Scripts/python.exe scripts/ai_startup_harness.py --test
.venv/Scripts/python.exe src/train_pg_rec.py --updates 1 --no_eval
.venv/Scripts/python.exe src/train_dqn_rec.py --steps 50 --tag smoke
.venv/Scripts/python.exe src/train_ppo_rec.py --updates 1 --no_eval
.venv/Scripts/python.exe research_v2/phase2_offline_rl/train_iql.py --steps 300 --no_eval
.venv/Scripts/python.exe research_v3/agentic_rec/train.py --algo grpo --smoke --tag <unique-tag>
.venv/Scripts/python.exe src/train_sasrec.py --epochs 1 --eval_every 1 --eval_users 100
.venv/Scripts/python.exe -m streamlit run app.py
.venv/Scripts/python.exe serve.py --port 8000 --preload --workers 2 --max-in-flight 8 --rate-limit 80 --rate-burst 8
.venv/Scripts/python.exe scripts/load_test_api.py --requests 500 --concurrency 16 --tag local-c16
.venv/Scripts/python.exe scripts/scaling_experiment.py --tag local-scaling
.venv/Scripts/python.exe scripts/production_smoke.py --tag local-production
.venv/Scripts/python.exe research_v2/phase2_offline_rl/train.py --smoke  # 离线RL全流程冒烟
.venv/Scripts/python.exe research_v3/agentic_rec/train.py --smoke --tag <unique-tag>
```

Use the smallest relevant smoke command before any long run. Do not overwrite a named checkpoint or log without an explicit task; use `--tag`/`--ckpt` for experiments.

## Known Limits

The RL user is a frozen SASRec-based simulator, rewards and fatigue are hand-designed, the action space is Top-200 popular items, and ML-25M validation is sampled. Treat the metrics as research evidence for this simulator, not online product guarantees.

Local `/metrics` remains process-local; `/metrics/aggregate` and
`/metrics/prometheus` merge fresh worker snapshots through an instance-specific
single-host runtime directory. Cross-host metrics and rate limiting still need
Prometheus/an ingress gateway. Middleware latency starts at ASGI entry, while
client load latency includes connection/event-loop queueing; use the latter for
end-to-end SLOs. Capacity evidence is localhost on a 12-logical-CPU Windows host,
not a production-network benchmark. Loaded RSS scales almost linearly with
workers (~0.89/1.78/3.55 GiB for 1/2/4).

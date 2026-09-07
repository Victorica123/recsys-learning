# AI Startup Harness

Use this page as the compact project cache for a new coding model (including Kimi K3).
Run ` .venv/Scripts/python.exe scripts/ai_startup_harness.py --check` first; it prints a shorter machine-readable summary.

## Project In One Sentence

An educational but runnable MovieLens recommendation stack: offline data exploration and matrix factorization, explicit positive-rating ranking with CTR-style LR/FM/DeepFM architectures (not real CTR), two-tower + Faiss retrieval, contextual bandits, SASRec sequence recommendation, and simulated-session RL for long-term satisfaction, exposed through a Streamlit demo.

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

- Phase 1 complete: MF Recall@10 0.062; DeepFM positive-rating AUC 0.754 (rating>=4 among observed ratings, temporal split, user-level GAUC 0.735; MovieLens has no impression labels, so this is not CTR); two-tower corrected Recall@10 0.098; LinUCB online-simulation CTR 0.704.
- **Funnel evaluation complete (2026-08-15) - read this before touching the ranker**: the *former* served path (recall 50 -> DeepFM -> Top-10) scores Recall@10 0.065 versus 0.098 for no reranking at all, CI [-0.0445, -0.0220]. Root cause is sample selection bias (Top-10 distinct items collapse 1853 -> 773). Five retraining versions in `src/train_reranker.py` closed 88.7% of the gap; the best (v6, residual-from-retrieval + 4-fold cross-fitted features) reaches 0.0943 with CI [-0.0132, +0.0053] - statistically tied with no reranking, so it was **not** promoted. The current default is two-tower Top-10. The residual gate itself settled at 6% of the retrieval weight *and negative*, i.e. the model learned to copy retrieval and penalise the DeepFM signal: **an information problem, not an implementation problem**. See `notes/15-端到端评估与精排重建.md`.
- Evaluation protocol comparison (2026-08-15): the headline Recall@10 0.098 uses leave-one-out, which permits time travel across users. A global time split scores 0.0691 [0.0553, 0.0847] versus 0.0980 [0.0884, 0.1075], intervals disjoint. Both are reported with the difference labelled; leave-one-out stays primary for comparability. `train_two_tower.py --split time`, `scripts/eval_split_protocols.py`.
- Rolling full-catalog retrieval benchmark complete (2026-08-15): three global-time cutoffs x fixed 5% future horizon, no sampled negatives, MostPopular + Item-kNN + TwoTower (3 seeds, 10 epochs). Macro Recall@10 is 0.0718 / 0.0691 / 0.0689 respectively, so TwoTower does not win relevance; it does deliver 17%-30% catalog coverage versus Popular's 1.5%-2.6% and the best worst-window Recall. Authoritative artifact suffix `p1-rolling-fullcatalog-3seed-v2`; v1 Item-kNN is invalid due to incomplete seen-item filtering. See `notes/16`.
- Multi-route/sequence ranking complete (2026-08-15): equal-weight RRF over TwoTower + Item-kNN + Popular reaches exploratory three-window macro Recall@10 0.0787 but collapses coverage to 3.5%-5.0%, so it is not promoted. A bounded sequence-aware residual listwise ranker, trained only on strictly prior windows, changes W2/W3 Recall by just +0.00047 on average (range -0.00337..+0.00397). Candidate Recall@200 is only 44%-51%, making complementary recall the next bottleneck. See `notes/17`.
- Agentic tool-gate cost sweep exploratory pass complete (2026-08-15): across three seeds, the rule gate has the higher mean for tool_cost <= 0.3, the gap is near zero around 0.5, and the learned gate has the higher mean at 0.8 (+1.915). Three seeds are insufficient for formal uncertainty or significance claims; the script now withholds confidence intervals and verdicts below 10 seeds. The phase transition remains a hypothesis pending a >=10-seed rerun.
- Serving hot path optimised (2026-08-15): recommender core 4.69 -> 1.66 ms (2.8x, byte-identical output) after removing pandas scalar indexing; the remaining HTTP cost is dominated by the synchronous SQLite impression write.
- Phase 2 complete: SASRec ML-1M test HR@10 0.7909 / NDCG@10 0.5428; ML-25M 20k-user scale check; DQN v2 stable; policy-gradient sampling reward 15.86 vs naive greedy 2.81 in the simulated session environment. A fair greedy-dedup baseline (skip the previous genre) scores 17.93 — above DQN 7.63 and PG 15.86 — so the hand-written fatigue reward makes "just alternate genres" near-optimal and RL adds little here.
- Agentic recommendation P0 complete: a DeepEyes-inspired two-action tool gate
  chooses between raw SASRec and fatigue-aware reranking under explicit cost and
  conditional reward. On 300 frozen-SASRec simulator sessions, an analytic gain
  gate preserved user reward 17.947 while reducing tool calls 12.8% and improving
  net reward to 16.203 versus 15.947 for always-tool. Group-relative REINFORCE
  collapsed to always-tool under argmax; decision: keep the rule gate, do not
  promote the learned gate. See `notes/13-Agentic推荐-主动工具调用.md`.
- Budgeted multi-tool Agentic prototype complete (2026-08-15): every
  recommendation can call fatigue/diversity/preference tools in a zero/one/two
  call trajectory before STOP_AND_SERVE; session budgets, action masks,
  repeated/over-budget invalid calls and recovery are explicit. In the formal
  10-training-seed x 120-session comparison, budgeted rule net reward is 15.371
  versus masked-GRPO sample 10.725; paired delta -4.646, 95% interval
  [-4.974, -4.322]. Keep the rule; the learned policy is not promoted. This new
  hidden-preference simulator/cost protocol must not be mixed with the old
  binary tool-cost sweep. See `notes/20-Agentic多工具预算轨迹.md`.
  **Cost-source correction**: the P0 artifact above uses tool_cost 0.10. The
  later same-budget GRPO comparison uses 0.03 and must not be mixed with P0;
  the cost sweep's 0.8 direction is exploratory until the >=10-seed rerun.
- Product decision is now enforced in code (2026-08-15): `app.py`, `serve.py`,
  `feedback_app.py`, and `Recommender.load()` default to two-tower Top-K and do
  not load DeepFM. `ranking_policy=deepfm` is an explicit experiment/failure
  reproduction path; API responses expose `ranking_policy` and `score_type`.
- Release/reproducibility surface complete (2026-08-15): Python is constrained
  to 3.12, `uv.lock` freezes the full dependency graph, plotting no longer
  imports a desktop-private helper, and `artifacts/release_manifest.json` pins
  data/checkpoints by size + SHA-256. `scripts/verify_release.py` separates the
  promoted `serve` profile from optional `research` artifacts.
- Checkpoint supply chain hardened (2026-08-16): `src/checkpoint_io.py` is the
  only serving-side loader; every checkpoint is loaded with `weights_only=True`
  and the legacy `weights_only=False` serving path is removed. Declared-weight
  SHA-256/size startup verification is explicit via
  `serve.py --verify-checkpoint-hashes` or `RECSYS_VERIFY_CHECKPOINTS=1`, while
  `--release` remains the offline hard check. The v6 OOF reranker is now pinned
  in the research release profile.
- Current-default performance re-measured (2026-08-16): retrieval-only
  recommender core is 0.19 ms; HTTP end-to-end is 5.42 ms / p95 9.67 ms
  (WSL/Linux CPU, SQLite on local fast disk, c1 x 200). The previous 4.6 ms
  Windows number belonged to the retired DeepFM path and is historical only.
- Production observability complete: `serve.py` exposes `/metrics`, bounded request latency/error metrics, correlation headers, structured access logs, and thread-safe single model loading; `scripts/load_test_api.py` writes reproducible load reports.
- Capacity hardening complete: synchronous inference is offloaded from the event loop, per-worker admission control sheds excess work with 429, and `serve.py` supports 1/2/4+ Uvicorn workers. Verified same-concurrency scaling and RSS evidence live in `experiments/serve_scaling_verified-20260726.json`.
- Production operations complete: live/ready probes, token-bucket rate limiting, rolling QPS, CPU/RSS/event-loop lag, same-host cross-worker JSON aggregation, Prometheus exposition/rules, bounded runtime registry cleanup, and a real-checkpoint production smoke harness.
- Feedback/OPE V2 complete (2026-08-15): SQLite/WAL candidate-level exposure logs,
  exact marginal propensities, immutable schema-v2 JSONL, mandatory policy/model
  cohort isolation, IPS/SNIPS, cross-fitted DR, actor-cluster bootstrap and
  formal quality gates. Synthetic oracle calibration shows IPS delta error
  0.00028 versus SNIPS 0.00859 and DR 0.00812; DR is a cross-check, not an
  automatic upgrade. The 21-recommendation/1-actor historical audit is blocked
  as `collect_more_data`. See `docs/FEEDBACK_LOOP.md` and `notes/18-曝光数据与OPE校准.md`.
- Human feedback experiment V2 complete: `feedback_app.py` collects anonymous
  real-human preferences under explicit epsilon exploration; the current v2
  policy compares two-tower Top-k with diversity reranking using paired IPS, bootstrap confidence
  intervals, minimum sample/coverage/ESS gates, and an explicit decision.
- TIGER-lite Semantic-ID baseline complete (2026-08-15): deterministic
  metadata vectors -> 64x64x64 residual K-Means IDs + collision token ->
  autoregressive Transformer. Under the exact P1 rolling full-catalog protocol,
  nested temporal epoch selection and 3 seeds, macro Recall@10 is 0.0557 and
  coverage is 2.75%, below Popular/Item-kNN/TwoTower; not promoted. This is a
  controlled core-mechanism baseline, not a SentenceT5+RQ-VAE reproduction.
  See `notes/19-TIGER语义ID生成式召回.md`.
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
| Multi-tool trajectory experiment | `scripts/agentic_multitool_experiment.py --tag <unique-tag>`; `notes/20-Agentic多工具预算轨迹.md` |
| Understand serving path | `app.py`, `src/serving.py`, `serve.py`, `src/train_two_tower.py`, `src/train_deepfm.py` |
| Inspect API capacity | `src/observability.py`, `scripts/load_test_api.py`, `experiments/serve_load_*.json` |
| Re-run worker/overload sweep | `scripts/scaling_experiment.py --tag <unique-tag>` |
| Deploy/monitor the API | `docs/PRODUCTION_SERVING.md`, `monitoring/prometheus_rules.yml` |
| Verify full production surface | `scripts/production_smoke.py --tag <unique-tag>` |
| Export feedback and run OPE | `scripts/feedback_replay.py --tag <unique-tag> --policy-name <exact> --model-version <exact> --target-k 10` |
| Calibrate OPE against an oracle | `scripts/validate_ope_estimators.py --tag <unique-tag> --recommendations 5000` |
| Run Semantic-ID generative retrieval | `scripts/benchmark_generative_retrieval.py --tag <unique-tag> --seeds 42,43,44` |
| Collect real human feedback | `streamlit run feedback_app.py --server.port 8502` |
| Check current files/dependencies | `scripts/ai_startup_harness.py --check`, `requirements.txt` |

## Safe Commands

```powershell
.venv/Scripts/python.exe scripts/ai_startup_harness.py --check
.venv/Scripts/python.exe scripts/ai_startup_harness.py --test
.venv/Scripts/python.exe scripts/ai_startup_harness.py --release
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

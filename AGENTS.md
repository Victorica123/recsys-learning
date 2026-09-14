# Project Agent Instructions

## Scope

This is a Python 3.12 video/content recommendation application and research
project, currently implemented and evaluated with MovieLens movie ratings.
Enterprise knowledge recommendation is an optional migration experiment,
not the project's default business domain.
Use the project virtual environment at `.venv/Scripts/python.exe` and run commands from this directory.

## Startup

1. Run `.venv/Scripts/python.exe scripts/ai_startup_harness.py --check`.
2. Read `docs/AI_STARTUP_HARNESS.md` for the compact architecture and current state.
3. Read `docs/AI_HANDOFF.md` only when planning new work or resuming a previous task.
4. Open the relevant `notes/`, `research/`, or source file lazily; do not load the whole project by default.

## Engineering Rules

- Preserve existing experiment artifacts, checkpoints, figures, and user changes.
- Prefer small, reproducible changes with a focused verification command.
- Do not retrain long experiments unless the task requires it; use existing checkpoints/logs first.
- Keep data paths relative to the project root via `Path(__file__).resolve()`.
- Treat reported metrics as protocol-specific; do not compare metrics across different datasets or evaluation protocols without stating the difference.
- Avoid adding heavyweight dependencies. Update `requirements.txt` only when a dependency is required by a runnable feature.

## Verification

- Python syntax: `.venv/Scripts/python.exe scripts/ai_startup_harness.py --check`
- Smoke tests (CPU-only, no network): `.venv/Scripts/python.exe scripts/ai_startup_harness.py --test`
- Model smoke tests: run the smallest relevant script arguments (for example `train_pg_rec.py --updates 1` or `train_sasrec.py --epochs 1 --eval_every 1`).
- Main web demo: `.venv/Scripts/python.exe serve.py --preload`, open port 8000.
- Scripted recommendation/exposure/click check:
  `.venv/Scripts/python.exe scripts/demo_recommendation_flow.py --tag <unique-tag>`.
- Optional Streamlit demo: `.venv/Scripts/python.exe -m streamlit run app.py`
- REST API: `.venv/Scripts/python.exe serve.py --port 8000 --preload`, then `GET /health`.

## Human Reading Defaults

- The user's primary job target is AI full-stack application development.
  For onboarding and interviews, lead with user preferences -> content
  recommendation -> UI/API -> exposure/feedback -> evaluation. Prioritize a
  working application and service reliability; model training and RL are
  deeper topics. Do not reinterpret this job target as a request to replace
  the project's video/content domain with enterprise knowledge or RAG.
- In search/advertising/recommendation, the implemented product is the
  recommendation part. Video keyword search and ad bidding/conversion
  prediction are future work. MovieLens ratings are not video-play, ad-click
  or conversion labels; short-video migration needs its own behavior schema.
- Preserve the supplied enterprise seed and completed offline work as an
  optional scenario exercise. Synthetic data is valid for development and
  learning, with measured real-user benefit evaluated separately. Do not
  relabel enterprise events as video behavior or replace the mainline with
  this extension.
- Keep `README.md`, `项目导学.md`, and `docs/INTERVIEW_PLAYBOOK.md` concise.
  Lead with the content recommendation scenario, one run command, and the
  app/feedback flow. Introduce technical terms only when needed and explain them.
- Keep full model material in `docs/MOVIELENS_LEARNING_PATH.md`,
  `docs/INTERVIEW_DEEP_DIVE.md`, and existing notes. Preserve implementations
  and experiment evidence when simplifying reader-facing documentation.

## Task Routing

- Product/demo surface: `web/index.html` calls the versioned POST routes in
  `serve.py`, which reuse `src/serving.py`; visible-item impressions and click
  retries are part of the same page. `app.py` remains an optional Streamlit
  view calling the Python core directly, not an HTTP client of `serve.py`.
- Container verification: `Dockerfile`, `compose.yaml`,
  `scripts/production_smoke.py --base-url <url> --tag <unique-tag>` (checks the
  existing service without launching/stopping another). Load tests support
  `--method POST`. Keep scripted demo/load events distinct from real-user
  evidence and separate successful-response latency from 429 rejections.
- Exposure/feedback: `feedback_app.py`, `src/feedback.py`,
  `docs/FEEDBACK_LOOP.md`. The separate feedback page needs no API process.
- Data and baseline models: `src/explore_data.py`, `src/train_mf.py`.
- Explicit positive-rating ranking with CTR-style LR/FM/DeepFM architectures
  (MovieLens has no impression/click labels; do not call this real CTR):
  `src/train_deepfm.py`.
- Retrieval: `src/train_two_tower.py` and `src/plot_two_tower.py`.
- Exploration: `src/run_bandit.py`.
- Sequence recommendation: `src/train_sasrec.py`.
- Long-term RL: `src/train_dqn_rec.py`, `src/train_pg_rec.py`,
  `src/train_ppo_rec.py` (PPO/GAE; roadmap in `notes/14-前沿RL论文与落地路线.md`).
- Offline RL / post-training: `research_v2/phase2_offline_rl/` (SFT/behavior cloning + CQL + IQL; logs → offline training).
- Agentic tool selection: `research_v3/agentic_rec/` (cost-aware baseline/tool gate + group-relative REINFORCE).
- Reranker retraining: `src/train_reranker.py` (retrieval-aligned hard
  negatives, listwise loss, cross features, residual-from-retrieval). Load an
  alternative ranker via
  `Recommender.load(ranking_policy="deepfm", deepfm_ckpt=...)`.
  Current status: v0 is a proven end-to-end net loss and must not be served;
  the best version (v6, `--residual --oof-folds 4`) only ties the no-reranking
  baseline (CI straddles zero). The residual gate settles at 6% of the
  retrieval weight and negative - the ranker has no information the retriever
  lacks, so further ranker tuning is not the productive direction.
- Evaluation protocols: `train_two_tower.py --split {loo,time}`. `loo` is the
  primary protocol (comparable with the SASRec reproduction) but permits time
  travel; `time` has none and scores 0.069 vs 0.098. Always name the protocol
  when quoting a retrieval metric; never quote only the higher one.
- Offline evaluation beyond per-stage metrics:
  `scripts/eval_end_to_end.py` (retrieval -> ranking funnel; the served path),
  `scripts/benchmark_retrieval.py` (rolling full-catalog baselines),
  `scripts/benchmark_generative_retrieval.py` (TIGER-lite semantic IDs),
  `scripts/eval_multiroute_ranker.py` (RRF + sequence residual ranker),
  and `scripts/agentic_cost_sweep.py` / `scripts/agentic_multitool_experiment.py`
  (tool budgets with >=10-seed inference guardrails). All refuse to overwrite a
  `--tag`.
- Checkpoint loading: use `src/checkpoint_io.py::load_torch_checkpoint` for
  project checkpoints. It forbids `weights_only=False`; manifest SHA-256 is
  verified when `verify_hash=True` / `RECSYS_VERIFY_CHECKPOINTS=1` is set.
  Never add a raw `torch.load(..., weights_only=False)` back into a
  serving/evaluation path.
- Release verification: `scripts/verify_release.py --profile serve|research`
  and `scripts/ai_startup_harness.py --release`; update
  `artifacts/release_manifest.json` whenever a pinned artifact legitimately changes.
- Automated tests: `tests/` (stdlib `unittest`, CPU-only; run via harness `--test`).
  Never hardcode the test count in documentation; cite the command instead.

### Optional Enterprise Migration

- Enterprise knowledge recommendation (synthetic offline track):
  `src/enterprise_{seed,dataset,recommendation,feedback,evaluation}.py`,
  `scripts/enterprise_pilot.py --tag <unique-tag>`,
  `docs/ENTERPRISE_PILOT_GUIDE.md`, `docs/ENTERPRISE_RECOMMENDATION.md`,
  and `notes/21-企业知识推荐-种子审计与业务迁移.md`.
  Keep oracle/hidden topic fields out of ranking, enforce authorization before
  text statistics, and freeze history by actual event time. The upstream
  top-16 pools are not full-catalog retrieval. Never promote a policy from
  synthetic metrics; optional live integration belongs in the existing EIP
  Agent Service, not a third backend. Keep generic enterprise records separate
  from MovieLens feedback schema-v2. Preserve existing tagged pilot outputs.

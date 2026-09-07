# Project Agent Instructions

## Scope

This is a Python 3.12 recommendation-systems learning and research project.
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
- Demo: `.venv/Scripts/python.exe -m streamlit run app.py`
- REST API: `.venv/Scripts/python.exe serve.py --port 8000 --preload`, then `GET /health`.

## Task Routing

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
  alternative ranker anywhere via `Recommender.load(deepfm_ckpt=...)`.
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
- Product/demo surface: `app.py` (Streamlit) and `serve.py` (REST API), both reusing `src/serving.py`.
- Automated tests: `tests/` (stdlib `unittest`, CPU-only; run via harness `--test`).
  Never hardcode the test count in documentation; cite the command instead.

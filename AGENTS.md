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
- CTR ranking: `src/train_deepfm.py`.
- Retrieval: `src/train_two_tower.py` and `src/plot_two_tower.py`.
- Exploration: `src/run_bandit.py`.
- Sequence recommendation: `src/train_sasrec.py`.
- Long-term RL: `src/train_dqn_rec.py`, `src/train_pg_rec.py`,
  `src/train_ppo_rec.py` (PPO/GAE; roadmap in `notes/14-前沿RL论文与落地路线.md`).
- Offline RL / post-training: `research_v2/phase2_offline_rl/` (SFT/behavior cloning + CQL + IQL; logs → offline training).
- Agentic tool selection: `research_v3/agentic_rec/` (cost-aware baseline/tool gate + group-relative REINFORCE).
- Product/demo surface: `app.py` (Streamlit) and `serve.py` (REST API), both reusing `src/serving.py`.
- Automated tests: `tests/` (stdlib `unittest`, CPU-only; run via harness `--test`).

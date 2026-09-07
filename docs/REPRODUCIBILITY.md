# Reproducibility Contract

This repository has three separate reproducibility surfaces:

1. `pyproject.toml` declares the direct dependency contract for Python 3.12.
2. `uv.lock` freezes the complete cross-platform dependency graph, including
   hashes. `uv sync --locked` must refuse an out-of-date lock.
3. `artifacts/release_manifest.json` pins the reference MovieLens files and
   checkpoints by path, byte size, SHA-256, role, task, and release profile.

## Release profiles

- `serve`: the promoted default, requiring MovieLens-1M plus the two-tower
  checkpoint. It intentionally does not require DeepFM or SASRec.
- `research`: adds the explicit DeepFM failure reproduction, the frozen
  SASRec simulator, and the v6 OOF residual reranker used by the reranker study.

Service-side enforcement: `src/checkpoint_io.py` always loads checkpoints with
`torch.load(weights_only=True)`. Deployed services can additionally start with
`--verify-checkpoint-hashes` (or `RECSYS_VERIFY_CHECKPOINTS=1`) to verify the
manifest size and SHA-256 of declared weights before load; a tampered or stale
declared checkpoint then fails startup instead of entering the process.

Verify the local reference release:

```powershell
.venv/Scripts/python.exe scripts/verify_release.py --profile serve
.venv/Scripts/python.exe scripts/verify_release.py --profile research
```

CI has no private model/data files, so it uses `--manifest-only`. That still
validates the schema, safe relative paths, selected artifact names, Python
minor version, and the SHA-256 of `uv.lock`.

## Environment setup

CPU/default:

```powershell
uv sync --locked
```

Windows RTX 50-series / CUDA 12.8:

```powershell
uv sync --locked --no-install-package torch
.venv/Scripts/python.exe -m pip install `
  --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128 torch==2.9.1
```

The optional LunarLander exercise is isolated because Box2D is not required by
the recommender or CI:

```powershell
uv sync --locked --extra lander
```

## Interpretation limits

- Matching a checkpoint hash reproduces the reference artifact exactly.
- Retraining with the same seed is a separate claim: GPU kernels and upstream
  libraries can still be nondeterministic. A new checkpoint must receive a new
  manifest hash/release ID rather than silently replacing the reference.
- The DeepFM checkpoint is a positive-rating classifier conditional on an
  observed MovieLens rating. It is not a CTR model and is not in `serve`.
- Existing experiment files are immutable evidence. New runs use new tags and
  do not overwrite an old artifact.

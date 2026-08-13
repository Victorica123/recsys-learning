# -*- coding: utf-8 -*-
"""Standard-library smoke tests for the recsys-learning stack.

Run all tests (no GPU, no network, no external services required):

    .venv/Scripts/python.exe -m unittest discover -s tests -v

Importing this package puts ``src/`` on ``sys.path`` so the test modules can
``import train_deepfm`` / ``train_sasrec`` / ``train_dqn_rec`` the same way the
scripts import their siblings.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

"""EEG motor imagery classification package."""

from pathlib import Path as _Path
import sys as _sys

PROJECT_ROOT = _Path(__file__).resolve().parent.parent
CODE_DIR = _Path(__file__).resolve().parent

for _p in (PROJECT_ROOT, CODE_DIR):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

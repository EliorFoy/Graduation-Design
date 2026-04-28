"""CLI entrypoint for single-subject training with leakage-safe CV."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, CODE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import argparse

from code.general_process import single_subject_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train EEG motor imagery classifiers for one training session.")
    parser.add_argument("--subject", default="A01T", help="Subject/session id, e.g. A01T")
    parser.add_argument("--data-root", default=None, help="Directory containing BCIC IV 2a GDF files")
    args = parser.parse_args()
    single_subject_pipeline(subject_id=args.subject, data_root=args.data_root)


if __name__ == "__main__":
    main()


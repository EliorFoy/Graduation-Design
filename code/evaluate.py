"""CLI entrypoint for train-on-T and evaluate-on-E workflow."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, CODE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import argparse

from code.test_on_evaluation_set import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate EEG classifier on an independent evaluation session.")
    parser.add_argument("--train-subject", default="A01T", help="Training subject/session id, e.g. A01T")
    parser.add_argument("--eval-subject", default="A01E", help="Evaluation subject/session id, e.g. A01E")
    parser.add_argument("--data-root", default=None, help="Directory containing BCIC IV 2a GDF files")
    args = parser.parse_args()
    train_and_evaluate(
        train_subject=args.train_subject,
        eval_subject=args.eval_subject,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()


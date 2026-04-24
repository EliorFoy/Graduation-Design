"""CLI entrypoint for batch preprocessing BCI IV 2a sessions."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, CODE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import argparse

from code.batch_processing import batch_process_subjects


def main():
    parser = argparse.ArgumentParser(description="Batch preprocess EEG sessions.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject ids without session suffix, e.g. A01 A02")
    parser.add_argument("--sessions", nargs="*", default=None, help="Session suffixes, e.g. T E")
    parser.add_argument("--data-root", default=None, help="Directory containing BCIC IV 2a GDF files")
    args = parser.parse_args()
    batch_process_subjects(subject_ids=args.subjects, sessions=args.sessions, data_root=args.data_root)


if __name__ == "__main__":
    main()


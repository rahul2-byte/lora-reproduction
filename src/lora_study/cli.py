"""Command-line entry point for one configured run."""

from __future__ import annotations

import argparse
import logging

from .config import load_config
from .runner import run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()
    try:
        run(load_config(args.config), args.output_root, args.resume_checkpoint)
    except Exception as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

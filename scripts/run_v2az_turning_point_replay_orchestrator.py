#!/usr/bin/env python3
"""CLI wrapper for the v2az turning-point replay orchestrator."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v2az_turning_point_replay_orchestrator as engine  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry_run", "replay"))
    args = parser.parse_args(argv)
    code, _ = engine.run(mode=args.mode)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

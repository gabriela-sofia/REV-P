#!/usr/bin/env python3
"""Run v2bb public geometry retrieval and feed builder."""

import argparse

from v2bb_public_geometry_retrieval_feed_builder import MODES, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="search_plan")
    args = parser.parse_args()
    code, _ = run(args.mode)
    raise SystemExit(code)

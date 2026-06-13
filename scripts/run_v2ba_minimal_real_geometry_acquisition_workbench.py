#!/usr/bin/env python3
"""Run the v2ba minimal real geometry acquisition workbench."""

from __future__ import annotations

import argparse

from v2ba_minimal_real_geometry_acquisition_workbench import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("source_scan", "ingest", "validate", "replay_ready_check"),
        default="source_scan",
    )
    args = parser.parse_args()
    code, _ = run(args.mode)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

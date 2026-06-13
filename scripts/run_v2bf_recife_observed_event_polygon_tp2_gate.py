#!/usr/bin/env python3
"""Run v2bf Recife observed-event polygon retrieval and TP2 gate."""

import argparse

from v2bf_recife_observed_event_polygon_tp2_gate import MODES, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="full")
    raise SystemExit(run(mode=parser.parse_args().mode)[0])

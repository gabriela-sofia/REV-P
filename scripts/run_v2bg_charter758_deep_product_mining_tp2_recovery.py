#!/usr/bin/env python3
"""Run v2bg Charter 758 deep product mining and TP2 recovery."""

import argparse

from v2bg_charter758_deep_product_mining_tp2_recovery import MODES, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="full")
    raise SystemExit(run(mode=parser.parse_args().mode)[0])

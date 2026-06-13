#!/usr/bin/env python3
"""Run v2bh Charter 758 Recife product georeferencing and digitization."""

import argparse

from v2bh_charter758_recife_product_georeferencing_digitization import MODES, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="full")
    args = parser.parse_args()
    run(mode=args.mode)

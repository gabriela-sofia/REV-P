#!/usr/bin/env python3
"""Thin wrapper for the v2ax Recife Geometry Intake Pack engine."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v2ax_recife_geometry_intake_pack_engine as engine  # noqa: E402


def main():
    code, _ = engine.run()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

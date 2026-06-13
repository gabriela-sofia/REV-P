#!/usr/bin/env python3
"""Thin wrapper for the v2ay event-scope reconciliation engine."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v2ay_event_scope_reconciliation_turning_point_engine as engine  # noqa: E402


def main():
    code, _ = engine.run()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

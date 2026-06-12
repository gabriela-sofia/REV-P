#!/usr/bin/env python3
"""Thin wrapper for the v2at Evidence Registry + Event-Patch Package Engine.

Delegates to ``v2at_evidence_registry_event_patch_engine.run``. Honours the
DATASET_DIR / OUTPUT_DIR / CONFIG_DIR environment variables, falling back to the
repository defaults (datasets/, outputs_public/, configs/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v2at_evidence_registry_event_patch_engine as engine  # noqa: E402


def main():
    code, _ = engine.run()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

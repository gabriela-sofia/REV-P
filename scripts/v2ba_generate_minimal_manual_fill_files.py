#!/usr/bin/env python3
"""Generate the two v2ba assisted manual-fill files without inventing geometry."""

from __future__ import annotations

from v2ba_minimal_real_geometry_acquisition_workbench import generate_fill_files, resolve_dirs


if __name__ == "__main__":
    dirs = resolve_dirs()
    paths = generate_fill_files(dirs["external_dir"])
    for path in paths:
        print(f"[v2ba] wrote={path}")

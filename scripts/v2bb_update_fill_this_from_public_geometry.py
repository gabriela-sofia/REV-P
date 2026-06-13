#!/usr/bin/env python3
"""Create review-only autofill candidates from valid v2bb public feeds."""

from v2bb_public_geometry_retrieval_feed_builder import update_autofill, resolve_dirs


if __name__ == "__main__":
    paths = update_autofill(resolve_dirs())
    print(f"[v2bb] autofill_candidates={len(paths)}")

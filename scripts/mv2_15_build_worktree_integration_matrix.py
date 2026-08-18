from __future__ import annotations

from pathlib import Path

from mv2_15_cronograma_common import OUTPUT_DIR, PROJECT_ROOT, build_worktree_integration_matrix, ensure_dirs


def main() -> None:
    ensure_dirs()
    rows = build_worktree_integration_matrix()
    print(f"[mv2_15_worktrees] worktrees={len(rows)}")
    print(f"[mv2_15_worktrees] output={(OUTPUT_DIR / 'mv2_15_worktree_integration_matrix.csv').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys

from mv2_15_cronograma_common import build_all, print_validation_result, status_counts


def main() -> int:
    build_all()
    counts = status_counts()
    print(f"[mv2_15_run] schedule_status={counts}")
    rc = print_validation_result()
    if rc != 0:
        return rc
    print("[mv2_15_run] cronograma engine concluido com guardrails preservados")
    return 0


if __name__ == "__main__":
    sys.exit(main())

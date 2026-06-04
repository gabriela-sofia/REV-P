#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uu_recife_common import run_non_occurrence_guard_builder
except ModuleNotFoundError:
    from revp_v1uu_recife_common import run_non_occurrence_guard_builder


if __name__ == "__main__":
    run_non_occurrence_guard_builder()

#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ut_recife_common import run_coordinate_row_join_audit
except ModuleNotFoundError:
    from revp_v1ut_recife_common import run_coordinate_row_join_audit


if __name__ == "__main__":
    run_coordinate_row_join_audit()

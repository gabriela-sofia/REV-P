#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uu_recife_common import run_out_of_window_occurrence_coordinate_audit
except ModuleNotFoundError:
    from revp_v1uu_recife_common import run_out_of_window_occurrence_coordinate_audit


if __name__ == "__main__":
    run_out_of_window_occurrence_coordinate_audit()

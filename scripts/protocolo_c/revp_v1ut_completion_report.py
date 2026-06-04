#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ut_recife_common import run_completion_report
except ModuleNotFoundError:
    from revp_v1ut_recife_common import run_completion_report


if __name__ == "__main__":
    run_completion_report()

#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uk_recife_common import run_completion_report, simple_main
except ModuleNotFoundError:
    from revp_v1uk_recife_common import run_completion_report, simple_main


if __name__ == "__main__":
    simple_main(run_completion_report)

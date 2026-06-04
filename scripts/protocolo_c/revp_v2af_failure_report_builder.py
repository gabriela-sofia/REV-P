#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2af_common import parse_args, run_failure_report_builder
except ModuleNotFoundError:
    from revp_v2af_common import parse_args, run_failure_report_builder


if __name__ == "__main__":
    run_failure_report_builder(parse_args())

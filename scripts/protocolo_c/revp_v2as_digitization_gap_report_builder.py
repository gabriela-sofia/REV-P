#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2as_common import parse_args, run_digitization_gap_report_builder
except ModuleNotFoundError:
    from revp_v2as_common import parse_args, run_digitization_gap_report_builder


if __name__ == "__main__":
    run_digitization_gap_report_builder(parse_args())

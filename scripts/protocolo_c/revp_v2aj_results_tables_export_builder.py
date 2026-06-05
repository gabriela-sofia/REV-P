#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aj_common import parse_args, run_results_tables_export_builder
except ModuleNotFoundError:
    from revp_v2aj_common import parse_args, run_results_tables_export_builder


if __name__ == "__main__":
    run_results_tables_export_builder(parse_args())

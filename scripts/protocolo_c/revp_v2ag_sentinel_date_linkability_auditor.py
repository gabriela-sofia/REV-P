#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ag_common import parse_args, run_sentinel_date_linkability_auditor
except ModuleNotFoundError:
    from revp_v2ag_common import parse_args, run_sentinel_date_linkability_auditor


if __name__ == "__main__":
    run_sentinel_date_linkability_auditor(parse_args())

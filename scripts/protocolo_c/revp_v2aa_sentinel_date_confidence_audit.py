#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aa_common import parse_args, run_sentinel_date_confidence_audit
except ModuleNotFoundError:
    from revp_v2aa_common import parse_args, run_sentinel_date_confidence_audit


if __name__ == "__main__":
    run_sentinel_date_confidence_audit(parse_args())

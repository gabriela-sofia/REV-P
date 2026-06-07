#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2an_common import parse_args, run_temporal_sentinel_crosswalk_audit
except ModuleNotFoundError:
    from revp_v2an_common import parse_args, run_temporal_sentinel_crosswalk_audit


if __name__ == "__main__":
    run_temporal_sentinel_crosswalk_audit(parse_args())

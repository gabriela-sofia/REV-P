#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2at_common import parse_args, run_audit_temporal_alignment
except ModuleNotFoundError:
    from revp_v2at_common import parse_args, run_audit_temporal_alignment
if __name__ == "__main__":
    run_audit_temporal_alignment(parse_args())

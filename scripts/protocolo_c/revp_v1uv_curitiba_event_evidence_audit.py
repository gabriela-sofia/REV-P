#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uv_curitiba_common import parse_args, run_event_evidence_audit
except ModuleNotFoundError:
    from revp_v1uv_curitiba_common import parse_args, run_event_evidence_audit


if __name__ == "__main__":
    run_event_evidence_audit(parse_args())

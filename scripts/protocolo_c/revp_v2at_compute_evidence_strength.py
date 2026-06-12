#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2at_common import parse_args, run_compute_evidence_strength
except ModuleNotFoundError:
    from revp_v2at_common import parse_args, run_compute_evidence_strength
if __name__ == "__main__":
    run_compute_evidence_strength(parse_args())

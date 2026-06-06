#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2al_common import parse_args, run_claim_alignment_audit_builder
except ModuleNotFoundError:
    from revp_v2al_common import parse_args, run_claim_alignment_audit_builder


if __name__ == "__main__":
    run_claim_alignment_audit_builder(parse_args())

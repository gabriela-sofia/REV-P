#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2am_common import parse_args, run_final_claim_consistency_audit
except ModuleNotFoundError:
    from revp_v2am_common import parse_args, run_final_claim_consistency_audit


if __name__ == "__main__":
    run_final_claim_consistency_audit(parse_args())

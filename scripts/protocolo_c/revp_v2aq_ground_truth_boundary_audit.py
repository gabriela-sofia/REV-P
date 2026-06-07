#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aq_common import parse_args, run_ground_truth_boundary_audit
except ModuleNotFoundError:
    from revp_v2aq_common import parse_args, run_ground_truth_boundary_audit


if __name__ == "__main__":
    run_ground_truth_boundary_audit(parse_args())

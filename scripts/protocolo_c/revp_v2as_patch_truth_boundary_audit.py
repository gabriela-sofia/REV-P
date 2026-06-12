#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2as_common import parse_args, run_patch_truth_boundary_audit
except ModuleNotFoundError:
    from revp_v2as_common import parse_args, run_patch_truth_boundary_audit


if __name__ == "__main__":
    run_patch_truth_boundary_audit(parse_args())

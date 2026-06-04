#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ag_common import parse_args, run_crosswalk_evidence_strength_auditor
except ModuleNotFoundError:
    from revp_v2ag_common import parse_args, run_crosswalk_evidence_strength_auditor


if __name__ == "__main__":
    run_crosswalk_evidence_strength_auditor(parse_args())

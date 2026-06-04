#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ux_curitiba_common import parse_args, run_candidate_evidence_classifier
except ModuleNotFoundError:
    from revp_v1ux_curitiba_common import parse_args, run_candidate_evidence_classifier


if __name__ == "__main__":
    run_candidate_evidence_classifier(parse_args())

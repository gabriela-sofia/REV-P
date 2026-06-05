#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ah_common import parse_args, run_candidate_evidence_dossier_index
except ModuleNotFoundError:
    from revp_v2ah_common import parse_args, run_candidate_evidence_dossier_index


if __name__ == "__main__":
    run_candidate_evidence_dossier_index(parse_args())

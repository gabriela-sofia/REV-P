#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1un_recife_common import run_human_review_evidence_consolidator, simple_main
except ModuleNotFoundError:
    from revp_v1un_recife_common import run_human_review_evidence_consolidator, simple_main

if __name__ == "__main__":
    simple_main(run_human_review_evidence_consolidator)

#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ag_common import parse_args, run_lineage_crosswalk_candidate_builder
except ModuleNotFoundError:
    from revp_v2ag_common import parse_args, run_lineage_crosswalk_candidate_builder


if __name__ == "__main__":
    run_lineage_crosswalk_candidate_builder(parse_args())

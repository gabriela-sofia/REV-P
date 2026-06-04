#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uo_multiregion_common import run_ground_truth_opportunity_ranker, simple_main
except ModuleNotFoundError:
    from revp_v1uo_multiregion_common import run_ground_truth_opportunity_ranker, simple_main

if __name__ == "__main__":
    simple_main(run_ground_truth_opportunity_ranker)

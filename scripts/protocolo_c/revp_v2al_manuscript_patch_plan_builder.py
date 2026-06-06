#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2al_common import parse_args, run_manuscript_patch_plan_builder
except ModuleNotFoundError:
    from revp_v2al_common import parse_args, run_manuscript_patch_plan_builder


if __name__ == "__main__":
    run_manuscript_patch_plan_builder(parse_args())

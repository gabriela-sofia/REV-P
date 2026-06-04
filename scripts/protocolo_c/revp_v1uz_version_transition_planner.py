#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uz_common import parse_args, run_version_transition_planner
except ModuleNotFoundError:
    from revp_v1uz_common import parse_args, run_version_transition_planner


if __name__ == "__main__":
    run_version_transition_planner(parse_args())

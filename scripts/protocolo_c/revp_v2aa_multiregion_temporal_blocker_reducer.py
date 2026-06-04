#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aa_common import parse_args, run_multiregion_temporal_blocker_reducer
except ModuleNotFoundError:
    from revp_v2aa_common import parse_args, run_multiregion_temporal_blocker_reducer


if __name__ == "__main__":
    run_multiregion_temporal_blocker_reducer(parse_args())

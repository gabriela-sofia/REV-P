#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2as_common import parse_args, run_geometry_candidate_classifier
except ModuleNotFoundError:
    from revp_v2as_common import parse_args, run_geometry_candidate_classifier


if __name__ == "__main__":
    run_geometry_candidate_classifier(parse_args())

#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ai_common import parse_args, run_review_outcome_registry_builder
except ModuleNotFoundError:
    from revp_v2ai_common import parse_args, run_review_outcome_registry_builder


if __name__ == "__main__":
    run_review_outcome_registry_builder(parse_args())

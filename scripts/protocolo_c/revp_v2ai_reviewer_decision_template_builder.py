#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ai_common import parse_args, run_reviewer_decision_template_builder
except ModuleNotFoundError:
    from revp_v2ai_common import parse_args, run_reviewer_decision_template_builder


if __name__ == "__main__":
    run_reviewer_decision_template_builder(parse_args())

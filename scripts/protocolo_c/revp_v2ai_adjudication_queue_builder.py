#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ai_common import parse_args, run_adjudication_queue_builder
except ModuleNotFoundError:
    from revp_v2ai_common import parse_args, run_adjudication_queue_builder


if __name__ == "__main__":
    run_adjudication_queue_builder(parse_args())

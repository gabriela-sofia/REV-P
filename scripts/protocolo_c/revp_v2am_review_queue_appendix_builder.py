#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2am_common import parse_args, run_review_queue_appendix_builder
except ModuleNotFoundError:
    from revp_v2am_common import parse_args, run_review_queue_appendix_builder


if __name__ == "__main__":
    run_review_queue_appendix_builder(parse_args())

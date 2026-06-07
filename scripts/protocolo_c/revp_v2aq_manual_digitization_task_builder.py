#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aq_common import parse_args, run_manual_digitization_task_builder
except ModuleNotFoundError:
    from revp_v2aq_common import parse_args, run_manual_digitization_task_builder


if __name__ == "__main__":
    run_manual_digitization_task_builder(parse_args())

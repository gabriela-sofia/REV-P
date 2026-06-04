#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uu_recife_common import run_next_action_ranker
except ModuleNotFoundError:
    from revp_v1uu_recife_common import run_next_action_ranker


if __name__ == "__main__":
    run_next_action_ranker()

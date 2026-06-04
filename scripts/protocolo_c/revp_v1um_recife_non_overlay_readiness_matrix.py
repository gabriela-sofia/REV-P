#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1um_recife_common import run_non_overlay_readiness_matrix, simple_main
except ModuleNotFoundError:
    from revp_v1um_recife_common import run_non_overlay_readiness_matrix, simple_main


if __name__ == "__main__":
    simple_main(run_non_overlay_readiness_matrix)

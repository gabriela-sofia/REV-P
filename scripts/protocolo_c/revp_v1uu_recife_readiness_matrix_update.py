#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uu_recife_common import run_readiness_matrix_update
except ModuleNotFoundError:
    from revp_v1uu_recife_common import run_readiness_matrix_update


if __name__ == "__main__":
    run_readiness_matrix_update()

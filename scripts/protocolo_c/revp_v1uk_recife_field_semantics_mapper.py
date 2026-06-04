#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uk_recife_common import run_field_semantics, simple_main
except ModuleNotFoundError:
    from revp_v1uk_recife_common import run_field_semantics, simple_main


if __name__ == "__main__":
    simple_main(run_field_semantics)

#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ul_recife_common import run_sensitive_review_packager, simple_main
except ModuleNotFoundError:
    from revp_v1ul_recife_common import run_sensitive_review_packager, simple_main


if __name__ == "__main__":
    simple_main(run_sensitive_review_packager)

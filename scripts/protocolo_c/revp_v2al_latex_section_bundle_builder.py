#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2al_common import parse_args, run_latex_section_bundle_builder
except ModuleNotFoundError:
    from revp_v2al_common import parse_args, run_latex_section_bundle_builder


if __name__ == "__main__":
    run_latex_section_bundle_builder(parse_args())

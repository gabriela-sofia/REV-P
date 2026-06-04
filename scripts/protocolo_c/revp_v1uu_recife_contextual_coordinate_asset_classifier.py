#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uu_recife_common import run_contextual_coordinate_asset_classifier
except ModuleNotFoundError:
    from revp_v1uu_recife_common import run_contextual_coordinate_asset_classifier


if __name__ == "__main__":
    run_contextual_coordinate_asset_classifier()

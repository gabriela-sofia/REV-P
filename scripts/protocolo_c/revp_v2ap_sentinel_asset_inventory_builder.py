#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ap_common import parse_args, run_sentinel_asset_inventory_builder
except ModuleNotFoundError:
    from revp_v2ap_common import parse_args, run_sentinel_asset_inventory_builder


if __name__ == "__main__":
    run_sentinel_asset_inventory_builder(parse_args())

#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ux_curitiba_common import parse_args, run_public_artifact_downloader
except ModuleNotFoundError:
    from revp_v1ux_curitiba_common import parse_args, run_public_artifact_downloader


if __name__ == "__main__":
    run_public_artifact_downloader(parse_args())

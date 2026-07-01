from __future__ import annotations

import argparse

import susc_17c24_multiscene_sentinel_common as s17c24


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C24 serie temporal pre-evento multi-cena Sentinel-2")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("select-scenes")
    sub.add_parser("materialize-multiscene")
    sub.add_parser("compute-scene-features")
    sub.add_parser("build-cloud-robust-matrix")
    sub.add_parser("build-offline")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.mode == "select-scenes":
        print(s17c24.select_scenes_text())
        return 0
    if args.mode == "materialize-multiscene":
        code, text = s17c24.materialize_multiscene_text()
        print(text)
        return code
    if args.mode == "compute-scene-features":
        print(s17c24.compute_scene_features_text())
        return 0
    if args.mode == "build-cloud-robust-matrix":
        print(s17c24.build_cloud_robust_matrix_text())
        return 0
    if args.mode == "build-offline":
        s17c24.build_all()
        print("build-offline: outputs 17C24 reconstruidos a partir dos artefatos locais.")
        return 0
    if args.mode == "audit":
        return s17c24.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

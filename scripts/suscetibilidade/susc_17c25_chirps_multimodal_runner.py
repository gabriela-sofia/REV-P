from __future__ import annotations

import argparse

import susc_17c25_chirps_multimodal_common as s17c25


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C25 CHIRPS real por AOI e matriz multimodal review-only")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("plan")
    sub.add_parser("resolve-chirps-source")
    sub.add_parser("fetch-chirps-real")
    sub.add_parser("compute-rainfall-windows")
    sub.add_parser("build-multimodal-matrix")
    sub.add_parser("build-offline")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.mode == "plan":
        print(s17c25.plan_text())
        return 0
    if args.mode == "resolve-chirps-source":
        print(s17c25.resolve_chirps_source_text())
        return 0
    if args.mode == "fetch-chirps-real":
        code, text = s17c25.fetch_chirps_real_text()
        print(text)
        return code
    if args.mode == "compute-rainfall-windows":
        print(s17c25.compute_rainfall_windows_text())
        return 0
    if args.mode == "build-multimodal-matrix":
        print(s17c25.build_multimodal_matrix_text())
        return 0
    if args.mode == "build-offline":
        s17c25.build_all()
        print("build-offline: outputs 17C25 reconstruidos a partir dos artefatos locais.")
        return 0
    if args.mode == "audit":
        return s17c25.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

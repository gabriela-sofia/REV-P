from __future__ import annotations

import argparse

import susc_17c19_temporal_scene_common as s17c19


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C19 consolidacao temporal e selecao canonica de cenas Sentinel-2")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("build-temporal-lineage")
    sub.add_parser("select-scenes")
    sub.add_parser("plan-light-tile")
    sub.add_parser("plan-chirps-runtime")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.mode == "build-temporal-lineage":
        print(s17c19.build_temporal_lineage_text())
        return 0
    if args.mode == "select-scenes":
        print(s17c19.select_scenes_text())
        return 0
    if args.mode == "plan-light-tile":
        print(s17c19.plan_light_tile_text())
        return 0
    if args.mode == "plan-chirps-runtime":
        print(s17c19.plan_chirps_runtime_text())
        return 0
    if args.mode == "audit":
        return s17c19.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

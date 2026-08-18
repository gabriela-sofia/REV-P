from __future__ import annotations

import argparse

import susc_17c17_sensor_snapshot_common as s17c17


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C17 snapshot de rede para metadados sensoriais")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("plan")
    sub.add_parser("capture-snapshot")
    sub.add_parser("replay-snapshot")
    sub.add_parser("plan-light-raster")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.mode == "plan":
        print(s17c17.plan_text())
        return 0
    if args.mode == "capture-snapshot":
        code, text = s17c17.capture_snapshot_text()
        print(text)
        return code
    if args.mode == "replay-snapshot":
        print(s17c17.replay_snapshot_text())
        return 0
    if args.mode == "plan-light-raster":
        print(s17c17.plan_light_raster_text())
        return 0
    if args.mode == "audit":
        return s17c17.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse

import susc_17c21_sensor_qa_compare_common as s17c21


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C21 QA sensorial e comparacao pre/pos-evento Sentinel-2")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("qa-pre-event")
    sub.add_parser("select-post-event")
    sub.add_parser("materialize-post-event")
    sub.add_parser("compute-delta")
    sub.add_parser("build-offline")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.mode == "qa-pre-event":
        print(s17c21.qa_pre_event_text())
        return 0
    if args.mode == "select-post-event":
        print(s17c21.select_post_event_text())
        return 0
    if args.mode == "materialize-post-event":
        code, text = s17c21.materialize_post_event_text()
        print(text)
        return code
    if args.mode == "compute-delta":
        code, text = s17c21.compute_delta_text()
        print(text)
        return code
    if args.mode == "build-offline":
        s17c21.build_all()
        print("build-offline: outputs 17C21 reconstruidos a partir dos artefatos locais.")
        return 0
    if args.mode == "audit":
        return s17c21.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

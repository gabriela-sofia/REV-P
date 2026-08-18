from __future__ import annotations

import argparse

import susc_17c27_ground_reference_acquisition_common as s17c27


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C27 aquisicao dirigida de artefatos de evento observado para G4/G5")
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ["plan-acquisition", "run-public-acquisition", "parse-artifacts", "extract-event-candidates",
                 "evaluate-g4-g5", "build-evidence-update", "build-offline", "audit"]:
        sub.add_parser(mode)
    args = parser.parse_args()

    if args.mode == "plan-acquisition":
        print(s17c27.plan_acquisition_text())
        return 0
    if args.mode == "run-public-acquisition":
        code, text = s17c27.run_public_acquisition_text()
        print(text)
        return code
    if args.mode == "parse-artifacts":
        print(s17c27.parse_artifacts_text())
        return 0
    if args.mode == "extract-event-candidates":
        print(s17c27.extract_event_candidates_text())
        return 0
    if args.mode == "evaluate-g4-g5":
        print(s17c27.evaluate_g4_g5_text())
        return 0
    if args.mode == "build-evidence-update":
        print(s17c27.build_evidence_update_text())
        return 0
    if args.mode == "build-offline":
        s17c27.build_all()
        print("build-offline: outputs 17C27 reconstruidos a partir dos artefatos adquiridos.")
        return 0
    if args.mode == "audit":
        return s17c27.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

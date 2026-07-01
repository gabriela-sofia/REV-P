from __future__ import annotations

import argparse

import susc_17c28_deep_ground_reference_common as s17c28


def _rebuild_and_print(text: str) -> int:
    s17c28.build_all()
    print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C28 aquisicao profunda de artefatos oficiais para G4/G5")
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in [
        "expand-search-plan",
        "run-deep-acquisition",
        "follow-source-links",
        "parse-deep-artifacts",
        "extract-specific-event-candidates",
        "resolve-locations",
        "classify-phenomena",
        "evaluate-g4-g5",
        "build-evidence-update",
        "build-offline",
        "status",
        "audit",
    ]:
        sub.add_parser(mode)
    args = parser.parse_args()

    if args.mode == "expand-search-plan":
        return _rebuild_and_print(s17c28.expand_search_plan_text())
    if args.mode == "run-deep-acquisition":
        code, text = s17c28.run_deep_acquisition_text()
        print(text)
        return code
    if args.mode == "follow-source-links":
        code, text = s17c28.follow_source_links_text()
        print(text)
        return code
    if args.mode == "parse-deep-artifacts":
        return _rebuild_and_print(s17c28.parse_deep_artifacts_text())
    if args.mode == "extract-specific-event-candidates":
        return _rebuild_and_print(s17c28.extract_specific_event_candidates_text())
    if args.mode == "resolve-locations":
        return _rebuild_and_print(s17c28.resolve_locations_text())
    if args.mode == "classify-phenomena":
        return _rebuild_and_print(s17c28.classify_phenomena_text())
    if args.mode == "evaluate-g4-g5":
        return _rebuild_and_print(s17c28.evaluate_g4_g5_text())
    if args.mode == "build-evidence-update":
        return _rebuild_and_print(s17c28.build_evidence_update_text())
    if args.mode == "build-offline":
        s17c28.build_all()
        print("build-offline: outputs 17C28 reconstruidos a partir dos artefatos adquiridos.")
        return 0
    if args.mode == "status":
        print(s17c28.status_text())
        return 0
    if args.mode == "audit":
        return s17c28.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

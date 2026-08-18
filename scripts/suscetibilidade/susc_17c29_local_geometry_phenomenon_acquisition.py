from __future__ import annotations

import argparse

import susc_17c29_local_geometry_phenomenon_common as s17c29


def _rebuild_and_print(text: str) -> int:
    s17c29.build_all()
    print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C29 aquisicao de geometria local e separacao de fenomeno para G4/G5")
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in [
        "expand-local-search",
        "run-local-acquisition",
        "follow-local-links",
        "parse-local-artifacts",
        "extract-local-event-candidates",
        "resolve-local-geometry",
        "classify-local-phenomena",
        "evaluate-g4-g5",
        "build-offline",
        "status",
        "audit",
    ]:
        sub.add_parser(mode)
    args = parser.parse_args()

    if args.mode == "expand-local-search":
        return _rebuild_and_print(s17c29.expand_local_search_text())
    if args.mode == "run-local-acquisition":
        code, text = s17c29.run_local_acquisition_text()
        print(text)
        return code
    if args.mode == "follow-local-links":
        code, text = s17c29.follow_local_links_text()
        print(text)
        return code
    if args.mode == "parse-local-artifacts":
        return _rebuild_and_print(s17c29.parse_local_artifacts_text())
    if args.mode == "extract-local-event-candidates":
        return _rebuild_and_print(s17c29.extract_local_event_candidates_text())
    if args.mode == "resolve-local-geometry":
        return _rebuild_and_print(s17c29.resolve_local_geometry_text())
    if args.mode == "classify-local-phenomena":
        return _rebuild_and_print(s17c29.classify_local_phenomena_text())
    if args.mode == "evaluate-g4-g5":
        return _rebuild_and_print(s17c29.evaluate_g4_g5_text())
    if args.mode == "build-offline":
        s17c29.build_all()
        print("build-offline: outputs 17C29 reconstruidos a partir dos artefatos locais adquiridos.")
        return 0
    if args.mode == "status":
        print(s17c29.status_text())
        return 0
    if args.mode == "audit":
        return s17c29.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

import susc_17c14_official_collection_common as s17c14


def _bool_arg(value: str) -> bool:
    if value.lower() == "false":
        return False
    if value.lower() == "true":
        return True
    raise argparse.ArgumentTypeError("use true ou false")


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C14 agente seguro de coleta oficial e intake condicionado")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("queue")
    collect = sub.add_parser("collect")
    collect.add_argument("--request-id")
    collect.add_argument("--dry-run", type=_bool_arg, default=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--request-id", required=True)
    submit.add_argument("--dry-run", type=_bool_arg, default=True)
    submit.add_argument("--confirm-authorized", action="store_true")
    intake = sub.add_parser("intake")
    intake.add_argument("--request-id", required=True)
    intake.add_argument("--artifact-path", required=True)
    validate = sub.add_parser("validate-artifact")
    validate.add_argument("--request-id", required=True)
    validate.add_argument("--artifact-path", required=True)
    sub.add_parser("status")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.mode == "queue":
        print(s17c14.queue_text())
        return 0
    if args.mode == "collect":
        print(s17c14.collect_text(args.request_id, args.dry_run))
        return 0
    if args.mode == "submit":
        code, text = s17c14.submit_text(args.request_id, args.dry_run, args.confirm_authorized)
        print(text)
        return code
    if args.mode == "intake":
        print(s17c14.intake_file(args.request_id, Path(args.artifact_path)))
        return 0
    if args.mode == "validate-artifact":
        print(s17c14.intake_file(args.request_id, Path(args.artifact_path)))
        return 0
    if args.mode == "status":
        print(s17c14.queue_text())
        return 0
    if args.mode == "audit":
        return s17c14.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

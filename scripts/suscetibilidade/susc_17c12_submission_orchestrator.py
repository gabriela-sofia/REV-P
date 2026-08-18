from __future__ import annotations

import argparse
from pathlib import Path

import susc_17c12_submission_orchestrator_common as s17c12


def _bool_arg(value: str) -> bool:
    if value.lower() == "false":
        return False
    if value.lower() == "true":
        return True
    raise argparse.ArgumentTypeError("use true ou false")


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C12 orquestrador seguro de submissao assistida")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("plan")
    prep = sub.add_parser("prepare")
    prep.add_argument("--request-id", required=True)
    open_channel = sub.add_parser("open-channel")
    open_channel.add_argument("--request-id", required=True)
    submit = sub.add_parser("submit-assisted")
    submit.add_argument("--request-id", required=True)
    submit.add_argument("--dry-run", type=_bool_arg, default=True)
    submit.add_argument("--confirm-human-reviewed", action="store_true")
    record = sub.add_parser("record-submission")
    record.add_argument("--request-id", required=True)
    record.add_argument("--submitted-at", required=True)
    record.add_argument("--channel-id", required=True)
    record.add_argument("--evidence-note", required=True)
    record.add_argument("--protocol-number", default="not_informed")
    intake = sub.add_parser("intake-response")
    intake.add_argument("--request-id", required=True)
    intake.add_argument("--response-file", required=True)
    intake.add_argument("--received-at", required=True)
    sub.add_parser("status")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.mode == "plan":
        print(s17c12.plan_text())
        return 0
    if args.mode == "prepare":
        print(s17c12.prepare_request(args.request_id))
        return 0
    if args.mode == "open-channel":
        print(s17c12.open_channel_instruction(args.request_id))
        return 0
    if args.mode == "submit-assisted":
        status, message = s17c12.submit_assisted(args.request_id, args.dry_run, args.confirm_human_reviewed)
        print(f"{status}: {message}")
        return 0 if status in {"dry_run_only", "blocked_manual_portal_required"} else 2
    if args.mode == "record-submission":
        print(s17c12.record_manual_submission(args.request_id, args.submitted_at, args.channel_id, args.evidence_note, args.protocol_number))
        return 0
    if args.mode == "intake-response":
        print(s17c12.intake_response(args.request_id, Path(args.response_file), args.received_at))
        return 0
    if args.mode == "status":
        print(s17c12.status_text())
        return 0
    if args.mode == "audit":
        return s17c12.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

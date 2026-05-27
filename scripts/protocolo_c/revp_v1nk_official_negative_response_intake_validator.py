"""REV-P v1nk - manual official response intake validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from revp_v1ni_v1nn_common import (
    DATASETS,
    DOCS,
    INBOX,
    SCHEMAS,
    lightweight_text,
    prevalidate_text,
    public_inbox_label,
    sha256_file,
    write_doc,
    write_outputs,
)


OUT_INTAKE = DATASETS / "official_negative_response_intake_registry.csv"
OUT_PREVALIDATION = DATASETS / "official_negative_response_prevalidation_matrix.csv"
SCHEMA_INTAKE = SCHEMAS / "official_negative_response_intake_schema.csv"
SCHEMA_PREVALIDATION = SCHEMAS / "official_negative_response_prevalidation_schema.csv"
DOC = DOCS / "protocolo_c_intake_respostas_oficiais_negativas_v1nk.md"

INTAKE_FIELDS = ["response_id", "inbox_file", "file_type", "sha256", "byte_count", "raw_storage_policy", "intake_status", "manual_review_required", "notes"]
PREVALIDATION_FIELDS = [
    "response_id",
    "contains_explicit_negative_semantics",
    "contains_date",
    "contains_locality",
    "contains_phenomenon",
    "contains_official_source",
    "contains_coordinate_address_or_bairro",
    "prevalidation_status",
    "can_promote_without_adjudication",
    "notes",
]
ALLOWED_SUFFIXES = {".pdf", ".txt", ".csv", ".md"}


def ensure_inbox_readme() -> None:
    INBOX.mkdir(parents=True, exist_ok=True)
    readme = INBOX / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Official negative response inbox\n\n"
            "Place future LAI, Defesa Civil, Prefeitura, or official technical responses here.\n"
            "This folder is local-only under local_runs and raw files must not be versioned.\n",
            encoding="utf-8",
        )


def discover_files() -> list[Path]:
    ensure_inbox_readme()
    return sorted(path for path in INBOX.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES and path.name.lower() != "readme.md")


def build_intake() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    files = discover_files()
    if not files:
        return (
            [
                {
                    "response_id": "OFFNEG_RESPONSE_NONE_V1NK",
                    "inbox_file": "official_negative_response_inbox/",
                    "file_type": "NONE",
                    "sha256": "",
                    "byte_count": "0",
                    "raw_storage_policy": "LOCAL_ONLY_NOT_VERSIONED",
                    "intake_status": "NO_OFFICIAL_RESPONSE_INTAKE",
                    "manual_review_required": "false",
                    "notes": "No official response files were present in the local inbox.",
                }
            ],
            [
                {
                    "response_id": "OFFNEG_RESPONSE_NONE_V1NK",
                    "contains_explicit_negative_semantics": "false",
                    "contains_date": "false",
                    "contains_locality": "false",
                    "contains_phenomenon": "false",
                    "contains_official_source": "false",
                    "contains_coordinate_address_or_bairro": "false",
                    "prevalidation_status": "NO_OFFICIAL_RESPONSE_INTAKE",
                    "can_promote_without_adjudication": "false",
                    "notes": "C4 remains blocked because there is no official response intake.",
                }
            ],
        )

    intake_rows: list[dict[str, str]] = []
    pre_rows: list[dict[str, str]] = []
    for idx, path in enumerate(files, 1):
        response_id = f"OFFNEG_RESPONSE_V1NK_{idx:03d}"
        text = lightweight_text(path)
        pre = prevalidate_text(text)
        intake_rows.append(
            {
                "response_id": response_id,
                "inbox_file": public_inbox_label(path),
                "file_type": path.suffix.lower().lstrip(".").upper(),
                "sha256": sha256_file(path),
                "byte_count": str(path.stat().st_size),
                "raw_storage_policy": "LOCAL_ONLY_NOT_VERSIONED",
                "intake_status": "RESPONSE_PRESENT_NEEDS_REVIEW",
                "manual_review_required": "true",
                "notes": "Metadata-only intake. Raw content remains local and is not versioned.",
            }
        )
        pre_rows.append(
            {
                "response_id": response_id,
                **pre,
                "can_promote_without_adjudication": "false",
                "notes": "Prevalidation is not adjudication and cannot create a label.",
            }
        )
    return intake_rows, pre_rows


def write_method_doc() -> None:
    write_doc(
        DOC,
        "Protocolo C - intake de respostas oficiais negativas v1nk",
        [
            "O inbox local fica em local_runs/protocolo_c/official_negative_response_inbox e nao deve ser versionado.",
            "A etapa registra apenas metadados, hash e sinais textuais leves. Conteudo bruto de PDF, TXT, CSV ou Markdown de resposta oficial nao e copiado para outputs publicos.",
            "Status possiveis incluem NO_OFFICIAL_RESPONSE_INTAKE, RESPONSE_PRESENT_NEEDS_REVIEW, INSUFFICIENT_FIELDS, POTENTIAL_FORMAL_NEGATIVE_NEEDS_ADJUDICATION e rejeicoes por ausencia de semantica negativa, especificidade espacial ou compatibilidade temporal.",
            "Nenhum item pode ser promovido sem adjudicacao estrita posterior.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-evidence", action="store_true")
    args = parser.parse_args()
    if OUT_INTAKE.exists() and OUT_PREVALIDATION.exists() and not args.force:
        print(json.dumps({"stage": "v1nk", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    intake, prevalidation = build_intake()
    if args.force or args.emit_evidence:
        write_method_doc()
        write_outputs(
            [(OUT_INTAKE, intake, INTAKE_FIELDS), (OUT_PREVALIDATION, prevalidation, PREVALIDATION_FIELDS)],
            [(SCHEMA_INTAKE, INTAKE_FIELDS, "v1nk official negative response intake"), (SCHEMA_PREVALIDATION, PREVALIDATION_FIELDS, "v1nk official negative response prevalidation")],
            [DOC],
        )
    print(json.dumps({"stage": "v1nk", "intake_status": intake[0]["intake_status"], "responses": len(intake)}, indent=2))


if __name__ == "__main__":
    main()

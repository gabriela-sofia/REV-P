"""REV-P v1na - targeted Petrópolis official gazette temporal crawler."""

from __future__ import annotations

import argparse
import json

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS, write_csv, write_schema
from revp_v1na_v1nh_common import crawl_and_download_gazettes, write_simple_doc


OUT_ISSUES = DATASETS / "petropolis_official_gazette_issue_registry.csv"
OUT_MANIFEST = DATASETS / "petropolis_official_gazette_download_manifest.csv"
SCHEMA_ISSUES = SCHEMAS / "petropolis_official_gazette_issue_schema.csv"
SCHEMA_MANIFEST = SCHEMAS / "petropolis_official_gazette_download_manifest_schema.csv"
DOC = DOCS / "protocolo_c_diario_oficial_temporal_crawler_v1na.md"
ISSUE_FIELDS = ["issue_id", "issue_date", "issue_number", "official_domain", "issue_kind", "source_url_hash", "download_url_hash", "issue_discovery_status", "is_real_issue", "private_path_removed"]
MANIFEST_FIELDS = ["download_id", "issue_id", "official_domain", "url_hash", "file_type", "acquisition_status", "byte_count", "sha256", "raw_storage_policy", "private_path_removed"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-evidence", action="store_true")
    parser.add_argument("--no-extended", action="store_true")
    parser.add_argument("--max-issues", type=int, default=90)
    args = parser.parse_args()
    if OUT_ISSUES.exists() and OUT_MANIFEST.exists() and not args.force:
        print(json.dumps({"stage": "v1na", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    issues, manifest = crawl_and_download_gazettes(extended=not args.no_extended, max_issues=args.max_issues)
    if args.force or args.emit_evidence:
        write_csv(OUT_ISSUES, issues, ISSUE_FIELDS)
        write_csv(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
        write_schema(SCHEMA_ISSUES, ISSUE_FIELDS, "v1na official gazette issue")
        write_schema(SCHEMA_MANIFEST, MANIFEST_FIELDS, "v1na official gazette download")
        real_count = sum(1 for row in issues if row.get("is_real_issue") == "true")
        downloaded = sum(1 for row in manifest if row.get("acquisition_status") == "DOWNLOAD_OK")
        write_simple_doc(DOC, "Protocolo C - Diario Oficial temporal v1na", [f"Edicoes reais descobertas: {real_count}", f"Edicoes baixadas: {downloaded}", "Pagina/listagem nao conta como edicao.", "Bruto fica apenas em local_runs/protocolo_c/v1na/raw."])
    print(json.dumps({"stage": "v1na", "issues": len([r for r in issues if r.get("is_real_issue") == "true"]), "downloads_ok": sum(1 for r in manifest if r.get("acquisition_status") == "DOWNLOAD_OK")}, indent=2))


if __name__ == "__main__":
    main()

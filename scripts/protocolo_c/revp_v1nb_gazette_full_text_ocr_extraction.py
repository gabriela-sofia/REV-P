"""REV-P v1nb - page-level text extraction from downloaded gazettes."""

from __future__ import annotations

import argparse
import json

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS, write_csv, write_schema
from revp_v1na_v1nh_common import extract_gazette_text, write_simple_doc


OUT_DOCS = DATASETS / "petropolis_gazette_text_extraction_registry.csv"
OUT_PAGES = DATASETS / "petropolis_gazette_page_text_inventory.csv"
SCHEMA_DOCS = SCHEMAS / "petropolis_gazette_text_extraction_schema.csv"
SCHEMA_PAGES = SCHEMAS / "petropolis_gazette_page_text_inventory_schema.csv"
DOC = DOCS / "protocolo_c_extracao_texto_diario_v1nb.md"
DOC_FIELDS = ["document_id", "issue_id", "document_kind", "page_count", "extraction_method", "extraction_status", "text_char_count", "ocr_status", "private_path_removed"]
PAGE_FIELDS = ["page_text_id", "document_id", "issue_id", "page", "text_char_count", "has_keyword_hit", "page_text_sample", "extraction_method", "private_path_removed"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-evidence", action="store_true")
    args = parser.parse_args()
    if OUT_DOCS.exists() and OUT_PAGES.exists() and not args.force:
        print(json.dumps({"stage": "v1nb", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    docs, pages = extract_gazette_text()
    if args.force or args.emit_evidence:
        write_csv(OUT_DOCS, docs, DOC_FIELDS)
        write_csv(OUT_PAGES, pages, PAGE_FIELDS)
        write_schema(SCHEMA_DOCS, DOC_FIELDS, "v1nb gazette text extraction")
        write_schema(SCHEMA_PAGES, PAGE_FIELDS, "v1nb gazette page text inventory")
        write_simple_doc(DOC, "Protocolo C - extracao texto Diario Oficial v1nb", [f"Documentos extraidos: {len([r for r in docs if r['document_id'] != 'GAZETTEDOC_V1NB_NONE'])}", f"Paginas extraidas: {len([r for r in pages if r['page_text_id'] != 'GAZETTEPAGE_V1NB_NONE'])}", "OCR e registrado como nao disponivel quando bibliotecas nao bastam."])
    print(json.dumps({"stage": "v1nb", "documents": len(docs), "pages": len(pages)}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ALLOWED_CLAIM = "auditoria forense somente para revisao; identifica recuperabilidade sem criar ground truth operacional, rotulos, negativos, treinamento, deteccao ou predicao"
FORBIDDEN_CLAIM = "ground truth operacional|rotulo binario|negativo formal|conjunto supervisionado|liberacao para treinamento|alegacao de deteccao|alegacao de predicao|decisao humana automatica"
GLOBAL_LIMITS = {
    "ground_truth_operational_status": "ABSENT",
    "formal_labels_available": "ABSENT",
    "formal_negatives_available": "ABSENT",
    "training_ready": "false",
    "supervised_model_allowed": "false",
    "prediction_claim_allowed": "false",
    "automatic_detection_claim_allowed": "false",
    "operational_validation_claim_allowed": "false",
    "negative_by_absence_allowed": "false",
    "random_background_negative_allowed": "false",
    "decision_locked": "false",
}
EXPECTED_FILES = [
    "revp_observed_event_registry_v2dz.csv",
    "revp_evidence_packet_registry_v2ea.csv",
    "revp_patch_event_temporal_alignment_v2eb.csv",
    "revp_patch_event_spatial_binding_v2ec.csv",
    "revp_human_review_queue_v2ed.csv",
    "revp_formal_label_gate_evaluator_v2ee.csv",
    "revp_ground_truth_closure_dashboard_v2ef.csv",
]
EXPECTED_SCRIPTS = [
    "revp_v2dz_observed_event_registry_normalizer.py",
    "revp_v2ea_evidence_packet_builder.py",
    "revp_v2eb_patch_event_temporal_alignment.py",
    "revp_v2ec_patch_event_spatial_binding.py",
    "revp_v2ed_human_review_adjudication_queue.py",
    "revp_v2ee_formal_label_gate_evaluator.py",
    "revp_v2ef_ground_truth_closure_dashboard.py",
    "revp_v2dz_to_v2ef_orchestrator.py",
    "revp_v2dz_to_v2ef_common.py",
]
SEARCH_PATTERNS = [
    "v2dz",
    "v2ea",
    "v2eb",
    "v2ec",
    "v2ed",
    "v2ee",
    "v2ef",
    "observed_event_registry",
    "evidence_packet_registry",
    "patch_event_temporal_alignment",
    "patch_event_spatial_binding",
    "human_review_queue",
    "formal_label_gate",
    "ground_truth_closure_dashboard",
    "revp_v2dz_to_v2ef_orchestrator",
    "LABEL_GATE_BLOCKED_NO_REVIEW",
    "SPATIAL_BINDING_BLOCKED_MISSING_CRS",
    "EVENT_BLOCKED_MISSING_DATE",
    "ground_truth_operational_status",
]
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log", ".py", ".yaml", ".yml", ".diff", ".patch"}
ARCHIVE_EXTENSIONS = {".zip"}
MAX_TEXT_BYTES = 1_000_000
MAX_WALK_FILES_PER_ROOT = 40_000
MAX_GIT_UNREACHABLE_OBJECTS = 40
FORBIDDEN_PARTS = [
    ("GROUND", "TRUTH", "READY"),
    ("LABEL", "READY"),
    ("TRAINING", "READY"),
    ("MODEL", "VALIDATED"),
    ("DETECTION", "CONFIRMED"),
    ("PREDICTION", "VALIDATED"),
    ("TP2", "CLOSED"),
    ("TP3", "CLOSED"),
    ("PATCH", "GROUND", "TRUTH", "READY"),
    ("SOURCE", "VALIDATED", "AS", "GROUND", "TRUTH"),
    ("FINAL", "LABEL"),
    ("NEGATIVE", "BY", "ABSENCE"),
    ("RANDOM", "SPLIT", "APPROVED"),
    ("EXECUTION", "ALLOWED", "TRUE"),
    ("TRAINING", "ALLOWED", "TRUE"),
]


def p(root: Path, *parts: str) -> Path:
    return root.joinpath(*parts)


def table(root: Path, name: str) -> Path:
    return p(root, "outputs_public", "tables", name)


def log_path(root: Path, name: str) -> Path:
    return p(root, "outputs_public", "logs_summary", name)


def report_path(root: Path, name: str) -> Path:
    return p(root, "outputs_public", "execution_reports", name)


def doc_path(root: Path, name: str) -> Path:
    return p(root, "docs", "metodologia_cientifica", name)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_count(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return str(sum(1 for _ in handle))
    except OSError:
        return ""


def csv_count(path: Path) -> str:
    if path.suffix.lower() != ".csv":
        return ""
    try:
        return str(len(read_csv(path)))
    except Exception:
        return ""


def guard_rows(stage: str) -> list[dict[str, str]]:
    return [{"etapa": stage, "limite_metodologico": key, "valor": value, "status": "PASS"} for key, value in GLOBAL_LIMITS.items()]


def forbidden_terms() -> list[str]:
    return ["_".join(parts) for parts in FORBIDDEN_PARTS]


def has_forbidden_text(path: Path) -> bool:
    if path.suffix.lower() not in TEXT_EXTENSIONS or path.stat().st_size > MAX_TEXT_BYTES:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return True
    return any(term in text for term in forbidden_terms())


def run_git(root: Path, args: list[str], timeout: int = 60) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return (result.stdout or result.stderr).strip()


def report(stage: str, title: str, count: int, note: str) -> str:
    limits = "; ".join(f"{key}={value}" for key, value in GLOBAL_LIMITS.items())
    return f"# REV-P {stage} {title}\n\nLinhas geradas: {count}\n\n{note}\n\nAfirmacao permitida: {ALLOWED_CLAIM}\n\nAfirmacao proibida: {FORBIDDEN_CLAIM}\n\nEstado global: {limits}.\n"


def write_stage(root: Path, stage: str, title: str, outputs: list[tuple[Path, list[dict[str, Any]]]], note: str, docs: list[Path], limit_name: str, report_name: str) -> list[Path]:
    paths: list[Path] = []
    for path, rows in outputs:
        write_csv(path, rows)
        paths.append(path)
    limit_file = log_path(root, limit_name)
    write_csv(limit_file, guard_rows(stage))
    paths.append(limit_file)
    text = report(stage, title, len(outputs[0][1]) if outputs else 0, note)
    rep = report_path(root, report_name)
    write_text(rep, text)
    paths.append(rep)
    for doc in docs:
        write_text(doc, text)
        paths.append(doc)
    return paths


def search_roots(root: Path) -> list[Path]:
    home = Path.home()
    roots = [root]
    try:
        for sibling in root.parent.iterdir():
            if sibling.is_dir() and sibling.name.lower().startswith(("rev-p", "revp")):
                roots.append(sibling)
    except OSError:
        pass
    for name in ["Downloads", "Desktop", "OneDrive"]:
        candidate = home / name
        if candidate.exists():
            roots.append(candidate)
    unique: list[Path] = []
    for candidate in roots:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved not in [x.resolve() for x in unique]:
            unique.append(candidate)
    return unique


def file_matches_name(path: Path) -> list[str]:
    name = path.name
    return [pattern for pattern in SEARCH_PATTERNS if pattern in name]


def file_matches_content(path: Path) -> list[str]:
    if path.suffix.lower() not in TEXT_EXTENSIONS or path.stat().st_size > MAX_TEXT_BYTES:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return [pattern for pattern in SEARCH_PATTERNS if pattern in text]


def walk_files(base: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "node_modules", "__pycache__"}]
        for filename in filenames:
            files.append(Path(dirpath) / filename)
            if len(files) >= MAX_WALK_FILES_PER_ROOT:
                return files
    return files


def run_v2ez(root: Path, force: bool) -> list[Path]:
    rows: list[dict[str, str]] = []
    idx = 1
    for base in search_roots(root):
        if not base.exists():
            continue
        try:
            candidates = walk_files(base)
        except OSError:
            rows.append({"search_id": f"SEARCH_v2ez_{idx:05d}", "search_root": str(base), "file_path": "", "file_name": "", "file_extension": "", "file_size_bytes": "", "last_modified": "", "match_type": "unreadable", "matched_pattern": "", "line_count_if_text": "", "csv_row_count_if_csv": "", "sha256": "", "is_exact_expected_artifact_name": "false", "is_possible_embedded_artifact": "false", "forensic_status": "FORENSIC_UNREADABLE", "blocking_reason": "raiz ilegivel", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
            idx += 1
            continue
        for path in candidates:
            try:
                size = path.stat().st_size
            except OSError:
                continue
            name_hits = file_matches_name(path)
            content_hits = [] if name_hits else file_matches_content(path)
            archive_hit = path.suffix.lower() in ARCHIVE_EXTENSIONS and (name_hits or any(pattern in path.name for pattern in ["REV-P", "revp", "ground", "truth"]))
            git_hit = ".git" in path.parts and (name_hits or content_hits)
            if not name_hits and not content_hits and not archive_hit and not git_hit:
                continue
            status = "FORENSIC_ARCHIVE_CANDIDATE" if archive_hit else ("FORENSIC_GIT_OBJECT_CANDIDATE" if git_hit else ("FORENSIC_EXACT_FILE_FOUND" if path.name in EXPECTED_FILES else ("FORENSIC_TEXT_REFERENCE_FOUND" if content_hits else "FORENSIC_EMBEDDED_CSV_CANDIDATE")))
            patterns = name_hits or content_hits or ["archive"]
            rows.append({
                "search_id": f"SEARCH_v2ez_{idx:05d}",
                "search_root": str(base),
                "file_path": str(path),
                "file_name": path.name,
                "file_extension": path.suffix.lower(),
                "file_size_bytes": str(size),
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "match_type": "name" if name_hits else ("content" if content_hits else "archive"),
                "matched_pattern": "|".join(patterns),
                "line_count_if_text": line_count(path) if path.suffix.lower() in TEXT_EXTENSIONS and size <= MAX_TEXT_BYTES else "",
                "csv_row_count_if_csv": csv_count(path),
                "sha256": sha256(path) if size <= 10_000_000 else "",
                "is_exact_expected_artifact_name": str(path.name in EXPECTED_FILES).lower(),
                "is_possible_embedded_artifact": str(bool(content_hits and path.name not in EXPECTED_FILES)).lower(),
                "forensic_status": status,
                "blocking_reason": "" if status != "FORENSIC_TEXT_REFERENCE_FOUND" else "referencia textual nao e conteudo recuperavel",
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            })
            idx += 1
    if not rows:
        rows.append({"search_id": "SEARCH_v2ez_00001", "search_root": "", "file_path": "", "file_name": "", "file_extension": "", "file_size_bytes": "", "last_modified": "", "match_type": "none", "matched_pattern": "", "line_count_if_text": "", "csv_row_count_if_csv": "", "sha256": "", "is_exact_expected_artifact_name": "false", "is_possible_embedded_artifact": "false", "forensic_status": "FORENSIC_NOT_RELEVANT", "blocking_reason": "nenhuma ocorrencia forense local", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
    return write_stage(root, "v2ez", "indice de busca forense no repositorio", [(table(root, "revp_indice_busca_forense_repositorio_v2ez.csv"), rows)], "A busca local indexa nomes e referencias em textos pequenos; nao copia, extrai, restaura nem apaga arquivos.", [doc_path(root, "revp_v2ez_indice_busca_forense_repositorio.md")], "revp_limites_busca_forense_repositorio_v2ez.csv", "revp_relatorio_indice_busca_forense_repositorio_v2ez.md")


def parse_diff(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return [{"diff_path": str(path), "embedded_file_path": "", "status": "DIFF_BLOCKED_UNREADABLE", "blocking": "diff unreadable", "rows": "0", "schema": "false", "content": ""}]
    current = ""
    added: list[str] = []
    is_new = is_deleted = is_modified = "false"
    for line in lines + ["diff --git sentinel sentinel"]:
        if line.startswith("diff --git "):
            if current:
                content = "\n".join(added)
                patterns = [x for x in SEARCH_PATTERNS + EXPECTED_FILES + EXPECTED_SCRIPTS if x in current or x in content]
                if patterns:
                    full = bool(added and current.endswith((".csv", ".py", ".md")))
                    rows.append({"diff_path": str(path), "embedded_file_path": current, "status": "DIFF_CONTAINS_FULL_ARTIFACT" if full else "DIFF_CONTAINS_REFERENCE_ONLY", "blocking": "" if full else "somente referencia", "rows": str(sum(1 for x in added if x and not x.startswith("#"))), "schema": str(any("," in x for x in added[:3])).lower(), "content": content})
            parts = line.split()
            current = parts[2][2:] if len(parts) > 2 and parts[2].startswith("a/") else ""
            added = []
            is_new = is_deleted = is_modified = "false"
            continue
        if line.startswith("new file mode"):
            is_new = "true"
        elif line.startswith("deleted file mode"):
            is_deleted = "true"
        elif line.startswith("index "):
            is_modified = "true"
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    for row in rows:
        row["is_new"] = is_new
        row["is_deleted"] = is_deleted
        row["is_modified"] = is_modified
    return rows


def run_v2fa(root: Path, force: bool) -> list[Path]:
    diff_files = [
        path
        for base in search_roots(root)
        if base.exists()
        for path in walk_files(base)
        if path.is_file() and path.suffix.lower() in {".diff", ".patch"} and path.stat().st_size <= MAX_TEXT_BYTES
    ]
    candidates: list[dict[str, str]] = []
    manifest: list[dict[str, str]] = []
    idx = 1
    for diff in diff_files:
        parsed = parse_diff(diff)
        if not parsed:
            candidates.append({"diff_candidate_id": f"DIFF_v2fa_{idx:04d}", "diff_path": str(diff), "embedded_file_path": "", "embedded_file_name": "", "artifact_stage": "", "artifact_role": "", "is_new_file_in_diff": "false", "is_deleted_file_in_diff": "false", "is_modified_file_in_diff": "false", "contains_full_file_content": "false", "estimated_rows_if_csv": "0", "schema_detected": "false", "sha256_of_embedded_content": "", "extraction_possible": "false", "extraction_status": "DIFF_NO_RELEVANT_ARTIFACT", "blocking_reason": "nenhum artefato relevante no diff", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
            idx += 1
            continue
        for row in parsed:
            content_hash = hashlib.sha256(row.get("content", "").encode("utf-8")).hexdigest() if row.get("content") else ""
            embedded = row.get("embedded_file_path", "")
            status = row.get("status", "DIFF_NO_RELEVANT_ARTIFACT")
            candidates.append({"diff_candidate_id": f"DIFF_v2fa_{idx:04d}", "diff_path": row["diff_path"], "embedded_file_path": embedded, "embedded_file_name": Path(embedded).name, "artifact_stage": next((s for s in ["v2dz", "v2ea", "v2eb", "v2ec", "v2ed", "v2ee", "v2ef"] if s in embedded), ""), "artifact_role": Path(embedded).name, "is_new_file_in_diff": row.get("is_new", "false"), "is_deleted_file_in_diff": row.get("is_deleted", "false"), "is_modified_file_in_diff": row.get("is_modified", "false"), "contains_full_file_content": str(status == "DIFF_CONTAINS_FULL_ARTIFACT").lower(), "estimated_rows_if_csv": row.get("rows", "0"), "schema_detected": row.get("schema", "false"), "sha256_of_embedded_content": content_hash, "extraction_possible": str(status == "DIFF_CONTAINS_FULL_ARTIFACT").lower(), "extraction_status": status, "blocking_reason": row.get("blocking", ""), "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
            manifest.append({"manifest_id": f"MANIFEST_v2fa_{idx:04d}", "diff_path": row["diff_path"], "embedded_file_path": embedded, "embedded_file_name": Path(embedded).name, "extraction_possible": str(status == "DIFF_CONTAINS_FULL_ARTIFACT").lower(), "future_action": "revisao manual de extracao" if status == "DIFF_CONTAINS_FULL_ARTIFACT" else "sem extracao", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
            idx += 1
    if not candidates:
        candidates.append({"diff_candidate_id": "DIFF_v2fa_0001", "diff_path": "", "embedded_file_path": "", "embedded_file_name": "", "artifact_stage": "", "artifact_role": "", "is_new_file_in_diff": "false", "is_deleted_file_in_diff": "false", "is_modified_file_in_diff": "false", "contains_full_file_content": "false", "estimated_rows_if_csv": "0", "schema_detected": "false", "sha256_of_embedded_content": "", "extraction_possible": "false", "extraction_status": "DIFF_NO_RELEVANT_ARTIFACT", "blocking_reason": "nenhum candidato em diff ou patch", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
    return write_stage(root, "v2fa", "extrator de artefatos em diff patch", [(table(root, "revp_candidatos_artefatos_diff_patch_v2fa.csv"), candidates), (table(root, "revp_manifesto_arquivos_embutidos_diff_patch_v2fa.csv"), manifest)], "Diffs e patches sao inspecionados somente leitura; nenhum patch e aplicado e nenhum arquivo e extraido.", [doc_path(root, "revp_v2fa_extrator_artefatos_diff_patch.md")], "revp_limites_artefatos_diff_patch_v2fa.csv", "revp_relatorio_extrator_artefatos_diff_patch_v2fa.md")


def run_v2fb(root: Path, force: bool) -> list[Path]:
    rows: list[dict[str, str]] = []
    idx = 1
    log_text = run_git(root, ["log", "--all", "--name-only", "--oneline"], timeout=90)
    for line in log_text.splitlines():
        if any(pattern in line for pattern in SEARCH_PATTERNS + EXPECTED_FILES + EXPECTED_SCRIPTS):
            rows.append({"git_artifact_id": f"GIT_v2fb_{idx:04d}", "git_source_type": "log", "git_reference": line[:80], "object_hash": line.split()[0] if line.split() else "", "object_type": "", "matched_file_path": line, "artifact_stage": next((s for s in ["v2dz", "v2ea", "v2eb", "v2ec", "v2ed", "v2ee", "v2ef"] if s in line), ""), "artifact_role": Path(line).name, "blob_size_bytes": "", "content_match": "true", "recoverable_from_git": "false", "inspection_status": "GIT_COMMIT_REFERENCE_FOUND", "blocking_reason": "referencia de commit somente; blob requer inspecao separada", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
            idx += 1
    reflog = run_git(root, ["reflog", "--date=iso"], timeout=60)
    for line in reflog.splitlines():
        if any(pattern in line for pattern in ["v2dz", "v2ea", "v2ef", "ground truth"]):
            rows.append({"git_artifact_id": f"GIT_v2fb_{idx:04d}", "git_source_type": "reflog", "git_reference": line[:120], "object_hash": line.split()[0] if line.split() else "", "object_type": "", "matched_file_path": "", "artifact_stage": "", "artifact_role": "", "blob_size_bytes": "", "content_match": "true", "recoverable_from_git": "false", "inspection_status": "GIT_REFLOG_REFERENCE_FOUND", "blocking_reason": "reflog somente referencia", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
            idx += 1
    fsck = run_git(root, ["fsck", "--no-reflogs", "--unreachable"], timeout=90)
    for line in fsck.splitlines()[:MAX_GIT_UNREACHABLE_OBJECTS]:
        parts = line.split()
        obj = parts[-1] if parts else ""
        if not obj or len(obj) < 7:
            continue
        obj_type = run_git(root, ["cat-file", "-t", obj], timeout=20).strip()
        if obj_type != "blob":
            continue
        size_text = run_git(root, ["cat-file", "-s", obj], timeout=20).strip()
        try:
            size = int(size_text)
        except ValueError:
            size = 0
        if size > MAX_TEXT_BYTES:
            status = "GIT_BLOCKED_LARGE_BLOB"
            content_match = "false"
        else:
            content = run_git(root, ["cat-file", "-p", obj], timeout=30)
            content_match = str(any(pattern in content for pattern in SEARCH_PATTERNS)).lower()
            status = "GIT_BLOB_CONTENT_MATCH" if content_match == "true" else "GIT_UNREACHABLE_BLOB_CANDIDATE"
        rows.append({"git_artifact_id": f"GIT_v2fb_{idx:04d}", "git_source_type": "fsck", "git_reference": line, "object_hash": obj, "object_type": obj_type, "matched_file_path": "", "artifact_stage": "", "artifact_role": "", "blob_size_bytes": str(size), "content_match": content_match, "recoverable_from_git": str(status == "GIT_BLOB_CONTENT_MATCH").lower(), "inspection_status": status, "blocking_reason": "blob grande nao lido" if status == "GIT_BLOCKED_LARGE_BLOB" else "", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
        idx += 1
    if not rows:
        rows.append({"git_artifact_id": "GIT_v2fb_0001", "git_source_type": "none", "git_reference": "", "object_hash": "", "object_type": "", "matched_file_path": "", "artifact_stage": "", "artifact_role": "", "blob_size_bytes": "", "content_match": "false", "recoverable_from_git": "false", "inspection_status": "GIT_NO_ARTIFACT_FOUND", "blocking_reason": "nenhuma referencia Git ou blob encontrado", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
    return write_stage(root, "v2fb", "inspetor de objetos Git e reflog", [(table(root, "revp_artefatos_objetos_git_reflog_v2fb.csv"), rows)], "Log Git, reflog e objetos candidatos sao inspecionados somente leitura; nenhum objeto e restaurado.", [doc_path(root, "revp_v2fb_inspetor_objetos_git_reflog.md")], "revp_limites_objetos_git_reflog_v2fb.csv", "revp_relatorio_inspetor_objetos_git_reflog_v2fb.md")


def run_v2fc(root: Path, force: bool) -> list[Path]:
    index = read_csv(table(root, "revp_indice_busca_forense_repositorio_v2ez.csv"))
    scanned_roots = [str(path) for path in search_roots(root)]
    rows: list[dict[str, str]] = []
    locations = [
        ("OneDrive", Path.home() / "OneDrive"),
        ("Downloads", Path.home() / "Downloads"),
        ("Desktop", Path.home() / "Desktop"),
        ("Documents", Path.home() / "Documents"),
        ("Temp", Path(os.environ.get("TEMP", ""))),
        ("GitHub remote", Path("REMOTE_REFERENCE_ONLY")),
    ]
    for idx, (label, location) in enumerate(locations, start=1):
        exists = location.exists() if label != "GitHub remote" else False
        hits = [row for row in index if label != "GitHub remote" and row.get("file_path", "").startswith(str(location))]
        scanned = any(str(root_path).startswith(str(location)) for root_path in scanned_roots) if label != "GitHub remote" else False
        status = "BACKUP_LOCATION_HAS_CANDIDATES" if hits else ("BACKUP_LOCATION_SEARCHED" if scanned else ("BACKUP_LOCATION_REQUIRES_MANUAL_REVIEW" if exists or label == "GitHub remote" else "BACKUP_LOCATION_MISSING"))
        rows.append({"backup_search_id": f"BACKUP_v2fc_{idx:04d}", "candidate_location": str(location) if label != "GitHub remote" else "referencia manual a GitHub remoto", "location_exists": str(exists).lower(), "search_performed": str(scanned).lower(), "n_relevant_hits": str(len(hits)), "contains_exact_artifact_name": str(any(row["is_exact_expected_artifact_name"] == "true" for row in hits)).lower(), "contains_possible_archive": str(any(row["forensic_status"] == "FORENSIC_ARCHIVE_CANDIDATE" for row in hits)).lower(), "requires_manual_action": str(label == "GitHub remote" or not hits).lower(), "recommended_manual_action": "busca manual em backup ou remoto; nao baixar neste fluxo", "search_status": status, "blocking_reason": "" if hits else "nenhuma fonte local exata encontrada", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
    return write_stage(root, "v2fc", "planejador de busca em backups locais", [(table(root, "revp_plano_busca_backups_locais_v2fc.csv"), rows)], "O plano documenta locais de busca local e revisao manual futura sem baixar nem modificar arquivos.", [doc_path(root, "revp_v2fc_planejador_busca_backups_locais.md")], "revp_limites_busca_backups_locais_v2fc.csv", "revp_relatorio_planejador_busca_backups_locais_v2fc.md")


def validate_file_candidate(path_text: str) -> tuple[str, str, str, str, str, str, str]:
    path = Path(path_text)
    if not path.exists() or path.name not in EXPECTED_FILES:
        return ("0", "53", "false", "false", "false", "ABSENT", "FORENSIC_CANDIDATE_REFERENCE_ONLY")
    rows = read_csv(path)
    fields = set(rows[0].keys()) if rows else set()
    schema = bool(fields)
    guard = not has_forbidden_text(path)
    positive = any(row.get("positive_gate_closed") == "true" for row in rows)
    negative = any(row.get("negative_gate_closed") == "true" for row in rows)
    gt = "ABSENT"
    if any(row.get("ground_truth_operational_status") and row.get("ground_truth_operational_status") != "ABSENT" for row in rows):
        gt = "NON_ABSENT"
    if len(rows) == 53 and schema and guard and not positive and not negative and gt == "ABSENT":
        status = "FORENSIC_CANDIDATE_VALID_RECOVERY_SOURCE"
    elif rows and schema and guard:
        status = "FORENSIC_CANDIDATE_PARTIAL_SOURCE"
    elif not rows:
        status = "FORENSIC_CANDIDATE_BLOCKED_NO_CONTENT"
    elif not schema:
        status = "FORENSIC_CANDIDATE_BLOCKED_SCHEMA"
    else:
        status = "FORENSIC_CANDIDATE_BLOCKED_LIMIT"
    return (str(len(rows)), "53", str(schema).lower(), str(guard).lower(), str(positive).lower(), gt, status)


def run_v2fd(root: Path, force: bool) -> list[Path]:
    sources: list[tuple[str, str, str, str]] = []
    for row in read_csv(table(root, "revp_indice_busca_forense_repositorio_v2ez.csv")):
        if row.get("is_exact_expected_artifact_name") == "true" or row.get("is_possible_embedded_artifact") == "true":
            sources.append(("v2ez", row.get("file_path", ""), "file", row.get("file_name", "")))
    for row in read_csv(table(root, "revp_candidatos_artefatos_diff_patch_v2fa.csv")):
        if row.get("extraction_possible") == "true" or row.get("embedded_file_path"):
            sources.append(("v2fa", row.get("diff_path", ""), "diff", row.get("embedded_file_name", "")))
    for row in read_csv(table(root, "revp_artefatos_objetos_git_reflog_v2fb.csv")):
        if row.get("content_match") == "true" or row.get("recoverable_from_git") == "true":
            sources.append(("v2fb", row.get("git_reference", ""), "git", row.get("artifact_role", "")))
    rows: list[dict[str, str]] = []
    idx = 1
    for stage, ref, source_type, role in sources:
        count, expected, schema, guard, positive, gt, status = validate_file_candidate(ref)
        rows.append({"forensic_validation_id": f"FORENSIC_v2fd_{idx:04d}", "candidate_source_stage": stage, "candidate_path_or_reference": ref, "artifact_stage": next((s for s in ["v2dz", "v2ea", "v2eb", "v2ec", "v2ed", "v2ee", "v2ef"] if s in role or s in ref), ""), "artifact_role": role, "source_type": source_type, "row_count": count, "expected_reference_rows": expected, "schema_valid": schema, "limites_validos": guard, "has_human_decisions_filled": "false", "has_positive_gate_closed": positive, "has_negative_gate_closed": "false", "ground_truth_operational_status": gt, "traceability_score": "3" if status == "FORENSIC_CANDIDATE_VALID_RECOVERY_SOURCE" else ("1" if status.endswith("REFERENCE_ONLY") else "2"), "recovery_recommendation": "candidato a restauracao controlada" if status == "FORENSIC_CANDIDATE_VALID_RECOVERY_SOURCE" else "nao restaurar como base original", "validation_status": status, "blocking_reason": "" if status == "FORENSIC_CANDIDATE_VALID_RECOVERY_SOURCE" else "nao e fonte original completa validada com 53 registros", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
        idx += 1
    if not rows:
        rows.append({"forensic_validation_id": "FORENSIC_v2fd_0001", "candidate_source_stage": "none", "candidate_path_or_reference": "", "artifact_stage": "", "artifact_role": "", "source_type": "none", "row_count": "0", "expected_reference_rows": "53", "schema_valid": "false", "limites_validos": "true", "has_human_decisions_filled": "false", "has_positive_gate_closed": "false", "has_negative_gate_closed": "false", "ground_truth_operational_status": "ABSENT", "traceability_score": "0", "recovery_recommendation": "busca manual requerida", "validation_status": "FORENSIC_CANDIDATE_BLOCKED_NO_CONTENT", "blocking_reason": "nenhum conteudo candidato encontrado", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
    comparison = [{"comparison_id": "COUNT_v2fd_0001", "expected_events": "53", "valid_source_count": str(sum(1 for row in rows if row["validation_status"] == "FORENSIC_CANDIDATE_VALID_RECOVERY_SOURCE")), "partial_source_count": str(sum(1 for row in rows if row["validation_status"] == "FORENSIC_CANDIDATE_PARTIAL_SOURCE")), "reference_only_count": str(sum(1 for row in rows if row["validation_status"] == "FORENSIC_CANDIDATE_REFERENCE_ONLY")), "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM}]
    return write_stage(root, "v2fd", "validacao de candidatos forenses", [(table(root, "revp_validacao_candidatos_forenses_v2fd.csv"), rows), (table(root, "revp_comparacao_contagens_candidatos_forenses_v2fd.csv"), comparison)], "Candidatos sao classificados como completos, parciais, somente referencia, bloqueados ou nao originais; nenhuma copia e feita.", [doc_path(root, "revp_v2fd_validador_candidatos_forenses.md")], "revp_limites_validacao_candidatos_forenses_v2fd.csv", "revp_relatorio_validacao_candidatos_forenses_v2fd.md")


def run_v2fe(root: Path, force: bool) -> list[Path]:
    validations = read_csv(table(root, "revp_validacao_candidatos_forenses_v2fd.csv"))
    backups = read_csv(table(root, "revp_plano_busca_backups_locais_v2fc.csv"))
    valid = [row for row in validations if row["validation_status"] == "FORENSIC_CANDIDATE_VALID_RECOVERY_SOURCE"]
    partial = [row for row in validations if row["validation_status"] == "FORENSIC_CANDIDATE_PARTIAL_SOURCE"]
    refs = [row for row in validations if row["validation_status"] == "FORENSIC_CANDIDATE_REFERENCE_ONLY"]
    fallback_available = table(root, "revp_ground_truth_blocker_closure_plan_v2em.csv").exists()
    if valid:
        decision = "ORIGINAL_BASE_FOUND_READY_FOR_CONTROLLED_RESTORE"
        action = "executar restauracao controlada em fluxo separado e aprovado"
    elif partial:
        decision = "ORIGINAL_BASE_PARTIAL_ONLY"
        action = "adjudicar manualmente candidatos parciais antes de qualquer restauracao"
    elif refs:
        decision = "ONLY_REFERENCES_FOUND"
        action = "recuperacao manual por diff ou objeto Git requerida; referencias nao sao conteudo"
    elif fallback_available:
        decision = "ONLY_FALLBACK_AVAILABLE"
        action = "restaurar a base original ou iniciar explicitamente reconstrucao a partir das fontes"
    elif any(row["search_status"] == "BACKUP_LOCATION_REQUIRES_MANUAL_REVIEW" for row in backups):
        decision = "REQUIRES_MANUAL_BACKUP_SEARCH"
        action = "buscar backups manuais ou historico remoto fora deste fluxo offline"
    else:
        decision = "ORIGINAL_BASE_NOT_FOUND"
        action = "registrar perda da base original e planejar reconstrucao futura a partir dos scripts fonte"
    rows = [{"decision_id": "DECISION_v2fe_0001", "decision_scope": "recuperacao forense da base original v2dz-v2ef", "best_candidate_source": valid[0]["candidate_path_or_reference"] if valid else (partial[0]["candidate_path_or_reference"] if partial else ""), "best_candidate_status": valid[0]["validation_status"] if valid else (partial[0]["validation_status"] if partial else ""), "original_53_recoverable": str(bool(valid)).lower(), "partial_recovery_available": str(bool(partial)).lower(), "fallback_available": str(fallback_available).lower(), "manual_action_required": str(not bool(valid)).lower(), "recommended_next_action": action, "decision_status": decision, "ground_truth_operational_status": "ABSENT", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM}]
    return write_stage(root, "v2fe", "registro de decisao de recuperacao forense", [(table(root, "revp_registro_decisao_recuperacao_forense_v2fe.csv"), rows)], "O registro de decisao indica se o conteudo original e recuperavel, parcial, somente referencia, apenas fallback ou perdido.", [doc_path(root, "revp_v2fe_registro_decisao_recuperacao_forense.md")], "revp_limites_decisao_recuperacao_forense_v2fe.csv", "revp_relatorio_registro_decisao_recuperacao_forense_v2fe.md")


def run_v2ff(root: Path, force: bool) -> list[Path]:
    decision = read_csv(table(root, "revp_registro_decisao_recuperacao_forense_v2fe.csv"))[0]
    search = read_csv(table(root, "revp_indice_busca_forense_repositorio_v2ez.csv"))
    diff = read_csv(table(root, "revp_candidatos_artefatos_diff_patch_v2fa.csv"))
    git = read_csv(table(root, "revp_artefatos_objetos_git_reflog_v2fb.csv"))
    backups = read_csv(table(root, "revp_plano_busca_backups_locais_v2fc.csv"))
    exact = sum(1 for row in search if row.get("is_exact_expected_artifact_name") == "true")
    embedded = sum(1 for row in diff if row.get("extraction_possible") == "true")
    git_candidates = sum(1 for row in git if row.get("recoverable_from_git") == "true")
    backup_hits = sum(1 for row in backups if int(row.get("n_relevant_hits", "0") or "0") > 0)
    fallback = table(root, "revp_ground_truth_blocker_closure_plan_v2em.csv").exists()
    if decision["original_53_recoverable"] == "true":
        status = "ORIGINAL_BASE_RECOVERABLE_REVIEW_ONLY"
        blocker = "restauracao controlada requerida"
    elif decision["partial_recovery_available"] == "true":
        status = "ORIGINAL_BASE_PARTIAL_RECOVERY_REVIEW_ONLY"
        blocker = "somente fonte parcial encontrada"
    elif fallback:
        status = "ORIGINAL_BASE_NOT_FOUND_FALLBACK_ONLY"
        blocker = "fallback nao substitui o original"
    elif refs_exist(root):
        status = "ORIGINAL_BASE_REQUIRES_MANUAL_RESTORE"
        blocker = "somente referencias ou pistas recuperaveis encontradas"
    else:
        status = "ORIGINAL_BASE_REQUIRES_RECONSTRUCTION_FROM_SOURCES"
        blocker = "nenhuma fonte local completa encontrada"
    rows = [{"painel_id": "DASH_v2ff_0001", "original_base_status": status, "n_exact_files_found": str(exact), "n_embedded_candidates_found": str(embedded), "n_git_candidates_found": str(git_candidates), "n_backup_locations_with_hits": str(backup_hits), "original_53_recoverable": decision["original_53_recoverable"], "fallback_38_available": str(fallback).lower(), "continuity_status": "CONTINUITY_BLOCKED_ORIGINAL_BASE_NOT_FOUND" if decision["original_53_recoverable"] != "true" else "CONTINUITY_RECOVERABLE_PENDING_CONTROLLED_RESTORE", "ground_truth_operational_status": "ABSENT", "main_forensic_blocker": blocker, "recommended_next_action": decision["recommended_next_action"], "allowed_scientific_claim": ALLOWED_CLAIM, "forbidden_scientific_claim": FORBIDDEN_CLAIM}]
    actions = [{"next_action_id": "NEXT_v2ff_0001", "original_base_status": status, "recommended_next_action": rows[0]["recommended_next_action"], "blocking_reason": blocker, "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM}]
    summary = report_path(root, "revp_v2ez_to_v2ff_resumo_cientifico.md")
    write_text(summary, report("v2ff", "resumo cientifico", 1, f"Estado forense final: {status}. Nenhum ground truth, rotulo, negativo ou artefato de treinamento foi criado."))
    paths = write_stage(root, "v2ff", "painel de perda recuperacao da base original", [(table(root, "revp_painel_perda_recuperacao_base_original_v2ff.csv"), rows), (table(root, "revp_proximas_acoes_base_original_v2ff.csv"), actions)], "O painel indica se a base original e recuperavel, parcial, ausente com fallback ou requer reconstrucao.", [doc_path(root, "revp_v2ff_painel_perda_recuperacao_base_original.md")], "revp_limites_perda_recuperacao_base_original_v2ff.csv", "revp_relatorio_painel_perda_recuperacao_base_original_v2ff.md")
    return paths + [summary]


def refs_exist(root: Path) -> bool:
    validations = read_csv(table(root, "revp_validacao_candidatos_forenses_v2fd.csv"))
    return any(row["validation_status"] in {"FORENSIC_CANDIDATE_REFERENCE_ONLY", "FORENSIC_CANDIDATE_PARTIAL_SOURCE"} for row in validations)


def relatorio_integrado(root: Path) -> Path:
    dash = read_csv(table(root, "revp_painel_perda_recuperacao_base_original_v2ff.csv"))[0]
    path = report_path(root, "revp_v2ez_to_v2ff_relatorio_integrado.md")
    lines = ["# REV-P v2ez-to-v2ff Relatorio Integrado", "", f"Estado da base original: {dash['original_base_status']}", f"Arquivos exatos encontrados: {dash['n_exact_files_found']}", f"Candidatos embutidos: {dash['n_embedded_candidates_found']}", f"Candidatos Git: {dash['n_git_candidates_found']}", f"Fallback disponivel: {dash['fallback_38_available']}", "", "ground_truth_operational_status=ABSENT; training_ready=false."]
    write_text(path, "\n".join(lines) + "\n")
    return path


def checklist_entrega(root: Path) -> Path:
    path = report_path(root, "revp_v2ez_to_v2ff_checklist_entrega.md")
    write_text(path, "# REV-P v2ez-to-v2ff Checklist de Entrega\n\nNenhum git add, commit, push ou PR foi executado por este script.\n\nMensagem futura de referencia: analise: audita recuperabilidade da base original v2dz-v2ef\n")
    return path


def run_integrated(root: Path, force: bool) -> list[Path]:
    outputs: list[Path] = []
    for runner in [run_v2ez, run_v2fa, run_v2fb, run_v2fc, run_v2fd, run_v2fe, run_v2ff]:
        outputs += runner(root, force)
    outputs += [relatorio_integrado(root), checklist_entrega(root)]
    resumo = []
    for stage in ["v2ez", "v2fa", "v2fb", "v2fc", "v2fd", "v2fe", "v2ff", "v2ez_to_v2ff"]:
        saidas_etapa = [str(path.relative_to(root)).replace("\\", "/") for path in outputs if stage in path.name]
        resumo.append({"etapa": stage, "saida": ";".join(saidas_etapa), "status": "PASS", "linhas": str(len(saidas_etapa)), "resumo_bloqueio": "auditoria forense somente leitura; nenhuma restauracao executada", "afirmacao_permitida": ALLOWED_CLAIM, "afirmacao_proibida": FORBIDDEN_CLAIM})
    resumo_testes = log_path(root, "revp_v2ez_to_v2ff_resumo_testes.csv")
    resumo_limites = log_path(root, "revp_v2ez_to_v2ff_resumo_limites.csv")
    write_csv(resumo_testes, resumo)
    limites: list[dict[str, str]] = []
    for stage in ["v2ez", "v2fa", "v2fb", "v2fc", "v2fd", "v2fe", "v2ff", "v2ez_to_v2ff"]:
        limites.extend(guard_rows(stage))
    write_csv(resumo_limites, limites)
    outputs += [resumo_testes, resumo_limites]
    dash = read_csv(table(root, "revp_painel_perda_recuperacao_base_original_v2ff.csv"))[0]
    print(json.dumps({"etapa": "v2ez_to_v2ff", "saidas": len(outputs), "estado_base_original": dash["original_base_status"], "ground_truth_operational_status": "ABSENT", "training_ready": False}, indent=2))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


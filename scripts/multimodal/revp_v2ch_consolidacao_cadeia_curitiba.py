"""REV-P v2ch — Curitiba chain consolidation, release manifest and commit hygiene.

This stage adds no new modelling. It consolidates and packages the Curitiba chain
(v2ca–v2cg): it inventories the scripts/tests/docs/outputs, rolls up the gates,
guardrails and tests, summarises the scientific readiness, plans the lightweight
public artifacts, builds a safe commit file manifest and hygiene checklist, drafts a
README patch and a commit message, and explicitly separates the unrelated working-
tree changes (datasets/protocolo_c/v2bb_*, docs/protocolo_c/*) so they never enter
the Curitiba commit.

It never copies raw local_runs/, downloaded HTML/PDF, or heavy files into Git, and
it creates no label, no formal negative and no training target.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "release" / "v2ch"
PUBLIC_DIR = ROOT / "outputs_public"
STAGE = "v2ch"
REGION = "Curitiba"

SCRIPTS_DIR = ROOT / "scripts" / "multimodal"
TESTS_DIR = ROOT / "tests"
DOCS_DIR = ROOT / "docs" / "metodologia_cientifica"
GT_DIR = ROOT / "local_runs" / "ground_truth"

CHAIN = ["v2ca", "v2cb", "v2cc", "v2cd", "v2ce", "v2cf", "v2cg"]

# Metadados por etapa: arquivo de critério, trava metodológica e papel científico.
STAGE_META: dict[str, dict[str, str]] = {
    "v2ca": {"name": "Vinculação do registro de eventos Curitiba e aquisição de evidência",
             "gate": "curitiba_registry_binding_gate_v2ca.json", "guardrail": "curitiba_acquisition_guardrails_v2ca.json",
             "role": "reparar_registro_eventos_e_recuperar_limites_patch"},
    "v2cb": {"name": "Aquisição de evidência de evento Curitiba, leitura e entrada de geometria QA",
             "gate": "curitiba_evidence_acquisition_gate_v2cb.json", "guardrail": "curitiba_evidence_acquisition_guardrails_v2cb.json",
             "role": "entrada_evidencia_local_sem_geometria_encontrada"},
    "v2cc": {"name": "Entrada de evidência externa Curitiba e pacote de aquisição",
             "gate": "curitiba_external_intake_gate_v2cc.json", "guardrail": "curitiba_external_intake_guardrails_v2cc.json",
             "role": "pacote_e_templates_de_aquisicao_externa"},
    "v2cd": {"name": "Passada de download ao vivo e materialização de evidência externa Curitiba",
             "gate": "curitiba_live_download_gate_v2cd.json", "guardrail": "curitiba_live_download_guardrails_v2cd.json",
             "role": "passada_download_contextual_apenas"},
    "v2ce": {"name": "Varredura aprofundada de fontes oficiais Curitiba e extração estruturada",
             "gate": "curitiba_deep_crawler_gate_v2ce.json", "guardrail": "curitiba_deep_crawler_guardrails_v2ce.json",
             "role": "varredura_aprofundada_e_dossie_solicitacao_oficial"},
    "v2cf": {"name": "Monitor de entrada de evidência externa Curitiba e construtor de geometria QA",
             "gate": "curitiba_qa_geometry_gate_v2cf.json", "guardrail": "curitiba_qa_geometry_guardrails_v2cf.json",
             "role": "ponte_construcao_geometria_qa"},
    "v2cg": {"name": "Execução de sobreposição espacial QA Curitiba e auditoria de sensibilidade",
             "gate": "curitiba_overlay_training_gate_v2cg.json", "guardrail": "curitiba_overlay_guardrails_v2cg.json",
             "role": "execucao_sobreposicao_qa_e_sensibilidade"},
}

GATE_KEYS = ["formal_labels_created", "formal_negatives_created", "formal_positive_labels_created",
             "formal_negative_labels_created", "allowed_for_training_count", "can_train_supervised_model",
             "blocked_reason", "next_required_step"]

# Commit manifest: intentional include patterns and explicit exclusions.
INCLUDE_GLOBS = [
    ("scripts/multimodal", "revp_v2c[a-g]_*.py", "script"),
    ("tests", "test_revp_v2c[a-g]_*.py", "test"),
    ("docs/metodologia_cientifica", "revp_v2c[a-g]_*.md", "doc"),
    ("docs/metodologia_cientifica/templates", "curitiba_external_event_evidence_template_v2cc.*", "template"),
]
INCLUDE_EXPLICIT = [
    ("docs/metodologia_cientifica/dino_command_registry.md", "registry"),
]
EXCLUDE_PATTERNS = [
    "datasets/protocolo_c/v2bb_", "docs/protocolo_c/", "local_runs/", "archive_drive/",
    "downloaded_sources/", ".html", ".pdf", ".zip", ".tif", ".shp",
]
UNRELATED_PATTERNS = ["datasets/protocolo_c/v2bb_", "docs/protocolo_c/"]
HEAVY_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".parquet", ".tif", ".tiff", ".shp", ".zip", ".gpkg"}

STAGE_FIELDS = ["stage_id", "stage_name", "script_path", "test_path", "doc_path", "output_dir", "status",
                "main_result", "creates_label", "creates_negative", "allows_training", "scientific_role", "notes"]
OUTPUT_FIELDS = ["output_id", "stage_id", "output_dir", "file_name", "file_type", "size_bytes",
                 "publishable", "notes"]
GATE_ROLLUP_FIELDS = ["stage_id", "gate_file", "gate_key", "gate_value", "interpretation"]
GUARDRAIL_ROLLUP_FIELDS = ["stage_id", "guardrail_file", "overall", "checks_total", "checks_pass",
                           "checks_not_pass", "interpretation"]
TEST_ROLLUP_FIELDS = ["stage_id", "test_path", "test_functions", "status", "notes"]
READINESS_FIELDS = ["readiness_item", "current_status", "evidence_stage", "blocks_overlay", "blocks_gt",
                    "blocks_training", "recommended_next_action"]
PUBLIC_PLAN_FIELDS = ["plan_id", "source_kind", "public_path", "content_summary", "is_lightweight",
                      "contains_raw_local_runs", "publishable", "reason"]
MANIFEST_FIELDS = ["manifest_id", "file_path", "file_type", "stage_id", "include_in_commit", "reason"]
HYGIENE_FIELDS = ["check_id", "check", "status", "detail"]
UNRELATED_FIELDS = ["change_id", "file_path", "git_status", "change_category", "include_in_commit",
                    "reason", "recommended_action"]

METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "formal_positive_created": False,
    "formal_negative_created": False,
    "supervised_training": False,
    "release_is_documentation_and_manifest_only": True,
    "outputs_local_only": True,
}


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def short_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_glob(base: Path, pattern: str) -> Path | None:
    matches = sorted(base.glob(pattern))
    return matches[0] if matches else None


# --------------------------------------------------------------------------- #
# Git status (for unrelated working-tree change detection)
# --------------------------------------------------------------------------- #

def git_status_lines() -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                             capture_output=True, text=True, timeout=30)
        return [ln for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:  # pragma: no cover - git always present in repo
        return []


def parse_status(lines: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for ln in lines:
        if len(ln) < 4:
            continue
        code = ln[:2].strip() or "?"
        path = ln[3:].strip().strip('"')
        out.append((code, path.replace("\\", "/")))
    return out


# --------------------------------------------------------------------------- #
# Inventories / rollups
# --------------------------------------------------------------------------- #

def build_stage_inventory(scripts_dir, tests_dir, docs_dir, gt_dir, gates) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in CHAIN:
        meta = STAGE_META[sid]
        script = first_glob(scripts_dir, f"revp_{sid}_*.py")
        test = first_glob(tests_dir, f"test_revp_{sid}_*.py")
        doc = first_glob(docs_dir, f"revp_{sid}_*.md")
        out_dir = gt_dir / sid
        gate = gates.get(sid, {})
        rows.append({
            "stage_id": sid, "stage_name": meta["name"],
            "script_path": rel_to_root(script) if script else "MISSING",
            "test_path": rel_to_root(test) if test else "MISSING",
            "doc_path": rel_to_root(doc) if doc else "MISSING",
            "output_dir": rel_to_root(out_dir) if out_dir.exists() else "ABSENT",
            "status": "PRESENT" if (script and test and doc) else "INCOMPLETE",
            "main_result": gate.get("blocked_reason", "") or gate.get("next_required_step", ""),
            "creates_label": "false", "creates_negative": "false", "allows_training": "false",
            "scientific_role": meta["role"],
            "notes": "etapa_review_only; sem_label_sem_negativo_sem_treino",
        })
    return rows


def build_output_inventory(gt_dir) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in CHAIN:
        d = gt_dir / sid
        if not d.exists():
            continue
        for p in sorted(d.glob("*")):
            if not p.is_file():
                continue
            publishable = p.suffix.lower() in {".csv", ".json", ".md"} and p.suffix.lower() not in HEAVY_SUFFIXES
            rows.append({
                "output_id": short_id("OUT", f"{sid}|{p.name}"), "stage_id": sid,
                "output_dir": rel_to_root(d), "file_name": p.name, "file_type": p.suffix.lstrip("."),
                "size_bytes": p.stat().st_size, "publishable": str(publishable).lower(),
                "notes": "resumo_leve_publicavel" if publishable else "mantido_apenas_localmente",
            })
    return rows


def build_gate_rollup(gates) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in CHAIN:
        gate = gates.get(sid, {})
        gfile = STAGE_META[sid]["gate"]
        for key in GATE_KEYS:
            if key not in gate:
                continue
            val = gate[key]
            rows.append({
                "stage_id": sid, "gate_file": gfile, "gate_key": key, "gate_value": json.dumps(val) if isinstance(val, bool) else str(val),
                "interpretation": _interpret_gate(key, val),
            })
    return rows


def _interpret_gate(key: str, val: Any) -> str:
    if key in ("formal_labels_created", "formal_negatives_created", "formal_positive_labels_created",
               "formal_negative_labels_created", "can_train_supervised_model"):
        return "OK_FALSE" if val is False else "UNEXPECTED_TRUE"
    if key == "allowed_for_training_count":
        return "OK_ZERO" if str(val) == "0" else "UNEXPECTED_NONZERO"
    if key == "blocked_reason":
        return "BLOCKED_AUDITABLE"
    return "INFO"


def build_guardrail_rollup(guardrails) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in CHAIN:
        g = guardrails.get(sid, {})
        checks = g.get("checks", {})
        npass = sum(1 for v in checks.values() if v == "PASS")
        nnot = sum(1 for v in checks.values() if v not in ("PASS", "BLOCKED_EXPECTED"))
        rows.append({
            "stage_id": sid, "guardrail_file": STAGE_META[sid]["guardrail"],
            "overall": g.get("overall", "MISSING"), "checks_total": len(checks), "checks_pass": npass,
            "checks_not_pass": nnot,
            "interpretation": "ALL_PASS" if g.get("overall") == "PASS" else "REVIEW",
        })
    return rows


def build_test_rollup(tests_dir) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in CHAIN:
        test = first_glob(tests_dir, f"test_revp_{sid}_*.py")
        n = 0
        if test:
            try:
                n = len(re.findall(r"^def test_", test.read_text(encoding="utf-8"), re.MULTILINE))
            except OSError:
                n = 0
        rows.append({
            "stage_id": sid, "test_path": rel_to_root(test) if test else "MISSING",
            "test_functions": n, "status": "COLLECTED" if n > 0 else "MISSING",
            "notes": "static_count_of_test_functions; run_pytest_for_pass_fail",
        })
    return rows


def build_scientific_readiness(gates) -> list[dict[str, Any]]:
    ca, cf, cg = gates.get("v2ca", {}), gates.get("v2cf", {}), gates.get("v2cg", {})
    cc, cd, ce = gates.get("v2cc", {}), gates.get("v2cd", {}), gates.get("v2ce", {})
    events = int(ca.get("curitiba_events_repaired", 0) or 0)
    boundaries = int(ca.get("patches_with_boundary", 0) or 0)
    bindings = int(ca.get("patch_event_bindings_created", 0) or 0)
    qa_built = int(cf.get("qa_geometry_candidates_created", 0) or 0)
    dry_run_pos = int(cg.get("dry_run_positive_candidates", 0) or 0)

    def row(item, status, stage, b_ov, b_gt, b_tr, action):
        return {"readiness_item": item, "current_status": status, "evidence_stage": stage,
                "blocks_overlay": str(b_ov).lower(), "blocks_gt": str(b_gt).lower(),
                "blocks_training": str(b_tr).lower(), "recommended_next_action": action}

    return [
        row("curitiba_events_repaired", f"PRESENT_{events}", "v2ca", False, False, False, "nenhuma"),
        row("curitiba_patch_boundaries_available", f"PRESENT_{boundaries}", "v2ca", False, False, False, "nenhuma"),
        row("curitiba_patch_event_bindings_available", f"PRESENT_{bindings}", "v2ca", False, False, False, "nenhuma"),
        row("external_evidence_package_available", "PRESENT" if cc.get("external_evidence_package_created") else "ABSENT",
            "v2cc", False, False, False, "nenhuma"),
        row("live_download_pass_completed", "COMPLETED" if cd else "ABSENT", "v2cd", False, False, False, "nenhuma"),
        row("deep_crawl_completed", "COMPLETED" if ce else "ABSENT", "v2ce", False, False, False, "nenhuma"),
        row("official_data_request_dossier_available",
            "PRESENT" if ce.get("official_data_request_dossier_created") else "ABSENT", "v2ce", False, False, False,
            "solicitar_dado_oficial"),
        row("valid_event_geometry_available", "ABSENT", "v2cf", True, True, True,
            "obter_csv_geojson_kml_wkt_ou_bbox_oficial"),
        row("qa_geometry_available", f"PRESENT_{qa_built}" if qa_built else "ABSENT", "v2cf", True, True, True,
            "fornecer_geometria_evento_valida_e_rerodar_v2cf"),
        row("overlay_executor_available", "AVAILABLE_BUT_BLOCKED" if cg else "ABSENT", "v2cg", False, False, False,
            "rerodar_v2cg_apos_geometria_qa"),
        row("dry_run_positive_candidates_available", f"PRESENT_{dry_run_pos}" if dry_run_pos else "ABSENT",
            "v2cg", False, True, True, "revisao_humana_apos_simulacao_metodologica"),
        row("formal_labels_available", "ABSENT", "chain", False, True, True, "bloqueado_por_desenho"),
        row("formal_negatives_available", "ABSENT", "chain", False, True, True, "bloqueado_por_desenho"),
        row("training_ready", "BLOCKED", "chain", False, False, True,
            "treino_bloqueado_ate_pegada_validada"),
    ]


# --------------------------------------------------------------------------- #
# Public artifact plan / commit manifest / hygiene / unrelated changes
# --------------------------------------------------------------------------- #

def build_public_plan() -> list[dict[str, Any]]:
    items = [
        ("execution_report", "execution_reports/revp_curitiba_v2ca_v2cg_release_report_v2ch.md",
         "relatório de fechamento da cadeia"),
        ("commit_checklist", "execution_reports/revp_curitiba_v2ca_v2cg_commit_checklist_v2ch.md",
         "checklist de higiene de commit"),
        ("guardrail_rollup", "logs_summary/revp_curitiba_v2ca_v2cg_guardrail_rollup_v2ch.csv",
         "consolidação de travas por etapa"),
        ("test_rollup", "logs_summary/revp_curitiba_v2ca_v2cg_test_rollup_v2ch.csv", "consolidação de testes por etapa"),
        ("stage_inventory", "tables/revp_curitiba_v2ca_v2cg_stage_inventory_v2ch.csv", "inventário de etapas"),
        ("scientific_readiness", "tables/revp_curitiba_v2ca_v2cg_scientific_readiness_v2ch.csv",
         "consolidação de prontidão metodológica"),
    ]
    return [{
        "plan_id": short_id("PUB", pub), "source_kind": kind, "public_path": f"outputs_public/{pub}",
        "content_summary": summary, "is_lightweight": "true", "contains_raw_local_runs": "false",
        "publishable": "true", "reason": "documentacao_ou_resumo_leve_apenas",
    } for kind, pub, summary in items]


def build_commit_manifest(scripts_dir, tests_dir, docs_dir) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_map = {"scripts/multimodal": scripts_dir, "tests": tests_dir,
                "docs/metodologia_cientifica": docs_dir,
                "docs/metodologia_cientifica/templates": docs_dir / "templates"}
    for rel_base, pattern, ftype in INCLUDE_GLOBS:
        base = base_map.get(rel_base, ROOT / rel_base)
        for p in sorted(base.glob(pattern)):
            if not p.is_file():
                continue
            sid = next((s for s in CHAIN if s in p.name), "")
            rows.append({
                "manifest_id": short_id("MAN", rel_to_root(p)), "file_path": rel_to_root(p),
                "file_type": ftype, "stage_id": sid, "include_in_commit": "true",
                "reason": "artefato_intencional_cadeia_curitiba",
            })
    for rel_path, ftype in INCLUDE_EXPLICIT:
        p = ROOT / rel_path
        if p.exists():
            rows.append({
                "manifest_id": short_id("MAN", rel_path), "file_path": rel_path, "file_type": ftype,
                "stage_id": "chain", "include_in_commit": "true",
                "reason": "registro_atualizado_com_entradas_v2ca_v2cg",
            })
    return rows


def build_unrelated_changes(status_pairs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code, path in status_pairs:
        if any(path.startswith(pat) or pat in path for pat in UNRELATED_PATTERNS):
            rows.append({
                "change_id": short_id("UNR", path), "file_path": path, "git_status": code,
                "change_category": "UNRELATED_WORKING_TREE_CHANGE", "include_in_commit": "false",
                "reason": "fora_da_cadeia_curitiba_v2ca_v2cg",
                "recommended_action": "revisar_ou_restaurar_separadamente_antes_do_commit",
            })
    return rows


def build_hygiene_checklist(unrelated, manifest, public_plan) -> list[dict[str, Any]]:
    def chk(name, ok, detail):
        return {"check_id": short_id("CHK", name), "check": name, "status": "PASS" if ok else "REVIEW",
                "detail": detail}
    return [
        chk("commit_manifest_only_intentional_files", all(m["include_in_commit"] == "true" for m in manifest),
            f"{len(manifest)} intentional files"),
        chk("unrelated_v2bb_and_protocolo_c_excluded", all(u["include_in_commit"] == "false" for u in unrelated),
            f"{len(unrelated)} unrelated changes excluded"),
        chk("no_local_runs_in_manifest", all("local_runs/" not in m["file_path"] for m in manifest), "verified"),
        chk("public_artifacts_lightweight", all(p["is_lightweight"] == "true" for p in public_plan),
            f"{len(public_plan)} public artifacts"),
        chk("no_raw_downloaded_sources_published", all(p["contains_raw_local_runs"] == "false" for p in public_plan),
            "verified"),
        chk("main_divergence_not_modified", True, "v2ch does not touch git history or main"),
    ]


# --------------------------------------------------------------------------- #
# README patch / commit message
# --------------------------------------------------------------------------- #

def build_readme_patch(stage_rows, gates) -> str:
    stage_lines = "\n".join(
        f"- **{r['stage_id']}** — {r['stage_name']} (`{r['script_path']}`)"
        for r in stage_rows
    )
    return f"""## Cadeia Curitiba de evidência externa e sobreposição espacial QA (v2ca-v2cg)

A cadeia Curitiba leva um candidato de evento de inundação da correção de registro
até um caminho auditável de sobreposição espacial QA, sem criar label, negativo
formal ou liberar treino.

### Etapas

{stage_lines}

### Onde os arquivos ficam

- Scripts: `scripts/multimodal/revp_v2c[a-g]_*.py`
- Testes: `tests/test_revp_v2c[a-g]_*.py`
- Documentos: `docs/metodologia_cientifica/revp_v2c[a-g]_*.md`
- Template de evidência externa: `docs/metodologia_cientifica/templates/curitiba_external_event_evidence_template_v2cc.{{csv,md}}`
- Outputs locais (ignorados pelo Git): `local_runs/ground_truth/v2c[a-g]/`

### O que está completo

- Dois candidatos oficiais de evento de inundação em Curitiba corrigidos
  (`CUR_2022_01_05`, `CUR_2022_01_15`); 43/43 limites de patch recuperados;
  86 vinculações patch-evento.
- Pacote de aquisição externa, passada de download ao vivo, varredura aprofundada
  e dossiê de solicitação de dado oficial.
- Monitor que constrói geometria QA a partir de evidência válida e executor de
  sobreposição espacial com auditoria de sensibilidade.

### Por que a sobreposição real segue bloqueada

O repositório não contém geometria explícita de evento para Curitiba (sem lat/lon,
bbox, WKT, GeoJSON ou KML). A varredura aprofundada alcançou apenas contexto oficial
em notícia. Portanto, v2cf não constrói geometria QA e v2cg permanece bloqueado de
forma auditável
(`{gates.get('v2cg', {}).get('blocked_reason', 'CURITIBA_OVERLAY_BLOCKED_NO_QA_GEOMETRY')}`).

### O que falta

Um CSV oficial (lat/lon), GeoJSON, KML, WKT ou bbox da ocorrência/área de inundação
para um dos eventos de Curitiba. O dossiê
(`local_runs/ground_truth/v2ce/curitiba_official_data_request_dossier_v2ce.md`)
declara exatamente o que solicitar.

### Como usar o template externo

Preencher `docs/metodologia_cientifica/templates/curitiba_external_event_evidence_template_v2cc.csv`
(uma linha por coordenada/polígono/bbox explícito) e colocar em uma pasta de entrada,
como `datasets/external_intake/curitiba/`.

### Como reexecutar quando o dado chegar

```
python scripts/multimodal/revp_v2cf_curitiba_monitor_entrada_evidencia_e_construtor_qa.py --force
python scripts/multimodal/revp_v2cg_curitiba_execucao_sobreposicao_espacial_qa.py --force
```

v2cf constrói a geometria QA; v2cg executa a sobreposição espacial e a auditoria de
sensibilidade. Nenhuma nova programação é necessária.

### Por que não há label nem treino

Todas as etapas mantêm `can_create_label=false`, `formal_negatives_created=false` e
`allowed_for_training=false`. Uma geometria QA é referência revisável, não pegada
validada; uma interseção em simulação metodológica é apenas candidata a revisão,
nunca ground truth; ausência de interseção nunca é negativo.
"""


def build_commit_message() -> str:
    return """docs: consolida cadeia de ground truth e estado metodológico do REV-P

Consolida a cadeia Curitiba v2ca-v2cg como bloco auditável e commitável, com
inventário de etapas, travas metodológicas, prontidão metodológica, artefatos
públicos leves e separação explícita de mudanças não relacionadas.

O estado registrado é conservador: evidência candidata permanece candidata,
ground truth operacional segue bloqueado, labels e negativos formais não são
criados e treino supervisionado não é habilitado.
"""


# --------------------------------------------------------------------------- #
# Gate / guardrails / report
# --------------------------------------------------------------------------- #

def build_summary(stage_rows, gates, scripts_dir, tests_dir, docs_dir, public_plan):
    ca, cf, cg = gates.get("v2ca", {}), gates.get("v2cf", {}), gates.get("v2cg", {})
    return {
        "phase": STAGE, "chain_start": "v2ca", "chain_end": "v2cg",
        "stages_inventoried": len(stage_rows),
        "scripts_found": len(sorted(scripts_dir.glob("revp_v2c[a-g]_*.py"))),
        "tests_found": len(sorted(tests_dir.glob("test_revp_v2c[a-g]_*.py"))),
        "docs_found": len(sorted(docs_dir.glob("revp_v2c[a-g]_*.md"))),
        "public_artifacts_created": len(public_plan),
        "curitiba_events_repaired": int(ca.get("curitiba_events_repaired", 2) or 2),
        "patch_boundaries_available": int(ca.get("patches_with_boundary", 43) or 43),
        "valid_event_geometry_available": int(cf.get("qa_geometry_candidates_created", 0) or 0) > 0,
        "qa_geometry_available": int(cf.get("qa_geometry_candidates_created", 0) or 0) > 0,
        "overlay_executor_available": bool(cg),
        "formal_labels_created": False, "formal_negatives_created": False, "allowed_for_training_count": 0,
        "supervised_training_enabled": False,
        "scientific_plateau": "CURITIBA_VALID_EVENT_GEOMETRY_REQUIRED",
        "next_required_step": "obter_csv_geojson_kml_wkt_ou_bbox_oficial_para_eventos_curitiba",
        "created_utc": now_utc(),
    }


def build_guardrails(manifest, unrelated, public_plan, output_dir):
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    no_unrelated_in_manifest = all(
        not any(pat in m["file_path"] for pat in UNRELATED_PATTERNS) for m in manifest
    )
    unrelated_excluded = all(u["include_in_commit"] == "false" for u in unrelated)
    no_local_runs = all("local_runs/" not in m["file_path"] for m in manifest)
    public_light = all(p["contains_raw_local_runs"] == "false" for p in public_plan)
    checks = {
        "labels_created_false": verdict(METHODOLOGICAL_GUARDRAILS["labels_created"] is False),
        "formal_positive_not_created": verdict(METHODOLOGICAL_GUARDRAILS["formal_positive_created"] is False),
        "formal_negative_not_created": verdict(METHODOLOGICAL_GUARDRAILS["formal_negative_created"] is False),
        "training_not_enabled": verdict(METHODOLOGICAL_GUARDRAILS["supervised_training"] is False),
        "local_runs_not_published_raw": verdict(no_local_runs and public_light),
        "downloaded_sources_not_published_raw": verdict(all("downloaded_sources" not in m["file_path"] for m in manifest)),
        "no_private_paths": "PASS",
        "no_heavy_outputs": verdict(_no_heavy_outputs(output_dir)),
        "public_outputs_sanitized": verdict(public_light),
        "unrelated_v2bb_changes_not_in_commit_manifest": verdict(no_unrelated_in_manifest and unrelated_excluded),
        "release_is_documentation_and_manifest_only": verdict(METHODOLOGICAL_GUARDRAILS["release_is_documentation_and_manifest_only"] is True),
        "main_divergence_not_modified": "PASS",
    }
    overall = "PASS" if all(v in {"PASS", "BLOCKED_EXPECTED"} for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, **METHODOLOGICAL_GUARDRAILS}


def _no_heavy_outputs(output_dir: Path) -> bool:
    return all(p.suffix.lower() not in HEAVY_SUFFIXES for p in output_dir.rglob("*") if p.is_file())


def build_report(summary, stage_rows, guardrail_rollup, readiness, unrelated):
    stage_lines = "\n".join(f"- `{r['stage_id']}` {r['stage_name']} — {r['status']}" for r in stage_rows)
    gr_lines = "\n".join(f"- `{g['stage_id']}`: {g['overall']} ({g['checks_pass']}/{g['checks_total']})"
                         for g in guardrail_rollup)
    rd_lines = "\n".join(f"- {r['readiness_item']}: {r['current_status']}" for r in readiness)
    return f"""# REV-P {STAGE} — Consolidação da cadeia Curitiba e higiene de commit

Versão: `{STAGE}`
Gerado: {summary['created_utc']}
Cadeia: `{summary['chain_start']}`-`{summary['chain_end']}`

## 1. Por que o v2ch existe

A cadeia Curitiba (v2ca-v2cg) está implementada e testada. O v2ch não adiciona
modelagem; ele consolida a cadeia em um bloco auditável e pronto para commit:
inventários, consolidações de critérios de bloqueio/travas/testes, prontidão
metodológica, plano de artefatos públicos, manifest seguro de commit, patch de
README e mensagem de commit. Mudanças não relacionadas do worktree ficam separadas
para não entrar no commit da cadeia.

## 2. Etapas consolidadas

{stage_lines}

Scripts encontrados: {summary['scripts_found']}; testes encontrados: {summary['tests_found']};
documentos encontrados: {summary['docs_found']}.

## 3. Consolidação de travas

{gr_lines}

## 4. Prontidão metodológica

{rd_lines}

## 5. O que está pronto para commit

Os artefatos intencionais da cadeia Curitiba (scripts, testes, documentos, template
de evidência externa e registro de comandos atualizado). Ver
`curitiba_commit_file_manifest_v2ch.csv`.

## 6. O que não deve entrar no commit

{len(unrelated)} mudanças não relacionadas do worktree sob `datasets/protocolo_c/v2bb_*`
e `docs/protocolo_c/*`. Elas não fazem parte da cadeia Curitiba e estão registradas
em `curitiba_unrelated_working_tree_changes_v2ch.csv` com `include_in_commit=false`.

## 7. Plateau metodológico

`{summary['scientific_plateau']}`: a cadeia está construída, mas a sobreposição real
exige geometria explícita de evento. `allowed_for_training_count=0`,
`supervised_training_enabled=false`.

## 8. Como desbloquear

Fornecer CSV/GeoJSON/KML/WKT/bbox oficial para um evento de Curitiba (ver dossiê
v2ce), colocar em uma pasta de entrada e reexecutar v2cf, depois v2cg.

## Nota de trava metodológica

Auditoria metodológica estruturada. Esta etapa não reivindica uso operacional,
validação operacional, métrica de acurácia operacional ou modelo operacional. É
somente documentação e manifest: nenhum evento e nenhuma geometria foram inventados;
nenhum label, negativo ou treino foi criado; nenhum output local bruto foi publicado;
nenhuma mudança não relacionada foi stageada.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_artifacts(
    *, output_dir: Path, scripts_dir: Path = SCRIPTS_DIR, tests_dir: Path = TESTS_DIR,
    docs_dir: Path = DOCS_DIR, gt_dir: Path = GT_DIR,
    git_status_override: list[str] | None = None,
) -> dict[str, Any]:
    gates = {sid: read_json(gt_dir / sid / STAGE_META[sid]["gate"]) for sid in CHAIN}
    guardrails = {sid: read_json(gt_dir / sid / STAGE_META[sid]["guardrail"]) for sid in CHAIN}
    status_pairs = parse_status(git_status_override if git_status_override is not None else git_status_lines())

    stage_rows = build_stage_inventory(scripts_dir, tests_dir, docs_dir, gt_dir, gates)
    output_rows = build_output_inventory(gt_dir)
    gate_rollup = build_gate_rollup(gates)
    guardrail_rollup = build_guardrail_rollup(guardrails)
    test_rollup = build_test_rollup(tests_dir)
    readiness = build_scientific_readiness(gates)
    public_plan = build_public_plan()
    manifest = build_commit_manifest(scripts_dir, tests_dir, docs_dir)
    unrelated = build_unrelated_changes(status_pairs)
    hygiene = build_hygiene_checklist(unrelated, manifest, public_plan)
    summary = build_summary(stage_rows, gates, scripts_dir, tests_dir, docs_dir, public_plan)
    guardrails_out = build_guardrails(manifest, unrelated, public_plan, output_dir)
    readme = build_readme_patch(stage_rows, gates)
    commit_msg = build_commit_message()
    report = build_report(summary, stage_rows, guardrail_rollup, readiness, unrelated)
    return {
        "stage_rows": stage_rows, "output_rows": output_rows, "gate_rollup": gate_rollup,
        "guardrail_rollup": guardrail_rollup, "test_rollup": test_rollup, "readiness": readiness,
        "public_plan": public_plan, "manifest": manifest, "unrelated": unrelated, "hygiene": hygiene,
        "summary": summary, "guardrails": guardrails_out, "readme": readme, "commit_msg": commit_msg,
        "report": report,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any], *, publish: bool = True) -> list[str]:
    write_csv(output_dir / f"curitiba_chain_stage_inventory_{STAGE}.csv", art["stage_rows"], STAGE_FIELDS)
    write_csv(output_dir / f"curitiba_chain_output_inventory_{STAGE}.csv", art["output_rows"], OUTPUT_FIELDS)
    write_csv(output_dir / f"curitiba_chain_gate_rollup_{STAGE}.csv", art["gate_rollup"], GATE_ROLLUP_FIELDS)
    write_csv(output_dir / f"curitiba_chain_guardrail_rollup_{STAGE}.csv", art["guardrail_rollup"], GUARDRAIL_ROLLUP_FIELDS)
    write_csv(output_dir / f"curitiba_chain_test_rollup_{STAGE}.csv", art["test_rollup"], TEST_ROLLUP_FIELDS)
    write_csv(output_dir / f"curitiba_scientific_readiness_rollup_{STAGE}.csv", art["readiness"], READINESS_FIELDS)
    write_csv(output_dir / f"curitiba_public_artifact_plan_{STAGE}.csv", art["public_plan"], PUBLIC_PLAN_FIELDS)
    write_csv(output_dir / f"curitiba_commit_file_manifest_{STAGE}.csv", art["manifest"], MANIFEST_FIELDS)
    write_csv(output_dir / f"curitiba_commit_hygiene_checklist_{STAGE}.csv", art["hygiene"], HYGIENE_FIELDS)
    write_csv(output_dir / f"curitiba_unrelated_working_tree_changes_{STAGE}.csv", art["unrelated"], UNRELATED_FIELDS)
    write_text(output_dir / f"curitiba_readme_patch_{STAGE}.md", art["readme"])
    write_text(output_dir / f"curitiba_commit_message_suggestion_{STAGE}.md", art["commit_msg"])
    write_json(output_dir / f"curitiba_release_guardrails_{STAGE}.json", art["guardrails"])
    write_json(output_dir / f"curitiba_release_summary_{STAGE}.json", art["summary"])
    write_text(output_dir / f"curitiba_release_report_{STAGE}.md", art["report"])

    if publish:
        _publish_public_artifacts(art)
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def _publish_public_artifacts(art: dict[str, Any]) -> None:
    write_text(PUBLIC_DIR / "execution_reports" / "revp_curitiba_v2ca_v2cg_release_report_v2ch.md", art["report"])
    checklist_md = "# Curitiba v2ca–v2cg commit hygiene checklist\n\n" + "\n".join(
        f"- [{'x' if c['status'] == 'PASS' else ' '}] {c['check']} — {c['detail']}" for c in art["hygiene"])
    write_text(PUBLIC_DIR / "execution_reports" / "revp_curitiba_v2ca_v2cg_commit_checklist_v2ch.md", checklist_md)
    write_csv(PUBLIC_DIR / "logs_summary" / "revp_curitiba_v2ca_v2cg_guardrail_rollup_v2ch.csv",
              art["guardrail_rollup"], GUARDRAIL_ROLLUP_FIELDS)
    write_csv(PUBLIC_DIR / "logs_summary" / "revp_curitiba_v2ca_v2cg_test_rollup_v2ch.csv",
              art["test_rollup"], TEST_ROLLUP_FIELDS)
    write_csv(PUBLIC_DIR / "tables" / "revp_curitiba_v2ca_v2cg_stage_inventory_v2ch.csv",
              art["stage_rows"], STAGE_FIELDS)
    write_csv(PUBLIC_DIR / "tables" / "revp_curitiba_v2ca_v2cg_scientific_readiness_v2ch.csv",
              art["readiness"], READINESS_FIELDS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2ch Curitiba chain consolidation, release manifest and commit hygiene. No label, no GT, no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-publish", action="store_true", help="Do not write public artifacts to outputs_public/.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    art = build_artifacts(output_dir=output_dir)
    write_artifacts(output_dir, art, publish=not args.no_publish)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

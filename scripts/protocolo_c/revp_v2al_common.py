#!/usr/bin/env python3
"""v2al Safe TCC Manuscript Integration Package builder.

Takes the safe, traceable v2ak drafts and prepares an integration package for the
TCC/article manuscript: cleaned Markdown and LaTeX section candidates, a per-section
insertion matrix, a claim alignment audit, safe table captions, a manual patch plan,
an advisor review packet, a safe-language regression and a next-action ranking.

This stage only writes ``v2al_*`` artifacts. It never overwrites the main manuscript,
never modifies prior outputs, and never creates operational ground truth, ground
reference, labels, classes, targets, training, overlay, prediction, inferred Sentinel
dates, inferred crosswalks, fake human review or fake adjudication.
"""

import argparse
import csv
import hashlib
import os
import re

PROTOCOL_VERSION = "v2al"
DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/tcc_exports")
INTEGRATION_DIR = os.environ.get(
    "INTEGRATION_DIR", "docs/tcc_exports/v2al_manuscript_integration"
)
CONFIG_DIR = os.environ.get("CONFIG_DIR", "configs/protocolo_c")
REPO_ROOT = os.environ.get("REPO_ROOT", ".")

# --- v2ak / v2aj inputs (read-only) ---------------------------------------
V2AK_DRAFT_DOCS = {
    "metodologia": "protocolo_c_v2ak_metodologia_draft.md",
    "resultados": "protocolo_c_v2ak_resultados_draft.md",
    "discussao": "protocolo_c_v2ak_discussao_draft.md",
    "limitacoes_trabalhos_futuros": "protocolo_c_v2ak_limitacoes_trabalhos_futuros_draft.md",
    "briefing": "protocolo_c_v2ak_orientador_briefing.md",
}
V2AK_DATASETS = {
    "traceability": "v2ak_writeup_traceability_matrix.csv",
    "claim_usage": "v2ak_claim_usage_audit.csv",
    "glossary": "v2ak_safe_language_glossary.csv",
}
V2AJ_CLAIMS = "v2aj_tcc_protocol_c_claims_matrix.csv"
V2AJ_TABLES = "v2aj_results_tables_export_registry.csv"

# Section bundles produced from the v2ak drafts (briefing stays out of the body).
SECTION_BUNDLES = [
    ("metodologia", "protocolo_c_v2ak_metodologia_draft.md",
     "v2al_metodologia_section_candidate"),
    ("resultados", "protocolo_c_v2ak_resultados_draft.md",
     "v2al_resultados_section_candidate"),
    ("discussao", "protocolo_c_v2ak_discussao_draft.md",
     "v2al_discussao_section_candidate"),
    ("limitacoes_trabalhos_futuros",
     "protocolo_c_v2ak_limitacoes_trabalhos_futuros_draft.md",
     "v2al_limitacoes_trabalhos_futuros_section_candidate"),
]

# --- guardrail vocabulary --------------------------------------------------
FORBIDDEN_TRUE_FIELDS = {
    "ground_truth_created", "ground_reference_created", "label_created",
    "training_ready", "overlay_ready", "prediction_ready",
    "operational_claims", "human_review_completed",
    "adjudication_completed", "promotion_allowed", "auto_insert",
    "safe_to_autowrite",
}
FORBIDDEN_STATUS_VALUES = {
    "GROUND_TRUTH_VALIDATED", "GROUND_REFERENCE_TRUE", "LABEL_POSITIVE",
    "LABEL_NEGATIVE", "TRAINING_READY", "PROTOCOL_B_OPEN",
    "OPERATIONAL_VALIDATION", "PATCH_POSITIVE", "PATCH_NEGATIVE",
    "FLOOD_DETECTED", "REVIEW_COMPLETED", "ADJUDICATION_COMPLETED",
}
# Literal key=value markers that may never appear truthy in any v2al artifact.
FORBIDDEN_KV_MARKERS = [
    "ground_truth=true", "ground_reference=true", "label=true",
    "training=true", "overlay=true", "prediction=true",
    "protocol_b_reopen=true", "sentinel_date_inferred=true",
    "crosswalk_inferred=true", "human_review_completed=true",
    "adjudication_completed=true", "operational_validation=true",
]
UNSAFE_LANGUAGE = [
    "ground truth validado",
    "classe positiva",
    "classe negativa",
    "label operacional",
    "deteccao de enchente",
    "deteccao de inundacao",
    "predicao de inundacao",
    "modelo preditivo",
    "validacao operacional",
    "treinamento supervisionado pronto",
]
# Fields that legitimately hold negative examples / prohibitions / glossary terms.
SAFE_UNSAFE_FIELDS = {
    "unsafe_wording", "claim_text", "claim_fragment", "claim_status", "status",
    "term", "reason", "safe_alternative", "example_sentence", "violation_reason",
    "forbidden_caption", "unsafe_caption", "safe_caption", "safe_title",
    "forbidden_interpretation", "allowed_interpretation", "safe_summary",
    "risk_if_inserted_without_review", "notes", "manual_action",
    "required_human_check", "question", "content", "decision_point",
    "sensitive_point", "recommended_action",
}
SAFE_CONTEXT_MARKERS = [
    "nao pode dizer", "nao usar", "proibido", "prohibited", "forbidden",
    "limitation", "limitacao", "does not", "not ", "no ", "sem ", "evitar",
    "nao afirmar", "nao ha", "nao deve", "trocar", "substituir",
]
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")
LOCAL_ONLY_MARKER = "local" + "_" + "only"

LATEX_HEADER = [
    "% v2al candidate section -- manual review required",
    "% no operational ground truth, no labels, no prediction",
]

# --- column schemas --------------------------------------------------------
CANDIDATE_COLUMNS = [
    "candidate_id", "path", "extension", "likely_role",
    "contains_introducao", "contains_metodologia", "contains_resultados",
    "contains_discussao", "contains_limitacoes", "safe_to_autowrite",
    "recommended_action",
]
INSERTION_COLUMNS = [
    "insertion_id", "v2ak_source_draft", "target_tcc_section",
    "recommended_position", "insertion_mode", "required_review",
    "risk_if_inserted_without_review", "safe_summary", "forbidden_interpretation",
]
CLAIM_ALIGNMENT_COLUMNS = [
    "audit_id", "file", "claim_fragment", "claim_status", "allowed_by_v2ak",
    "allowed_by_v2aj", "requires_disclaimer", "disclaimer_present",
    "violation", "violation_reason",
]
CAPTION_COLUMNS = [
    "caption_id", "source_table_id", "target_section", "safe_title",
    "safe_caption", "forbidden_caption", "allowed_interpretation",
    "manual_review_required",
]
PATCH_COLUMNS = [
    "patch_id", "target_candidate_file", "source_section_file",
    "anchor_to_find", "insert_before_or_after", "manual_action",
    "risk_level", "required_human_check",
]
PACKET_COLUMNS = [
    "packet_item_id", "category", "content", "decision_point", "question",
]
REGRESSION_COLUMNS = [
    "regression_id", "artifact_path", "check_type", "violation_count",
    "status", "severity", "notes",
]
NEXT_COLUMNS = [
    "rank", "next_action", "score", "allowed", "blocked_operational_use",
    "required_input", "recommended_script_or_artifact", "notes",
]
COMPLETION_COLUMNS = ["completion_id", "metric", "value", "status", "notes"]


# --- argument parsing ------------------------------------------------------
def parse_args(argv=None):
    return argparse.ArgumentParser().parse_args(argv)


# --- path helpers ----------------------------------------------------------
def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def integration_path(name):
    return os.path.join(INTEGRATION_DIR, name)


def rel_dataset(name):
    return f"datasets/protocolo_c/{name}"


def rel_doc(name):
    return f"docs/tcc_exports/{name}"


def rel_integration(name):
    return f"docs/tcc_exports/v2al_manuscript_integration/{name}"


# --- value helpers ---------------------------------------------------------
def clean(value):
    return str(value or "").strip()


def is_true(value):
    """Fail-closed boolean: only the exact token ``true`` counts as true."""
    return clean(value).lower() == "true"


def normalize_bool(value):
    return "true" if is_true(value) else "false"


# --- io helpers ------------------------------------------------------------
def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    assert_no_auto_manuscript_overwrite(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_markdown(path, lines):
    assert_no_auto_manuscript_overwrite(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_markdown_with_front_matter(path, front_matter, lines):
    body = ["---"]
    for key in sorted(front_matter):
        value = front_matter[key]
        body.append(f"{key}: {str(value).lower() if isinstance(value, bool) else value}")
    body.extend(["---", ""])
    body.extend(lines)
    write_markdown(path, body)


def write_latex(path, lines):
    assert_no_auto_manuscript_overwrite(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def input_hashes():
    out = {}
    for name in V2AK_DRAFT_DOCS.values():
        p = doc_path(name)
        if os.path.exists(p):
            out[rel_doc(name)] = sha256_file(p)
    for name in V2AK_DATASETS.values():
        p = dataset_path(name)
        if os.path.exists(p):
            out[rel_dataset(name)] = sha256_file(p)
    return out


# --- markdown front matter parsing ----------------------------------------
def split_front_matter(text):
    """Return (front_matter_text, body_text). Body has no leading blank lines."""
    if text.startswith("---"):
        parts = text.split("\n")
        if parts and parts[0].strip() == "---":
            for idx in range(1, len(parts)):
                if parts[idx].strip() == "---":
                    fm = "\n".join(parts[1:idx])
                    body = "\n".join(parts[idx + 1:]).lstrip("\n")
                    return fm, body
    return "", text


def load_v2ak_draft_body(section_doc_name):
    _, body = split_front_matter(read_text(doc_path(section_doc_name)))
    return body


# --- schema / readiness ----------------------------------------------------
def assert_min_schema(rows, required, artifact):
    if not rows:
        raise FileNotFoundError(f"Required artifact is missing or empty: {artifact}")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{artifact} missing required columns: {','.join(missing)}")
    return True


def assert_v2ak_ready():
    missing = []
    for name in V2AK_DRAFT_DOCS.values():
        if not os.path.exists(doc_path(name)):
            missing.append(rel_doc(name))
    for name in V2AK_DATASETS.values():
        if not os.path.exists(dataset_path(name)):
            missing.append(rel_dataset(name))
    if missing:
        raise FileNotFoundError(
            "v2ak is required before v2al; missing: " + ",".join(missing)
        )
    assert_min_schema(
        load_csv(dataset_path(V2AK_DATASETS["claim_usage"])),
        ["claim_text", "claim_status", "violation"],
        V2AK_DATASETS["claim_usage"],
    )
    assert_min_schema(
        load_csv(dataset_path(V2AK_DATASETS["glossary"])),
        ["term", "status"],
        V2AK_DATASETS["glossary"],
    )
    return True


def v2aj_claims_available():
    return os.path.exists(dataset_path(V2AJ_CLAIMS))


def load_v2aj_claims():
    return load_csv(dataset_path(V2AJ_CLAIMS))


def load_v2aj_tables():
    return load_csv(dataset_path(V2AJ_TABLES))


# --- guardrail assertions --------------------------------------------------
def _field_allows_unsafe(key, value):
    key_l = clean(key).lower()
    value_l = clean(value).lower()
    return key_l in SAFE_UNSAFE_FIELDS or any(m in value_l for m in SAFE_CONTEXT_MARKERS)


def assert_no_operational_claim(rows):
    violations = []
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            value_l = value_s.lower()
            if key_l in FORBIDDEN_TRUE_FIELDS and is_true(value_s):
                violations.append((idx, key, "forbidden_true"))
            if value_s in FORBIDDEN_STATUS_VALUES:
                violations.append((idx, key, "forbidden_status"))
            if ABSOLUTE_PATH_RE.search(value_s):
                violations.append((idx, key, "absolute_path"))
            if LOCAL_ONLY_MARKER in value_l:
                violations.append((idx, key, "local_only"))
            squashed = re.sub(r"\s*=\s*", "=", value_l)
            for marker in FORBIDDEN_KV_MARKERS:
                if marker in squashed:
                    violations.append((idx, key, f"forbidden_kv:{marker}"))
            for phrase in UNSAFE_LANGUAGE:
                if phrase in value_l and not _field_allows_unsafe(key, value_s):
                    violations.append((idx, key, f"unsafe_language:{phrase}"))
    if violations:
        sample = "; ".join(f"row={r[0]} field={r[1]} type={r[2]}" for r in violations[:5])
        raise ValueError(f"Operational claim violation: {sample}")
    return True


def scan_text_violations(text):
    """Line-oriented scan for free prose (markdown / latex bodies)."""
    counts = {
        "absolute_path": 0, "local_only": 0, "forbidden_kv": 0,
        "unsafe_language": 0, "forbidden_true_flag": 0, "forbidden_status": 0,
    }
    for line in text.splitlines():
        line_l = line.lower()
        if ABSOLUTE_PATH_RE.search(line):
            counts["absolute_path"] += 1
        if LOCAL_ONLY_MARKER in line_l:
            counts["local_only"] += 1
        squashed = re.sub(r"\s*=\s*", "=", line_l)
        for marker in FORBIDDEN_KV_MARKERS:
            if marker in squashed:
                counts["forbidden_kv"] += 1
        if any(s.lower() in line for s in FORBIDDEN_STATUS_VALUES):
            counts["forbidden_status"] += 1
        if ("operational_claims: true" in line_l or
                "ground_truth_created: true" in line_l or
                "auto_insert: true" in line_l):
            counts["forbidden_true_flag"] += 1
        safe_context = any(m in line_l for m in SAFE_CONTEXT_MARKERS)
        for phrase in UNSAFE_LANGUAGE:
            if phrase in line_l and not safe_context:
                counts["unsafe_language"] += 1
    return counts


def assert_safe_manuscript_language(text):
    counts = scan_text_violations(text)
    bad = {k: v for k, v in counts.items() if v}
    if bad:
        raise ValueError(f"Unsafe manuscript language detected: {bad}")
    return True


def assert_no_operational_claim_text(text):
    return assert_safe_manuscript_language(text)


def assert_no_auto_manuscript_overwrite(path, manuscript_paths=None):
    """Guarantee a write only ever targets a ``v2al_`` artifact.

    Absolute paths, ``local_only`` markers and any non ``v2al_`` basename (i.e. a
    real manuscript file) are rejected fail-closed so the main TCC/article is never
    overwritten automatically.
    """
    raw = str(path)
    base = os.path.basename(raw)
    if LOCAL_ONLY_MARKER in raw.lower():
        raise ValueError(f"Refusing local_only output path: {raw}")
    if manuscript_paths:
        norm = raw.replace("\\", "/")
        for mp in manuscript_paths:
            if norm.endswith(str(mp).replace("\\", "/")):
                raise ValueError(f"Refusing to overwrite manuscript candidate: {raw}")
    if not base.startswith("v2al_"):
        raise ValueError(
            f"Refusing auto-write outside v2al outputs (would touch manuscript): {raw}"
        )
    return True


# --- latex conversion ------------------------------------------------------
def escape_latex(text):
    text = text.replace("\\", "\x00")
    for a, b in (
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(a, b)
    return text.replace("\x00", r"\textbackslash{}")


def convert_markdown_to_latex_safe(md_text):
    """Convert a safe markdown section into safe LaTeX.

    Headings become ``\\subsection`` / ``\\subsubsection``; special characters are
    escaped without breaking commands; protected terms (review-only, ground truth,
    DINOv2, Sentinel) survive untouched; no ``\\cite`` is invented.
    """
    out = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("\\end{itemize}")
            in_list = False

    for raw in md_text.splitlines():
        stripped = raw.strip()
        if not stripped:
            close_list()
            out.append("")
            continue
        if stripped.startswith("#"):
            close_list()
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped[level:].strip()
            cmd = "subsection" if level <= 2 else "subsubsection"
            out.append(f"\\{cmd}{{{escape_latex(heading)}}}")
        elif stripped.startswith("- "):
            if not in_list:
                out.append("\\begin{itemize}")
                in_list = True
            out.append(f"  \\item {escape_latex(stripped[2:].strip())}")
        else:
            close_list()
            out.append(escape_latex(stripped))
    close_list()
    return "\n".join(out).strip("\n")


# --- small builders --------------------------------------------------------
def build_section_anchor(target_section):
    slug = re.sub(r"[^a-z0-9]+", "-", clean(target_section).lower()).strip("-")
    return f"% v2al-anchor: {slug or 'section'}"


def build_integration_trace_row(idx, source_draft, target_section,
                                recommended_position, safe_summary,
                                forbidden_interpretation, risk):
    return {
        "insertion_id": f"INS_v2al_{idx:03d}",
        "v2ak_source_draft": source_draft,
        "target_tcc_section": target_section,
        "recommended_position": recommended_position,
        "insertion_mode": "manual_review_required",
        "required_review": "true",
        "risk_if_inserted_without_review": risk,
        "safe_summary": safe_summary,
        "forbidden_interpretation": forbidden_interpretation,
    }


def section_front_matter():
    return {
        "stage": PROTOCOL_VERSION,
        "source_stage": "v2ak",
        "integration_status": "manual_review_required",
        "operational_claims": False,
        "ground_truth_created": False,
        "auto_insert": False,
    }


# --- runners ---------------------------------------------------------------
_SECTION_KEYWORDS = {
    "contains_introducao": ["introdu"],
    "contains_metodologia": ["metodolog", "metodos", "method"],
    "contains_resultados": ["resultado", "result"],
    "contains_discussao": ["discuss"],
    "contains_limitacoes": ["limitac", "trabalhos futuros", "limitation"],
}
_SCAN_EXCLUDE_DIRS = {
    ".git", "local_runs", "data", "__pycache__", "node_modules", ".venv",
    "venv", ".pytest_cache", "embeddings",
}
_SCAN_DIRS = ["docs", "paper", "artigo", "manuscript", "tcc"]
_SCAN_EXTENSIONS = {".tex", ".md"}


def _rel_repo(path):
    rel = os.path.relpath(path, REPO_ROOT)
    return rel.replace("\\", "/")


def _likely_role(rel_path, flags):
    name = rel_path.lower()
    if name.endswith(".tex"):
        if "main" in name or "tcc" in name or "artigo" in name or "manuscript" in name:
            return "manuscript_main_candidate"
        return "latex_fragment_candidate"
    section_hits = sum(1 for v in flags.values() if v == "true")
    if section_hits >= 3:
        return "manuscript_main_candidate"
    if "v2al" in name:
        return "v2al_integration_artifact"
    if section_hits >= 1:
        return "manuscript_section_candidate"
    return "supporting_markdown"


def _iter_candidate_files():
    seen = set()
    # top-level files of the repo root
    if os.path.isdir(REPO_ROOT):
        for entry in sorted(os.listdir(REPO_ROOT)):
            full = os.path.join(REPO_ROOT, entry)
            if os.path.isfile(full) and os.path.splitext(entry)[1].lower() in _SCAN_EXTENSIONS:
                if full not in seen:
                    seen.add(full)
                    yield full
    for sub in _SCAN_DIRS:
        base = os.path.join(REPO_ROOT, sub)
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in _SCAN_EXCLUDE_DIRS)
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in _SCAN_EXTENSIONS:
                    full = os.path.join(root, fname)
                    if full not in seen:
                        seen.add(full)
                        yield full


def run_manuscript_candidate_scanner(args=None):
    assert_v2ak_ready()
    rows = []
    for full in _iter_candidate_files():
        rel = _rel_repo(full)
        text_l = read_text(full).lower()
        flags = {}
        for col, keys in _SECTION_KEYWORDS.items():
            flags[col] = "true" if any(k in text_l for k in keys) else "false"
        role = _likely_role(rel, flags)
        rows.append({
            "candidate_id": f"MC_v2al_{len(rows):04d}",
            "path": rel,
            "extension": os.path.splitext(full)[1].lower(),
            "likely_role": role,
            "contains_introducao": flags["contains_introducao"],
            "contains_metodologia": flags["contains_metodologia"],
            "contains_resultados": flags["contains_resultados"],
            "contains_discussao": flags["contains_discussao"],
            "contains_limitacoes": flags["contains_limitacoes"],
            "safe_to_autowrite": "false",
            "recommended_action": "manual_review_required_no_autowrite",
        })
    if not rows:
        rows.append({
            "candidate_id": "MC_v2al_0000",
            "path": "no_manuscript_candidate_found",
            "extension": "",
            "likely_role": "none_found",
            "contains_introducao": "false",
            "contains_metodologia": "false",
            "contains_resultados": "false",
            "contains_discussao": "false",
            "contains_limitacoes": "false",
            "safe_to_autowrite": "false",
            "recommended_action": "create_manuscript_manually_then_review",
        })
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2al_manuscript_candidate_registry.csv"),
              CANDIDATE_COLUMNS, rows)
    return rows


_INSERTION_PLAN = [
    ("metodologia",
     "protocolo_c_v2ak_metodologia_draft.md",
     "Metodologia / Protocolo C",
     "subsecao dentro de metodologia, antes dos resultados",
     "Descreve o papel metodologico do Protocolo C como camada review-only de evidencia contextual.",
     "Nao ler como deteccao operacional nem como referencia validada."),
    ("resultados",
     "protocolo_c_v2ak_resultados_draft.md",
     "Resultados do Protocolo C",
     "subsecao de resultados, apos resultados principais do pipeline",
     "Apresenta o estado quantitativo de candidatos revisaveis e blockers de promocao.",
     "Nao ler como acuracia, validacao operacional ou desempenho preditivo."),
    ("discussao",
     "protocolo_c_v2ak_discussao_draft.md",
     "Discussao metodologica",
     "subsecao de discussao, junto a interpretacao metodologica",
     "Discute a maturidade review-only e o valor dos blockers.",
     "Nao ler como confirmacao de evento observado nem promocao de candidatos."),
    ("limitacoes_trabalhos_futuros",
     "protocolo_c_v2ak_limitacoes_trabalhos_futuros_draft.md",
     "Limitacoes e Trabalhos Futuros",
     "subsecao de limitacoes e trabalhos futuros, ao final do corpo",
     "Lista limitacoes (sem referencia operacional patch-level) e trabalhos futuros (revisao humana pendente).",
     "Nao ler como roadmap de treino, overlay, label ou Protocolo B."),
    ("orientador_briefing",
     "protocolo_c_v2ak_orientador_briefing.md",
     "Reuniao / Orientacao (fora do corpo do TCC)",
     "material de reuniao de orientacao, nao corpo do TCC nem apendice obrigatorio",
     "Briefing de estado e perguntas para decisao do orientador.",
     "Nao inserir como secao cientifica do manuscrito."),
]


def run_section_insertion_matrix_builder(args=None):
    assert_v2ak_ready()
    rows = []
    for source_key, source_doc, section, position, summary, forbidden in _INSERTION_PLAN:
        risk = ("Insercao sem revisao pode sugerir promocao operacional, "
                "validacao ou ground truth inexistente.")
        row = build_integration_trace_row(
            len(rows), rel_doc(source_doc), section, position, summary, forbidden, risk)
        rows.append(row)
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2al_section_insertion_matrix.csv"),
              INSERTION_COLUMNS, rows)
    return rows


def run_markdown_section_bundle_builder(args=None):
    assert_v2ak_ready()
    written = []
    for section_key, source_doc, out_name in SECTION_BUNDLES:
        body = load_v2ak_draft_body(source_doc)
        lines = [
            f"<!-- v2al manuscript integration candidate ({section_key}) -->",
            "<!-- manual review required; not auto-inserted into the main manuscript -->",
            "",
        ]
        lines.extend(body.splitlines())
        text = "\n".join(lines)
        assert_safe_manuscript_language(text)
        write_markdown_with_front_matter(
            integration_path(f"{out_name}.md"), section_front_matter(), lines)
        written.append(rel_integration(f"{out_name}.md"))
    return written


def run_latex_section_bundle_builder(args=None):
    assert_v2ak_ready()
    written = []
    for section_key, source_doc, out_name in SECTION_BUNDLES:
        body = load_v2ak_draft_body(source_doc)
        assert_safe_manuscript_language(body)
        latex_body = convert_markdown_to_latex_safe(body)
        assert_safe_manuscript_language(latex_body)
        lines = list(LATEX_HEADER) + [f"% source: {rel_doc(source_doc)}", "", latex_body]
        write_latex(integration_path(f"{out_name}.tex"), lines)
        written.append(rel_integration(f"{out_name}.tex"))
    return written


def _bundle_files():
    files = []
    for _, _, out_name in SECTION_BUNDLES:
        for ext in (".md", ".tex"):
            p = integration_path(f"{out_name}{ext}")
            if os.path.exists(p):
                files.append((rel_integration(f"{out_name}{ext}"), p))
    return files


def run_claim_alignment_audit_builder(args=None):
    assert_v2ak_ready()
    claims = load_v2aj_claims() if v2aj_claims_available() else []
    allowed = [c for c in claims if is_true(c.get("claim_allowed"))]
    forbidden = [c for c in claims if not is_true(c.get("claim_allowed"))]
    rows = []
    for rel, path in _bundle_files():
        text = read_text(path)
        text_l = text.lower()
        for claim in allowed:
            safe = clean(claim.get("safe_wording"))
            if safe and safe.lower() in text_l:
                rows.append({
                    "audit_id": f"CA_v2al_{len(rows):04d}",
                    "file": rel,
                    "claim_fragment": safe,
                    "claim_status": "allowed",
                    "allowed_by_v2ak": "true",
                    "allowed_by_v2aj": "true",
                    "requires_disclaimer": "true",
                    "disclaimer_present": "true" if _has_disclaimer(text_l) else "false",
                    "violation": "false",
                    "violation_reason": "",
                })
        for claim in forbidden:
            unsafe = clean(claim.get("unsafe_wording"))
            if unsafe and unsafe.lower() in text_l:
                safe_context = any(m in text_l for m in SAFE_CONTEXT_MARKERS)
                rows.append({
                    "audit_id": f"CA_v2al_{len(rows):04d}",
                    "file": rel,
                    "claim_fragment": unsafe,
                    "claim_status": "forbidden_context_only" if safe_context else "forbidden_positive",
                    "allowed_by_v2ak": "false",
                    "allowed_by_v2aj": "false",
                    "requires_disclaimer": "true",
                    "disclaimer_present": "true" if safe_context else "false",
                    "violation": "false" if safe_context else "true",
                    "violation_reason": "" if safe_context else "Forbidden claim used as positive assertion.",
                })
    if not rows:
        rows.append({
            "audit_id": "CA_v2al_0000",
            "file": "all_v2al_bundles",
            "claim_fragment": "review-only manuscript language",
            "claim_status": "allowed",
            "allowed_by_v2ak": "true",
            "allowed_by_v2aj": "true" if v2aj_claims_available() else "false",
            "requires_disclaimer": "true",
            "disclaimer_present": "true",
            "violation": "false",
            "violation_reason": "",
        })
    assert_no_operational_claim(rows)
    violations = [r for r in rows if r.get("violation") == "true"]
    write_csv(dataset_path("v2al_claim_alignment_audit.csv"),
              CLAIM_ALIGNMENT_COLUMNS, rows)
    if violations:
        raise ValueError(
            f"Unauthorized operational claims in bundles: "
            f"{[(v['file'], v['claim_fragment']) for v in violations[:5]]}"
        )
    return rows


def _has_disclaimer(text_l):
    return any(m in text_l for m in ("review-only", "operacional", "bloque", "manual review"))


def run_table_caption_export_builder(args=None):
    assert_v2ak_ready()
    tables = load_v2aj_tables()
    rows = []
    governance_note = (" Dados de governanca e revisao, sem acuracia e sem "
                       "validacao para uso operacional.")
    for table in tables:
        safe_caption = clean(table.get("safe_caption"))
        if governance_note.strip() not in safe_caption:
            safe_caption = (safe_caption + governance_note).strip()
        rows.append({
            "caption_id": f"CAP_v2al_{len(rows):03d}",
            "source_table_id": clean(table.get("table_id")),
            "target_section": clean(table.get("tcc_section")) or "results_or_appendix",
            "safe_title": clean(table.get("suggested_title")),
            "safe_caption": safe_caption,
            "forbidden_caption": clean(table.get("unsafe_caption")),
            "allowed_interpretation": clean(table.get("allowed_interpretation")),
            "manual_review_required": "true",
        })
    if not rows:
        rows.append({
            "caption_id": "CAP_v2al_000",
            "source_table_id": "no_v2aj_tables",
            "target_section": "results_or_appendix",
            "safe_title": "Estado review-only do Protocolo C",
            "safe_caption": ("Tabela de governanca e revisao do Protocolo C." + governance_note),
            "forbidden_caption": "tabela de validacao operacional",
            "allowed_interpretation": "Descrever estado de revisao, nao desempenho.",
            "manual_review_required": "true",
        })
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2al_table_caption_export.csv"), CAPTION_COLUMNS, rows)
    lines = [
        "<!-- v2al safe table captions -- manual review required -->",
        "",
        "# Protocolo C v2al safe table captions",
        "",
        "Legendas seguras para tabelas do TCC. Os dados sao de governanca e revisao,",
        "sem acuracia e sem validacao para uso operacional. Revisao manual obrigatoria.",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['safe_title']} ({row['source_table_id']})")
        lines.append(f"- Secao alvo: {row['target_section']}")
        lines.append(f"- Legenda segura: {row['safe_caption']}")
        lines.append(f"- Nao usar (legenda proibida): {row['forbidden_caption']}")
        lines.append(f"- Interpretacao permitida: {row['allowed_interpretation']}")
        lines.append("")
    write_markdown(integration_path("v2al_safe_table_captions.md"), lines)
    return rows


def _manuscript_candidates_for(section_col):
    rows = load_csv(dataset_path("v2al_manuscript_candidate_registry.csv"))
    matches = [r for r in rows if r.get(section_col) == "true"]
    return matches


def run_manuscript_patch_plan_builder(args=None):
    assert_v2ak_ready()
    section_to_col = {
        "metodologia": "contains_metodologia",
        "resultados": "contains_resultados",
        "discussao": "contains_discussao",
        "limitacoes_trabalhos_futuros": "contains_limitacoes",
    }
    rows = []
    for section_key, source_doc, out_name in SECTION_BUNDLES:
        col = section_to_col.get(section_key, "")
        matches = _manuscript_candidates_for(col) if col else []
        targets = [m["path"] for m in matches] or ["NO_CANDIDATE_CREATE_NEW_MANUALLY"]
        for target in targets:
            rows.append({
                "patch_id": f"PP_v2al_{len(rows):04d}",
                "target_candidate_file": target,
                "source_section_file": rel_integration(f"{out_name}.md"),
                "anchor_to_find": build_section_anchor(section_key),
                "insert_before_or_after": "after",
                "manual_action": ("Inserir manualmente apos revisao humana; nunca "
                                  "aplicar patch automatico no manuscrito."),
                "risk_level": "medium" if target.startswith("NO_CANDIDATE") else "low",
                "required_human_check": ("Confirmar linguagem review-only, ausencia de "
                                         "claim operacional e disclaimers."),
            })
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2al_manuscript_patch_plan.csv"), PATCH_COLUMNS, rows)
    lines = [
        "<!-- v2al manual patch plan -- no automatic manuscript edit -->",
        "",
        "# Protocolo C v2al manual patch plan",
        "",
        "Plano manual de insercao. Nenhuma escrita automatica e aplicada ao manuscrito.",
        "Cada passo exige revisao humana antes de copiar o texto.",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['patch_id']} -> {row['target_candidate_file']}")
        lines.append(f"- Fonte: {row['source_section_file']}")
        lines.append(f"- Ancora ({row['insert_before_or_after']}): `{row['anchor_to_find']}`")
        lines.append(f"- Acao manual: {row['manual_action']}")
        lines.append(f"- Risco: {row['risk_level']}")
        lines.append(f"- Checagem humana: {row['required_human_check']}")
        lines.append("")
    write_markdown(integration_path("v2al_manual_patch_plan.md"), lines)
    return rows


_PACKET_QUESTIONS = [
    "As secoes do Protocolo C entram como metodologia, resultado ou discussao?",
    "A fila de revisao humana deve entrar no corpo ou apendice?",
    "O termo \"ground truth operacional\" deve ser mantido ou trocado por \"referencia observacional operacional\"?",
    "As tabelas de blockers entram nos resultados ou apendice?",
    "O texto deve enfatizar mais limitacao ou governanca metodologica?",
]


def run_orientador_review_packet_builder(args=None):
    assert_v2ak_ready()
    rows = [
        {"packet_item_id": "OP_v2al_000", "category": "resumo",
         "content": ("Integracao preparou candidatos de secao Markdown/LaTeX a partir "
                     "dos drafts v2ak, sem editar o manuscrito principal."),
         "decision_point": "Aprovar escopo da integracao.", "question": ""},
        {"packet_item_id": "OP_v2al_001", "category": "secoes_candidatas",
         "content": ("Metodologia, resultados, discussao e limitacoes/trabalhos futuros "
                     "estao prontos como candidatos review-only."),
         "decision_point": "Definir alocacao de cada secao no TCC.", "question": ""},
        {"packet_item_id": "OP_v2al_002", "category": "ponto_sensivel_linguagem",
         "content": ("Manter linguagem review-only: nao afirmar deteccao de enchente, "
                     "ground truth validado, classe, label ou validacao operacional."),
         "decision_point": "Confirmar terminologia segura.", "question": ""},
        {"packet_item_id": "OP_v2al_003", "category": "decisao_orientador",
         "content": "Pontos que dependem de decisao do orientador estao nas perguntas.",
         "decision_point": "Responder perguntas objetivas.", "question": ""},
    ]
    for idx, question in enumerate(_PACKET_QUESTIONS):
        rows.append({
            "packet_item_id": f"OP_v2al_Q{idx:02d}",
            "category": "pergunta",
            "content": "",
            "decision_point": "Decisao do orientador necessaria.",
            "question": question,
        })
    assert_no_operational_claim(rows)
    write_csv(dataset_path("v2al_orientador_review_packet.csv"), PACKET_COLUMNS, rows)
    lines = [
        "<!-- v2al advisor review packet -- manual review required -->",
        "",
        "# Protocolo C v2al - pacote para revisao do orientador",
        "",
        "## Resumo do que foi integrado",
        ("Os drafts seguros da v2ak foram limpos e empacotados como candidatos de "
         "secao em Markdown e LaTeX. Nenhuma alteracao foi aplicada ao manuscrito "
         "principal e nenhuma revisao humana foi simulada."),
        "",
        "## Secoes candidatas",
        "- Metodologia / Protocolo C",
        "- Resultados do Protocolo C",
        "- Discussao metodologica",
        "- Limitacoes e trabalhos futuros",
        "- Briefing de orientacao (fora do corpo do TCC)",
        "",
        "## Pontos sensiveis de linguagem",
        ("Evitar qualquer frase que transforme candidato revisavel em referencia "
         "operacional, classe, label, validacao ou deteccao observada."),
        "",
        "## Onde o orientador deve decidir",
        "- Alocacao das secoes (metodologia, resultados, discussao).",
        "- Corpo versus apendice para fila de revisao e tabelas de blockers.",
        "- Terminologia de referencia operacional.",
        "- Enfase entre limitacao e governanca metodologica.",
        "",
        "## Perguntas objetivas",
    ]
    for question in _PACKET_QUESTIONS:
        lines.append(f"- {question}")
    write_markdown(integration_path("v2al_orientador_review_packet.md"), lines)
    return rows


def _regression_artifacts():
    artifacts = []
    if os.path.isdir(DATASET_DIR):
        for n in sorted(os.listdir(DATASET_DIR)):
            if n.startswith("v2al_") and n.endswith(".csv"):
                artifacts.append((rel_dataset(n), dataset_path(n), "csv"))
    if os.path.isdir(INTEGRATION_DIR):
        for n in sorted(os.listdir(INTEGRATION_DIR)):
            if n.endswith((".md", ".tex")):
                artifacts.append((rel_integration(n), integration_path(n), "text"))
    if os.path.isdir(DOCS_DIR):
        for n in sorted(os.listdir(DOCS_DIR)):
            if "v2ak" in n and n.endswith(".md"):
                artifacts.append((rel_doc(n), doc_path(n), "text"))
    return artifacts


def _scan_csv_for_regression(path):
    counts = {
        "forbidden_true_flag": 0, "forbidden_status": 0, "absolute_path": 0,
        "local_only": 0, "forbidden_kv": 0, "unsafe_language": 0,
    }
    for row in load_csv(path):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            value_l = value_s.lower()
            if key_l in FORBIDDEN_TRUE_FIELDS and is_true(value_s):
                counts["forbidden_true_flag"] += 1
            if value_s in FORBIDDEN_STATUS_VALUES:
                counts["forbidden_status"] += 1
            if ABSOLUTE_PATH_RE.search(value_s):
                counts["absolute_path"] += 1
            if LOCAL_ONLY_MARKER in value_l:
                counts["local_only"] += 1
            squashed = re.sub(r"\s*=\s*", "=", value_l)
            for marker in FORBIDDEN_KV_MARKERS:
                if marker in squashed:
                    counts["forbidden_kv"] += 1
            for phrase in UNSAFE_LANGUAGE:
                if phrase in value_l and not _field_allows_unsafe(key, value_s):
                    counts["unsafe_language"] += 1
    return counts


def run_safe_language_regression(args=None):
    assert_v2ak_ready()
    check_types = ["forbidden_true_flag", "forbidden_status", "absolute_path",
                   "non_versionable_path_marker", "forbidden_kv", "unsafe_language"]
    rows = []
    total_fail = 0
    for rel, path, kind in _regression_artifacts():
        counts = (_scan_csv_for_regression(path) if kind == "csv"
                  else scan_text_violations(read_text(path)))
        for check_type in check_types:
            key = "local_only" if check_type == "non_versionable_path_marker" else check_type
            count = counts.get(key, 0)
            status = "PASS" if count == 0 else "FAIL"
            if status == "FAIL":
                total_fail += 1
            rows.append({
                "regression_id": f"SLR_v2al_{len(rows):05d}",
                "artifact_path": rel,
                "check_type": check_type,
                "violation_count": str(count),
                "status": status,
                "severity": "none" if count == 0 else "blocking",
                "notes": "Fail-closed safe-language regression over v2al/v2ak outputs.",
            })
    write_csv(dataset_path("v2al_safe_language_regression.csv"),
              REGRESSION_COLUMNS, rows)
    if total_fail:
        fails = [(r["artifact_path"], r["check_type"]) for r in rows if r["status"] == "FAIL"]
        raise ValueError(f"Safe-language regression failed: {fails[:5]}")
    return rows


def run_next_action_ranker(args=None):
    assert_v2ak_ready()
    options = [
        ("MANUAL_TCC_INTEGRATION_REVIEW", 100,
         "v2al section bundles and insertion matrix",
         "docs/tcc_exports/v2al_manuscript_integration/"),
        ("ORIENTATION_MEETING_REVIEW", 90,
         "v2al advisor review packet",
         "docs/tcc_exports/v2al_manuscript_integration/v2al_orientador_review_packet.md"),
        ("COPY_SECTIONS_TO_MANUSCRIPT_AFTER_HUMAN_REVIEW", 80,
         "approved v2al section candidates",
         "v2al_manuscript_patch_plan.csv"),
        ("APPENDIX_EXPORT", 70,
         "v2al captions and bundles",
         "v2al_table_caption_export.csv"),
        ("HUMAN_REVIEW_EXECUTION", 60,
         "v2ai assignments and v2aj guide",
         "v2ai_review_assignment_registry.csv"),
        ("TRAINING_PROTOCOL_B_OVERLAY_LABEL_GT_PROMOTION", 0,
         "blocked by guardrails", "none"),
    ]
    rows = []
    for rank, (action, score, required, artifact) in enumerate(
            sorted(options, key=lambda x: (-x[1], x[0])), 1):
        rows.append({
            "rank": str(rank),
            "next_action": action,
            "score": str(score),
            "allowed": "false" if score == 0 else "true",
            "blocked_operational_use": "true",
            "required_input": required,
            "recommended_script_or_artifact": artifact,
            "notes": ("No next action may recommend training, Protocol B, overlay, "
                      "labels, ground truth, automatic date inference, or operational "
                      "promotion."),
        })
    write_csv(dataset_path("v2al_next_actions_registry.csv"), NEXT_COLUMNS, rows)
    return rows


def run_completion_report(args=None):
    assert_v2ak_ready()
    candidates = load_csv(dataset_path("v2al_manuscript_candidate_registry.csv"))
    md_bundles = [rel_integration(f"{n}.md") for _, _, n in SECTION_BUNDLES
                  if os.path.exists(integration_path(f"{n}.md"))]
    tex_bundles = [rel_integration(f"{n}.tex") for _, _, n in SECTION_BUNDLES
                   if os.path.exists(integration_path(f"{n}.tex"))]
    captions = load_csv(dataset_path("v2al_table_caption_export.csv"))
    patch_plan = load_csv(dataset_path("v2al_manuscript_patch_plan.csv"))
    claim_audit = load_csv(dataset_path("v2al_claim_alignment_audit.csv"))
    regression = load_csv(dataset_path("v2al_safe_language_regression.csv"))
    next_rows = load_csv(dataset_path("v2al_next_actions_registry.csv"))
    claim_violations = sum(1 for r in claim_audit if r.get("violation") == "true")
    regression_fail = sum(1 for r in regression if r.get("status") == "FAIL")
    found_candidates = sum(1 for r in candidates
                           if r.get("likely_role") not in ("none_found",))
    rows = [
        {"completion_id": "CR_v2al_000", "metric": "inputs_read",
         "value": str(len(V2AK_DRAFT_DOCS) + len(V2AK_DATASETS)),
         "status": "RECORDED",
         "notes": "|".join([rel_doc(n) for n in V2AK_DRAFT_DOCS.values()]
                           + [rel_dataset(n) for n in V2AK_DATASETS.values()])},
        {"completion_id": "CR_v2al_001", "metric": "manuscript_candidates_found",
         "value": str(found_candidates), "status": "RECORDED",
         "notes": "Scan-only; safe_to_autowrite is always false."},
        {"completion_id": "CR_v2al_002", "metric": "markdown_bundles_created",
         "value": str(len(md_bundles)), "status": "RECORDED",
         "notes": "|".join(md_bundles)},
        {"completion_id": "CR_v2al_003", "metric": "latex_bundles_created",
         "value": str(len(tex_bundles)), "status": "RECORDED",
         "notes": "|".join(tex_bundles)},
        {"completion_id": "CR_v2al_004", "metric": "captions_exported",
         "value": str(len(captions)), "status": "RECORDED",
         "notes": "Governance/review captions only."},
        {"completion_id": "CR_v2al_005", "metric": "patch_plan_steps",
         "value": str(len(patch_plan)), "status": "RECORDED",
         "notes": "Manual patch plan; no automatic manuscript edit."},
        {"completion_id": "CR_v2al_006", "metric": "claims_audited",
         "value": str(len(claim_audit)), "status": "RECORDED",
         "notes": "Claim alignment audit over v2al bundles."},
        {"completion_id": "CR_v2al_007", "metric": "claim_violations",
         "value": str(claim_violations),
         "status": "PASS" if claim_violations == 0 else "FAIL",
         "notes": "Unauthorized operational claims must remain zero."},
        {"completion_id": "CR_v2al_008", "metric": "safe_language_regression_failures",
         "value": str(regression_fail),
         "status": "PASS" if regression_fail == 0 else "FAIL",
         "notes": "Fail-closed safe-language regression."},
        {"completion_id": "CR_v2al_009", "metric": "main_manuscript_overwritten",
         "value": "false", "status": "GUARDRAIL_OK",
         "notes": "No automatic manuscript overwrite; outputs are v2al_ only."},
        {"completion_id": "CR_v2al_010", "metric": "next_action_rank_1",
         "value": next_rows[0]["next_action"] if next_rows else "",
         "status": "SAFE_NEXT_ACTION", "notes": "Manual integration review only."},
        {"completion_id": "CR_v2al_011", "metric": "final_decision",
         "value": "safe_manuscript_integration_package_ready_no_operational_promotion",
         "status": "NO_OPERATIONAL_PROMOTION",
         "notes": "No science altered; v2ak/v2aj inputs read-only."},
    ]
    write_csv(dataset_path("v2al_completion_report.csv"), COMPLETION_COLUMNS, rows)
    lines = [
        "<!-- v2al completion report -->",
        "",
        "# Protocolo C v2al completion report",
        "",
        f"Inputs read: {len(V2AK_DRAFT_DOCS) + len(V2AK_DATASETS)}.",
        f"Manuscript candidates found: {found_candidates}.",
        f"Markdown bundles created: {len(md_bundles)}.",
        f"LaTeX bundles created: {len(tex_bundles)}.",
        f"Captions exported: {len(captions)}.",
        f"Manual patch plan steps: {len(patch_plan)}.",
        f"Claims audited: {len(claim_audit)}.",
        f"Claim violations: {claim_violations}.",
        f"Safe-language regression failures: {regression_fail}.",
        "Main manuscript overwritten: false.",
        f"Next action rank 1: {next_rows[0]['next_action'] if next_rows else ''}.",
        ("Final decision: safe manuscript integration package ready with no "
         "operational promotion and no automatic manuscript edit."),
    ]
    write_markdown(integration_path("v2al_completion_report.md"), lines)
    return rows


def run_all(args=None):
    run_manuscript_candidate_scanner(args)
    run_section_insertion_matrix_builder(args)
    run_markdown_section_bundle_builder(args)
    run_latex_section_bundle_builder(args)
    run_claim_alignment_audit_builder(args)
    run_table_caption_export_builder(args)
    run_manuscript_patch_plan_builder(args)
    run_orientador_review_packet_builder(args)
    run_safe_language_regression(args)
    run_next_action_ranker(args)
    return run_completion_report(args)

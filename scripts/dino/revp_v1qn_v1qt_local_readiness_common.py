"""Shared helpers for REV-P DINO local readiness block v1qn-v1qt.

Transforms the review-only DINO queue (v1qa/v1qh) into a locally executable
smoke package. Never reads pixels by default. Never downloads. Never trains.
No vector becomes a label, target, or ground truth.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

# Re-export shared IO from the representation common so callers only import here.
from revp_v1pg_v1pm_dino_representation_common import (
    DATASETS, DOCS, ROOT, SCHEMAS,
    _p, assert_no_forbidden_true, normalize_region, path_hash,
    require_no_abs_paths, sanitized_rel_path, sha256_short,
    write_csv, write_doc, write_schema, read_csv, read_csv_header, read_json_safe,
)

__all__ = [
    "DATASETS", "DOCS", "ROOT", "SCHEMAS",
    "_p", "assert_no_forbidden_true", "normalize_region", "path_hash",
    "require_no_abs_paths", "sanitized_rel_path", "sha256_short",
    "write_csv", "write_doc", "write_schema", "read_csv", "read_csv_header",
    "read_json_safe", "write_json",
    "READINESS_FORBIDDEN_FIELDS", "IMAGE_EXTS", "READINESS_PHRASE",
    "env_str", "env_bool", "env_int", "local_roots", "model_env",
    "mask_abs", "file_sha256_short", "detect_abs",
    "read_smoke_sample", "read_v1qa_queue", "read_v1fu_manifest",
    "read_v1fm_designation",
    "normalize_patch", "candidate_roots",
    "resolve_candidate", "ranked_candidates",
    "guardrail_row_ok",
]

READINESS_PHRASE = (
    "A etapa v1qn–v1qt não altera o estatuto científico dos patches. Ela apenas "
    "transforma a fila review-only em um pacote localmente executável, indicando "
    "quais arquivos e configurações ainda faltam para produzir embeddings 768D sem "
    "criar rótulos, targets ou ground truth."
)

READINESS_FORBIDDEN_FIELDS = (
    "can_create_label", "can_train_model", "target_created", "ground_truth_created",
    "cluster_is_label", "similarity_validates_event", "pca_validates_event",
)

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp"}

ABS_PATH_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def local_roots() -> dict[str, Path]:
    """Return {env_name: path} for configured local root env vars (existing only)."""
    names = ("REVP_SENTINEL_LOCAL_ROOT", "REVP_DINO_VISUAL_ROOT",
             "REVP_DINO_ASSET_ROOT", "REVP_DINO_SOURCE_ROOT")
    out: dict[str, Path] = {}
    for n in names:
        v = env_str(n)
        if v:
            p = Path(v)
            if p.exists():
                out[n] = p
    return out


def model_env() -> dict[str, str]:
    return {
        "model_path": env_str("REVP_DINO_MODEL_PATH"),
        "allow_download": env_str("REVP_DINO_ALLOW_DOWNLOAD", "false"),
        "dry_run": env_str("REVP_DINO_DRY_RUN", "true"),
        "pixel_read_allowed": env_str("REVP_DINO_PIXEL_READ_ALLOWED", "false"),
        "hf_hub_offline": env_str("HF_HUB_OFFLINE", ""),
    }


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")


def mask_abs(val: str) -> str:
    """Replace absolute path with a masked token (never written to versionable CSV)."""
    if ABS_PATH_RE.search(val):
        return f"masked_abs:{path_hash(val)}"
    return val


def detect_abs(val: str) -> bool:
    return bool(ABS_PATH_RE.search(val))


def file_sha256_short(path: Path, n: int = 16) -> str:
    try:
        if not path.exists() or not path.is_file():
            return ""
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:n]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Manifest / queue readers
# ---------------------------------------------------------------------------

_V1QH = _p("REVP_V1QN_IN_SMOKE",
            DATASETS / "dino_smoke_sample_selection_v1qh.csv")
_V1QA = _p("REVP_V1QN_IN_QUEUE",
            DATASETS / "dino_execution_queue_from_visual_expansion_v1qa.csv")
_V1FU = _p("REVP_V1QN_IN_V1FU",
            ROOT / "manifests" / "dino_inputs" /
            "revp_v1fu_dino_sentinel_input_manifest" /
            "dino_sentinel_input_status_v1fu.csv")
_V1FM = _p("REVP_V1QN_IN_V1FM",
            ROOT / "manifests" / "patch_grounding" /
            "revp_v1fm_explicit_patch_tif_designation" /
            "patch_designation_table_v1fm.csv")


def read_smoke_sample(path: Path | None = None) -> list[dict[str, str]]:
    return read_csv(path or _V1QH)


def read_v1qa_queue(path: Path | None = None) -> list[dict[str, str]]:
    return read_csv(path or _V1QA)


def read_v1fu_manifest(path: Path | None = None) -> list[dict[str, str]]:
    return read_csv(path or _V1FU)


def read_v1fm_designation(path: Path | None = None) -> list[dict[str, str]]:
    return read_csv(path or _V1FM)


# ---------------------------------------------------------------------------
# Identity normalization
# ---------------------------------------------------------------------------

def normalize_patch(row: dict[str, str]) -> tuple[str, str, str]:
    pid = (row.get("patch_id", "") or row.get("canonical_patch_id", "") or "UNKNOWN").strip().upper()
    alias = (row.get("alias", "") or pid).strip()
    region = normalize_region(row.get("region", ""))
    return pid, alias, region


# ---------------------------------------------------------------------------
# Local asset resolution
# ---------------------------------------------------------------------------

def candidate_roots(extra: dict[str, Path] | None = None) -> list[tuple[str, Path]]:
    """Return [(env_name, root_path)] from env + extra, ordered by priority."""
    order = ("REVP_SENTINEL_LOCAL_ROOT", "REVP_DINO_VISUAL_ROOT",
             "REVP_DINO_ASSET_ROOT", "REVP_DINO_SOURCE_ROOT")
    roots: list[tuple[str, Path]] = []
    for n in order:
        v = env_str(n)
        if v:
            p = Path(v)
            if p.exists():
                roots.append((n, p))
    if extra:
        for n, p in extra.items():
            if (n, p) not in roots:
                roots.append((n, p))
    return roots


def _iter_candidates(root: Path, exts: set[str] | None = None) -> list[Path]:
    """List image-like files under root (non-recursive first, then recursive)."""
    target_exts = exts or IMAGE_EXTS
    files: list[Path] = []
    try:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in target_exts:
                files.append(p)
    except Exception:
        pass
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in target_exts and p not in files:
                files.append(p)
    except Exception:
        pass
    return files


def _match_score(candidate: Path, rel: str, filename: str, patch_id: str,
                 alias: str) -> tuple[str, float]:
    """Score a candidate file against target identifiers.

    Returns (match_type, confidence) where higher confidence is better.
    """
    name = candidate.name
    stem = candidate.stem.lower()
    rel_norm = rel.replace("\\", "/").lower() if rel else ""
    pid_low = patch_id.lower()
    alias_low = alias.lower()
    fn_low = filename.lower() if filename else ""

    # Exact relative path
    if rel_norm:
        try:
            cand_rel = candidate.as_posix().lower()
            if cand_rel.endswith(rel_norm) or cand_rel == rel_norm:
                return ("exact_relative", 1.0)
        except Exception:
            pass
    # Exact filename
    if fn_low and name.lower() == fn_low:
        return ("filename_exact", 0.95)
    # patch_id in stem
    if pid_low and pid_low in stem:
        return ("patch_id_match", 0.80)
    # alias in stem
    if alias_low and alias_low != pid_low and alias_low in stem:
        return ("alias_match", 0.70)
    # filename substring
    if fn_low and fn_low in name.lower():
        return ("filename_substr", 0.60)
    return ("no_match", 0.0)


def resolve_candidate(rel: str, filename: str, patch_id: str, alias: str,
                      roots: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    """Return ranked candidate dicts for a single smoke item."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for env_name, root in roots:
        for cand in _iter_candidates(root):
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            mt, conf = _match_score(cand, rel, filename, patch_id, alias)
            if conf <= 0.0:
                continue
            try:
                size = cand.stat().st_size
            except Exception:
                size = 0
            # Derive a versionable relative candidate path (relative to root)
            try:
                rel_cand = cand.relative_to(root).as_posix()
            except Exception:
                rel_cand = cand.name
            candidates.append({
                "env_name": env_name,
                "filename": cand.name,
                "relative_candidate": rel_cand,
                "local_path_hash": path_hash(str(cand)),
                "match_type": mt,
                "match_confidence": conf,
                "file_exists": True,
                "file_size_bytes": size,
                "file_sha256_short": "",  # computed on demand
            })
    candidates.sort(key=lambda c: -c["match_confidence"])
    return candidates


def ranked_candidates(smoke_rows: list[dict[str, str]],
                      roots: list[tuple[str, Path]]) -> dict[str, list[dict[str, Any]]]:
    """Map smoke_id → ranked candidates for each row."""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in smoke_rows:
        pid, alias, _ = normalize_patch(r)
        rel = r.get("relative_path", "") or ""
        fn = Path(rel).name if rel else ""
        smoke_id = r.get("smoke_id", "")
        out[smoke_id] = resolve_candidate(rel, fn, pid, alias, roots)
    return out


# ---------------------------------------------------------------------------
# Row-level guardrail
# ---------------------------------------------------------------------------

def guardrail_row_ok(row: dict[str, Any]) -> tuple[bool, str]:
    for f in READINESS_FORBIDDEN_FIELDS:
        if str(row.get(f, "false")).strip().lower() == "true":
            return False, f"guardrail_{f}_true"
    return True, ""

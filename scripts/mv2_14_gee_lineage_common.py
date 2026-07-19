"""Núcleo compartilhado do marco MV2-14 — GEE Scene Lineage Recovery & Temporal Metadata Binding.

Recupera e reconcilia lineage temporal/cena (scene_id, acquisition_datetime, tile,
cloud_cover, proveniência de export GEE) a partir de fontes locais leves, para os 128
bindings MEDIUM do MV2-13. NÃO baixa raster, NÃO executa STAC, NÃO cria crop, NÃO
calcula feature espectral, NÃO cria label/silver/negativo.

Fail-closed:
- ausência de evidência nunca vira negativo;
- cidade/região/tile sozinhos nunca estabelecem vínculo (no máximo CANDIDATE_REVIEW);
- scene_id nunca é inferido de filename fraco;
- datetime futuro é sempre inválido;
- PNG/NPZ/render/DINOv2 nunca é raster Sentinel nativo;
- nenhuma cena é auto-vinculada a patch por tile/zona.

Reusa o resolvedor de worktree e utilitários de IO do MV2-13 (uma só fonte de verdade).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from mv2_13_sentinel_binding_common import (  # noqa: F401  (reuso intencional)
    PROJECT_ROOT,
    S2_SCENE_RE,
    _branch_of,
    _worktree_roots,
    classify_datetime,
    clean,
    crs_status,
    ensure_output_dir as _ensure_mv13_dir,
    read_csv_rows,
    write_csv,
    write_json,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_gee_lineage_recovery"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

MV2_13_BINDING_MATRIX = PROJECT_ROOT / "outputs_public" / "mv2_sentinel_binding" / "mv2_13_binding_matrix.csv"

# Extensões leves permitidas na varredura de fontes (nada de raster/binário pesado).
ALLOWED_SCAN_EXT = {".py", ".js", ".ipynb", ".json", ".csv", ".md", ".txt", ".yml", ".yaml"}

# Raízes de varredura na working tree LOCAL (varredura completa).
SCAN_ROOTS_REL = [
    "local_runs/protocolo_c",
    "scripts/protocolo_c",
    "configs/protocolo_c",
    "configs",
    "datasets",
    "manifests/dino_inputs",
    "outputs_public/mv2_sentinel_binding",
    "outputs_public/mv2_data_readiness",
]

# Diretórios pequenos de OUTROS worktrees lidos read-only (não varrer o worktree todo).
SCAN_ROOTS_OTHER_WORKTREE_REL = [
    "outputs_public/mv2_spectral_reconstruction",
    "outputs_public/mv2_regional_rebalancing",
    "outputs_public/mv2_lineage",
]

EXCLUDE_DIR_PARTS = {
    ".git",
    ".claude",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

# Padrões de termos para pontuar relevância.
GEE_TERMS = re.compile(r"\b(ee\.|earthengine|earth engine|COPERNICUS|Export\.image|ImageCollection|filterDate|system:index|CLOUDY_PIXEL_PERCENTAGE)\b", re.IGNORECASE)
SENTINEL_TERMS = re.compile(r"\b(sentinel|S2_SR|S2_MSI|S1_GRD|MSI|SAR)\b", re.IGNORECASE)
EXPORT_TERMS = re.compile(r"\b(export|gee_export|export_task|manual_export|download)\b", re.IGNORECASE)
ASSET_TERMS = re.compile(r"\basset_id\b", re.IGNORECASE)
PATCH_TERMS = re.compile(r"\bpatch_id\b|\b(CUR|PET|REC)_\d{3,5}\b")
COLLECTION_RE = re.compile(r"COPERNICUS/S2[A-Z_]*")
TILE_RE = re.compile(r"\bT\d{2}[A-Z]{3}\b")
CLOUD_RE = re.compile(r"cloud[_a-z]*\s*[:=]?\s*([0-9]+\.[0-9]+)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Taxonomias obrigatórias
# ---------------------------------------------------------------------------

LINEAGE_STATUSES = {
    "LINEAGE_RECOVERED_STRONG",
    "LINEAGE_RECOVERED_PARTIAL",
    "LINEAGE_CANDIDATE_REVIEW",
    "LINEAGE_CONFLICT",
    "LINEAGE_INVALID",
    "LINEAGE_NOT_FOUND",
}

STAC_READINESS = {
    "READY_FOR_STAC_DRY_RUN",
    "READY_FOR_STAC_REAL_REVIEW_REQUIRED",
    "BLOCKED_NO_SCENE_ID",
    "BLOCKED_NO_DATETIME",
    "BLOCKED_NO_TILE",
    "BLOCKED_NO_CLOUD_COVER",
    "BLOCKED_NO_EXPORT_PROVENANCE",
    "BLOCKED_CONFLICT",
    "BLOCKED_INVALID",
    "BLOCKED_NO_LINEAGE",
}

RISK_TYPES = [
    "RISK_SCENE_WITHOUT_PATCH",
    "RISK_PATCH_WITHOUT_SCENE",
    "RISK_TILE_AUTO_JOIN",
    "RISK_FILENAME_INFERENCE",
    "RISK_FUTURE_DATETIME",
    "RISK_CITY_REGION_PROXY",
    "RISK_CLOUD_COVER_MISSING",
    "RISK_EXPORT_PROVENANCE_MISSING",
    "RISK_STAC_BEFORE_LINEAGE",
    "RISK_DAY10_FALSE_UNLOCK",
    "RISK_RASTER_DOWNLOAD_WITHOUT_BINDING",
    "RISK_PNG_AS_RASTER",
    "RISK_UNKNOWN_AS_NEGATIVE",
]

# Mapa zona UTM -> tile MGRS por região (apenas para HIPÓTESE de revisão, nunca join).
REGION_ZONE = {
    "Curitiba": ("EPSG:32722", "T22"),
    "Petropolis": ("EPSG:32723", "T23"),
    "Recife": ("EPSG:32725", "T25"),
}

_WORKTREE_ROOTS_CACHE: list[Path] | None = None
_BRANCH_CACHE: dict[Path, str] = {}


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def worktree_roots_cached() -> list[Path]:
    """Lista worktrees uma vez; evita subprocesso por arquivo varrido."""
    global _WORKTREE_ROOTS_CACHE
    if _WORKTREE_ROOTS_CACHE is None:
        _WORKTREE_ROOTS_CACHE = _worktree_roots()
    return _WORKTREE_ROOTS_CACHE


def branch_of_cached(root: Path) -> str:
    if root not in _BRANCH_CACHE:
        _BRANCH_CACHE[root] = _branch_of(root)
    return _BRANCH_CACHE[root]


def resolve_scan_roots() -> list[tuple[Path, str]]:
    """Retorna (root_abs, origin) das raízes de varredura existentes.

    A working tree LOCAL é varrida nas raízes amplas; outros worktrees são lidos
    read-only apenas em diretórios MV2 pequenos e específicos (evita varrer worktrees
    inteiros, que seria lento e desnecessário).
    """
    out: list[tuple[Path, str]] = []
    seen: set[str] = set()
    # Local
    for rel in SCAN_ROOTS_REL:
        cand = PROJECT_ROOT / rel
        if cand.exists() and cand.as_posix() not in seen:
            seen.add(cand.as_posix())
            out.append((cand, "LOCAL"))
    # Outros worktrees: só dirs MV2 específicos e ainda não presentes localmente
    for wt in worktree_roots_cached():
        if wt == PROJECT_ROOT:
            continue
        origin = f"READ_ONLY_FROM_OTHER_WORKTREE:{branch_of_cached(wt)}"
        for rel in SCAN_ROOTS_OTHER_WORKTREE_REL:
            if (PROJECT_ROOT / rel).exists():
                continue  # já existe local; não duplicar de outro worktree
            cand = wt / rel
            if cand.exists() and cand.as_posix() not in seen:
                seen.add(cand.as_posix())
                out.append((cand, origin))
    return out


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIR_PARTS for part in path.parts)


def iter_scannable_files(root: Path):
    """Itera arquivos com poda antecipada de diretórios pesados/irrelevantes."""
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIR_PARTS)
        for name in sorted(files):
            path = Path(current) / name
            if not is_excluded(path):
                yield path


def rel_to_root(path: Path) -> str:
    """Caminho relativo ao worktree de origem — nunca caminho absoluto privado."""
    try:
        for wt in worktree_roots_cached():
            if str(path).startswith(str(wt)):
                return path.relative_to(wt).as_posix()
    except ValueError:
        pass
    return path.name

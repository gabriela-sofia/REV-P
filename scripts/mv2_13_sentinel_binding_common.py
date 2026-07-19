"""Núcleo compartilhado do marco MV2-13 — Sentinel Binding & Spectral Eligibility Gate.

Reconcilia âncoras Sentinel (MV2-12 Spectral Reconstruction), inventário de assets
(lineage manifest) e inventário de patches, resolvendo binding, geometria/AOI/CRS,
temporalidade e prontidão STAC — SEM baixar raster, SEM executar STAC e SEM gerar crop.

Toda a lógica de classificação vive aqui para que scripts e testes compartilhem uma
única fonte de verdade. Princípios fail-closed:
- ausência de evidência nunca vira negativo nem classe 0;
- cidade/região nunca é proxy de label nem de binding forte;
- filename/texto solto nunca promove binding fraco a forte;
- PNG/JPG/NPZ/render/DINOv2 nunca é raster Sentinel espectral nativo;
- nenhum binding fraco/none/conflito/inválido autoriza STAC real;
- Dia 10 nunca é desbloqueado sem raster Sentinel nativo validado;
- nada pesado é escrito em outputs_public.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_sentinel_binding"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

TODAY = date(2026, 6, 23)

# Caminhos relativos das entradas (resolvidos local OU em outro worktree, sem checkout).
INPUT_RELPATHS = {
    "anchors": "outputs_public/mv2_spectral_reconstruction/sentinel_spectral_anchors.csv",
    "spectral_summary": "outputs_public/mv2_spectral_reconstruction/mv2_12_sentinel_spectral_reconstruction_summary.json",
    "stac_resolution_plan": "outputs_public/mv2_spectral_reconstruction/stac_resolution_plan.csv",
    "lineage_manifest": "outputs_public/mv2_lineage/asset_scene_lineage_manifest.csv",
    "raster_metadata_index": "outputs_public/mv2_lineage/raster_asset_metadata_index.csv",
    "patch_corpus_registry": "datasets/patch_corpus_registry.csv",
    "data_readiness_matrix": "outputs_public/mv2_data_readiness/mv2_12_missing_data_matrix.csv",
    "raster_native_backlog": "outputs_public/mv2_data_readiness/mv2_12_sentinel_native_raster_backlog.csv",
    "regional_coverage": "outputs_public/mv2_regional_rebalancing/regional_representation_coverage.csv",
    "gap_matrix": "outputs_public/audits/mv2_10_gap_audit/mv2_10_gap_matrix.csv",
}

INPUT_STAGE = {
    "anchors": "MV2-12_SPECTRAL_RECONSTRUCTION",
    "spectral_summary": "MV2-12_SPECTRAL_RECONSTRUCTION",
    "stac_resolution_plan": "MV2-12_SPECTRAL_RECONSTRUCTION",
    "lineage_manifest": "MV2-03_LINEAGE",
    "raster_metadata_index": "MV2-03_LINEAGE",
    "patch_corpus_registry": "CORPUS_REGISTRY",
    "data_readiness_matrix": "MV2-12_DATA_READINESS",
    "raster_native_backlog": "MV2-12_DATA_READINESS",
    "regional_coverage": "MV2-11_REGIONAL_REBALANCING",
    "gap_matrix": "MV2-10_GAP_AUDIT",
}

# ---------------------------------------------------------------------------
# Taxonomias obrigatórias
# ---------------------------------------------------------------------------

BINDING_STRENGTHS = {
    "BINDING_STRONG",
    "BINDING_MEDIUM",
    "BINDING_WEAK",
    "BINDING_NONE",
    "BINDING_CONFLICT",
    "BINDING_INVALID",
}

STAC_STATUSES = {
    "STAC_DRY_RUN_ELIGIBLE",
    "STAC_REAL_FUTURE_ELIGIBLE",
    "STAC_BLOCKED_NO_PATCH",
    "STAC_BLOCKED_NO_ASSET",
    "STAC_BLOCKED_NO_SCENE",
    "STAC_BLOCKED_NO_DATETIME",
    "STAC_BLOCKED_NO_GEOMETRY",
    "STAC_BLOCKED_NO_CRS",
    "STAC_BLOCKED_NO_SENSOR",
    "STAC_BLOCKED_WEAK_BINDING",
    "STAC_BLOCKED_CONFLICT",
    "STAC_BLOCKED_INVALID",
    "STAC_NOT_APPLICABLE",
}

DAY10_STATUSES = {
    "DAY10_BLOCKED_NO_NATIVE_RASTER",
    "DAY10_BLOCKED_NO_BINDING",
    "DAY10_BLOCKED_NO_GEOMETRY",
    "DAY10_BLOCKED_NO_CRS",
    "DAY10_BLOCKED_NO_TEMPORAL_METADATA",
    "DAY10_BLOCKED_NO_SCENE_ID",
    "DAY10_READY_FOR_STAC_DRY_RUN_ONLY",
    "DAY10_READY_FOR_STAC_REAL_NEXT_STEP",
    "DAY10_READY_FOR_PRIVATE_CROP_NEXT_STEP",
    "DAY10_READY_FOR_SPECTRAL_FEATURES_NEXT_STEP",
    "DAY10_NOT_READY",
}

RISK_TYPES = [
    "RISK_WEAK_FILENAME_INFERENCE",
    "RISK_CITY_PROXY_BINDING",
    "RISK_REGION_PROXY_LABEL",
    "RISK_PNG_AS_RASTER",
    "RISK_NPZ_AS_SPECTRAL",
    "RISK_MISSING_CRS",
    "RISK_MISSING_GEOMETRY",
    "RISK_MISSING_SCENE_ID",
    "RISK_MISSING_DATETIME",
    "RISK_STAC_WITHOUT_PATCH",
    "RISK_STAC_WITHOUT_ASSET",
    "RISK_DAY10_FALSE_UNLOCK",
    "RISK_TRAINING_BEFORE_SILVER",
    "RISK_UNKNOWN_AS_NEGATIVE",
]

# Padrão de granule Sentinel-2 (ex.: 20220118T130239_20220118T130322_T23KPR).
S2_SCENE_RE = re.compile(r"\d{8}T\d{6}_\d{8}T\d{6}_T\d{2}[A-Z]{3}")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ---------------------------------------------------------------------------
# Resolução dinâmica de entradas (local + worktrees), sem hardcode de path privado
# ---------------------------------------------------------------------------


def _worktree_roots() -> list[Path]:
    roots: list[Path] = [PROJECT_ROOT]
    try:
        out = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return roots
    for line in out.splitlines():
        if line.startswith("worktree "):
            p = Path(line[len("worktree ") :].strip())
            if p not in roots:
                roots.append(p)
    return roots


def _branch_of(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return out or "DESCONHECIDO"
    except OSError:
        return "DESCONHECIDO"


def resolve_input(key: str) -> tuple[Path | None, str]:
    """Resolve uma entrada por chave. Retorna (path_absoluto_ou_None, origem_tag).

    origem_tag = "LOCAL" se estiver na working tree atual, ou
    "READ_ONLY_FROM_OTHER_BRANCH:<branch>" se vier de outro worktree.
    """
    rel = INPUT_RELPATHS[key]
    local = PROJECT_ROOT / rel
    if local.exists():
        return local, "LOCAL"
    for root in _worktree_roots():
        if root == PROJECT_ROOT:
            continue
        cand = root / rel
        if cand.exists():
            return cand, f"READ_ONLY_FROM_OTHER_BRANCH:{_branch_of(root)}"
    return None, "MISSING"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, columns: list[str], rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def clean(value: object) -> str:
    return str(value or "").strip()


# ---------------------------------------------------------------------------
# Classificadores atômicos
# ---------------------------------------------------------------------------


def classify_datetime(raw: str) -> tuple[str, str]:
    """Retorna (datetime_normalizado, status). Não infere data; só valida o texto."""
    txt = clean(raw)
    if not txt:
        return "", "ABSENT"
    if ISO_DATE_RE.match(txt):
        year = int(txt[:4])
        if year >= 2025:
            # Data futura/processamento — contradiz aquisição Sentinel histórica.
            return txt, "INVALID_FUTURE"
        if 2015 <= year <= 2024:
            return txt, "VALID_SINGLE"
        return txt, "INVALID_RANGE"
    # Faixas como "2022-2023" ou "2021-2022" são grosseiras, não uma aquisição.
    if re.match(r"^\d{4}-\d{4}$", txt) or re.match(r"^\d{4}$", txt):
        return txt, "COARSE_RANGE"
    return txt, "UNPARSEABLE"


def sensor_from_scene(scene_id: str) -> tuple[str, str, str]:
    """(sensor, platform, confidence) a partir do formato do scene_id."""
    if S2_SCENE_RE.search(clean(scene_id)):
        return "S2_MSI", "Sentinel-2", "HIGH"
    return "", "", "NONE"


def has_geometry(geometry_status: str, bbox: str) -> bool:
    # BOUNDS_ONLY conta como geometria (grosseira); ABSENT/vazio não conta.
    gs = clean(geometry_status).upper()
    return bool(clean(bbox)) and gs not in {"", "ABSENT", "NONE"}


def crs_status(crs: str) -> str:
    c = clean(crs).upper()
    if not c:
        return "ABSENT"
    if re.match(r"^EPSG:\d{4,5}$", c):
        return "PRESENT_VALID"
    return "PRESENT_UNVERIFIED"


# ---------------------------------------------------------------------------
# Loaders normalizados
# ---------------------------------------------------------------------------


def load_anchors() -> list[dict[str, str]]:
    path, _ = resolve_input("anchors")
    return read_csv_rows(path)


def normalize_anchors() -> list[dict[str, object]]:
    """Tarefa 2 — normaliza as âncoras Sentinel sem inferir vínculo forte."""
    path, origin = resolve_input("anchors")
    rows = read_csv_rows(path)
    src = INPUT_RELPATHS["anchors"]
    out: list[dict[str, object]] = []
    for r in rows:
        scene = clean(r.get("scene_id"))
        sensor, platform, sensor_conf = sensor_from_scene(scene)
        dt_norm, dt_status = classify_datetime(r.get("datetime", ""))
        confidence = "NONE"
        if scene and sensor_conf == "HIGH":
            confidence = "SCENE_ID_STRONG"
        elif clean(r.get("asset_id")) and clean(r.get("patch_id")):
            confidence = "ASSET_PATCH_LINEAGE"
        elif clean(r.get("regiao")):
            confidence = "REGION_TEXT_ONLY"
        out.append(
            {
                "anchor_id": clean(r.get("anchor_id")),
                "source_file": src,
                "raw_anchor_key": clean(r.get("asset_id")) or clean(r.get("patch_id")) or scene,
                "raw_anchor_text": clean(r.get("observacao"))[:160],
                "normalized_anchor_key": clean(r.get("anchor_id")),
                "extracted_sensor": sensor,
                "extracted_platform": platform,
                "extracted_scene_id": scene,
                "extracted_tile": clean(r.get("tile_id")),
                "extracted_datetime": dt_norm,
                "extracted_region": clean(r.get("regiao")),
                "extracted_asset_id": clean(r.get("asset_id")),
                "extracted_patch_id": clean(r.get("patch_id")),
                "extraction_confidence": confidence,
                "extraction_notes": (
                    f"origin={origin};datetime_status={dt_status};"
                    f"anchor_strength={clean(r.get('anchor_strength'))}"
                ),
            }
        )
    return out


def normalize_assets() -> list[dict[str, object]]:
    """Tarefa 4 — inventário de assets a partir do lineage manifest.

    Separa render/PNG/NPZ/DINOv2 de raster Sentinel espectral nativo: aqui todos os
    assets têm raster_readable=false, logo is_native_sentinel_raster=false.
    """
    path, origin = resolve_input("lineage_manifest")
    rows = read_csv_rows(path)
    src = INPUT_RELPATHS["lineage_manifest"]
    out: list[dict[str, object]] = []
    for r in rows:
        readable = clean(r.get("raster_readable")).lower() == "true"
        sensor_family = clean(r.get("sensor_family"))
        is_native = readable  # nunca de PNG/NPZ/DINOv2
        out.append(
            {
                "asset_id": clean(r.get("asset_id")),
                "patch_id": clean(r.get("patch_id")),
                "region": clean(r.get("regiao")),
                "sensor": sensor_family,
                "platform": clean(r.get("sentinel_platform")),
                "scene_id": clean(r.get("source_scene_id")),
                "acquisition_datetime": clean(r.get("acquisition_datetime_utc")),
                "cloud_cover": clean(r.get("cloud_cover_percent")),
                "tile_id": clean(r.get("tile_id")),
                "asset_kind": "sentinel_lineage_reference",
                "is_visual_render": "false",
                "is_npz_visual": "false",
                "is_dinov2_embedding": "false",
                "is_native_sentinel_raster": str(is_native).lower(),
                "source_path": "asset_path_reference_local_only",
                "source_file": src,
                "metadata_status": clean(r.get("temporal_metadata_status")) or "MISSING",
                "notes": (
                    f"origin={origin};raster_readable={readable};"
                    f"lineage_status={clean(r.get('lineage_status'))}"
                ),
            }
        )
    return out


def normalize_patches() -> list[dict[str, object]]:
    """Tarefa 3 — inventário de patches a partir do lineage manifest."""
    path, origin = resolve_input("lineage_manifest")
    rows = read_csv_rows(path)
    src = INPUT_RELPATHS["lineage_manifest"]
    seen: dict[str, dict[str, object]] = {}
    for r in rows:
        pid = clean(r.get("patch_id"))
        if not pid or pid in seen:
            continue
        bounds = clean(r.get("raster_bounds"))
        crs = clean(r.get("raster_crs"))
        geom_present = bool(bounds)
        seen[pid] = {
            "patch_id": pid,
            "region": clean(r.get("regiao")),
            "city": clean(r.get("cidade")) or clean(r.get("regiao")),
            "canonical_status": "BOUNDS_DERIVED_CANDIDATE",
            "has_geometry": str(geom_present).lower(),
            "geometry_path": src if geom_present else "",
            "geometry_type": "bbox_bounds" if geom_present else "NONE",
            "bbox": bounds,
            "crs": crs,
            "crs_status": crs_status(crs),
            "source_file": src,
            "notes": f"origin={origin};geometry=BOUNDS_ONLY_NOT_DIGITIZED",
        }
    return list(seen.values())


# ---------------------------------------------------------------------------
# Reconciliação anchor → asset → patch (núcleo)
# ---------------------------------------------------------------------------

BINDING_COLUMNS = [
    "binding_id", "anchor_id", "asset_id", "patch_id", "region", "sensor",
    "platform", "scene_id", "acquisition_datetime", "cloud_cover", "tile_id",
    "geometry_status", "geometry_path", "bbox", "crs", "crs_status",
    "binding_strength", "binding_evidence_source", "binding_evidence_level",
    "conflict_status", "stac_status", "day10_status", "can_stac_dry_run",
    "can_stac_real_next_step", "can_crop_private_next_step",
    "can_compute_spectral_features_now", "blocking_reason", "next_action",
]


def _binding_strength(spatial_ok, scene_strong, has_asset, has_patch,
                      has_region, conflict, invalid):
    if invalid:
        return "BINDING_INVALID"
    if conflict:
        return "BINDING_CONFLICT"
    if spatial_ok and scene_strong:
        return "BINDING_STRONG"
    if (has_asset or has_patch) and (spatial_ok or has_asset):
        return "BINDING_MEDIUM"
    if scene_strong or has_region:
        return "BINDING_WEAK"
    return "BINDING_NONE"


def _stac_status(strength, has_patch, has_asset, spatial_ok, crs, sensor, scene,
                 dt_status, conflict, invalid):
    if invalid:
        return "STAC_BLOCKED_INVALID", "evidencia_temporal_invalida_sem_cena_nem_geometria"
    if conflict:
        return "STAC_BLOCKED_CONFLICT", "anchor_referencia_asset_ou_patch_fora_do_inventario"
    if strength == "BINDING_NONE":
        return "STAC_NOT_APPLICABLE", "sem_anchor_util"
    if not has_patch:
        return "STAC_BLOCKED_NO_PATCH", "sem_patch_id_para_AOI_e_overlay"
    if not has_asset:
        return "STAC_BLOCKED_NO_ASSET", "sem_asset_id_oficial"
    if not spatial_ok:
        return "STAC_BLOCKED_NO_GEOMETRY", "sem_geometria_bbox_para_AOI"
    if crs_status(crs) == "ABSENT":
        return "STAC_BLOCKED_NO_CRS", "sem_CRS"
    if not sensor:
        return "STAC_BLOCKED_NO_SENSOR", "sem_sensor_confirmado"
    if not scene:
        return "STAC_BLOCKED_NO_SCENE", "lado_espacial_pronto_mas_sem_scene_id_nem_temporal"
    if dt_status != "VALID_SINGLE":
        return "STAC_BLOCKED_NO_DATETIME", "sem_datetime_de_aquisicao_valido"
    return "STAC_BLOCKED_WEAK_BINDING", "binding_insuficiente"


def _day10_status(strength, spatial_ok, crs, scene, dt_status, native_raster):
    if not native_raster:
        if strength in {"BINDING_NONE", "BINDING_CONFLICT", "BINDING_INVALID"}:
            return "DAY10_BLOCKED_NO_BINDING"
        if not spatial_ok:
            return "DAY10_BLOCKED_NO_GEOMETRY"
        if crs_status(crs) == "ABSENT":
            return "DAY10_BLOCKED_NO_CRS"
        if not scene:
            return "DAY10_BLOCKED_NO_SCENE_ID"
        if dt_status != "VALID_SINGLE":
            return "DAY10_BLOCKED_NO_TEMPORAL_METADATA"
        return "DAY10_BLOCKED_NO_NATIVE_RASTER"
    return "DAY10_NOT_READY"


def _next_action(strength, spatial_ok, scene):
    if strength == "BINDING_MEDIUM" and spatial_ok and not scene:
        return "recuperar_scene_id_e_datetime_para_este_patch_via_historico_GEE"
    if strength == "BINDING_WEAK" and scene:
        return "vincular_esta_cena_a_um_patch_e_asset_oficial_sem_inferencia_de_tile"
    if strength == "BINDING_NONE":
        return "buscar_anchor_oficial_ou_descartar_para_binding"
    return "recuperar_campos_faltantes_antes_de_qualquer_STAC"


def reconcile() -> list[dict[str, object]]:
    """Tarefa 6 — cruza âncoras, assets e patches; classifica binding_strength."""
    anchors = normalize_anchors()
    assets = {a["asset_id"]: a for a in normalize_assets() if a["asset_id"]}
    patches = {p["patch_id"]: p for p in normalize_patches() if p["patch_id"]}

    out: list[dict[str, object]] = []
    for i, a in enumerate(anchors, start=1):
        asset_id = clean(a["extracted_asset_id"])
        patch_id = clean(a["extracted_patch_id"])
        scene = clean(a["extracted_scene_id"])
        region = clean(a["extracted_region"])
        tile = clean(a["extracted_tile"])
        dt_norm, dt_status = classify_datetime(str(a["extracted_datetime"]))

        asset_row = assets.get(asset_id)
        patch_row = patches.get(patch_id)

        bbox = clean(patch_row["bbox"]) if patch_row else ""
        crs = clean(patch_row["crs"]) if patch_row else ""
        geom_status = "BOUNDS_ONLY" if (patch_row and bbox) else "ABSENT"
        geom_path = clean(patch_row["geometry_path"]) if patch_row else ""

        sensor = clean(a["extracted_sensor"])
        platform = clean(a["extracted_platform"])
        if not sensor and asset_row:
            fam = clean(asset_row["sensor"])
            sensor = "S2_PROBABLE" if fam == "OPTICAL" else ("UNKNOWN" if fam else "")
            platform = clean(asset_row["platform"])
        cloud = clean(asset_row["cloud_cover"]) if asset_row else ""

        has_asset = bool(asset_id)
        has_patch = bool(patch_id)
        has_region = bool(region)
        spatial_ok = bool(patch_row and bbox and crs and geom_status != "ABSENT")
        scene_strong = bool(scene) or bool(
            tile and dt_status == "VALID_SINGLE" and sensor
        )

        conflict = bool((asset_id and not asset_row) or (patch_id and not patch_row))
        invalid = dt_status == "INVALID_FUTURE" and not scene and not spatial_ok

        strength = _binding_strength(
            spatial_ok, scene_strong, has_asset, has_patch, has_region, conflict, invalid
        )

        evidence_src = []
        if asset_row:
            evidence_src.append("lineage_manifest")
        if scene:
            evidence_src.append("scene_id_text")
        if region and not asset_row and not scene:
            evidence_src.append("region_text_only")
        evidence_level = {
            "BINDING_STRONG": "EXPLICIT",
            "BINDING_MEDIUM": "MANIFEST_PARTIAL",
            "BINDING_WEAK": "TEXT_OR_SCENE_UNLINKED",
            "BINDING_NONE": "NONE",
            "BINDING_CONFLICT": "CONFLICTING",
            "BINDING_INVALID": "CONTRADICTORY",
        }[strength]

        stac_status, stac_block = _stac_status(
            strength, has_patch, has_asset, spatial_ok, crs, sensor, scene,
            dt_status, conflict, invalid,
        )
        day10 = _day10_status(strength, spatial_ok, crs, scene, dt_status, False)

        out.append(
            {
                "binding_id": f"MV2_13_BIND_{i:05d}",
                "anchor_id": clean(a["anchor_id"]),
                "asset_id": asset_id,
                "patch_id": patch_id,
                "region": region,
                "sensor": sensor,
                "platform": platform,
                "scene_id": scene,
                "acquisition_datetime": dt_norm,
                "cloud_cover": cloud,
                "tile_id": tile,
                "geometry_status": geom_status,
                "geometry_path": geom_path,
                "bbox": bbox,
                "crs": crs,
                "crs_status": crs_status(crs),
                "binding_strength": strength,
                "binding_evidence_source": "|".join(evidence_src) or "none",
                "binding_evidence_level": evidence_level,
                "conflict_status": (
                    "CONFLICT" if conflict else ("INVALID" if invalid else "NONE")
                ),
                "stac_status": stac_status,
                "day10_status": day10,
                # Fail-closed: nada libera real/crop/feature nesta etapa.
                "can_stac_dry_run": "false",
                "can_stac_real_next_step": "false",
                "can_crop_private_next_step": "false",
                "can_compute_spectral_features_now": "false",
                "blocking_reason": stac_block,
                "next_action": _next_action(strength, spatial_ok, scene),
            }
        )
    return out

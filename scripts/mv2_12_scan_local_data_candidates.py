"""MV2-12 — Varredura local leve de candidatos a dado.

Varre a working tree procurando arquivos potencialmente úteis para o backlog de
dados (PNG, JPG, TIFF/TIF, JP2, SAFE, GeoJSON, GPKG, SHP, CSV, NPZ relevantes).

NÃO copia nenhum arquivo pesado: apenas indexa caminho, extensão, tamanho e tenta
inferir região e vínculo patch/asset. Renderização visual (PNG/JPG/NPZ) é marcada
explicitamente como NÃO-espectral. Nenhum candidato é promovido a canônico aqui:
promoção exige vínculo patch_id/asset_id forte e revisão posterior.

Saída: outputs_public/mv2_data_readiness/mv2_12_local_recovery_candidates.csv
"""

from __future__ import annotations

import re
from pathlib import Path

from mv2_12_data_readiness_common import (
    NATIVE_SPECTRAL_EXTENSIONS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    VISUAL_RENDER_EXTENSIONS,
    ensure_output_dir,
    infer_region_from_text,
    write_csv,
)

# Diretórios excluídos da varredura (ruído, históricos, cópias de outras branches).
EXCLUDE_DIR_PARTS = {
    ".git",
    ".claude",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
}

# Onde procurar candidatos visuais regionais e geometrias/tabelas relevantes.
SCAN_ROOTS = [
    PROJECT_ROOT / "local_runs",
    PROJECT_ROOT / "outputs_public" / "figures",
    PROJECT_ROOT / "datasets" / "external_sources",
    PROJECT_ROOT / "docs" / "protocolo_c",
    PROJECT_ROOT / "data_quarantine",
    PROJECT_ROOT / "external_data_local",
    PROJECT_ROOT / "data_raw_local",
]

# Foco do scanner de RECUPERAÇÃO local: render visual (rebalanceamento regional),
# raster espectral nativo (Dia 10), geometria vetorial e .npz visual. CSVs genéricos
# de tabela ficam fora deste índice (entram nas matrizes de dado faltante/download).
RELEVANT_EXTENSIONS = (
    VISUAL_RENDER_EXTENSIONS
    | NATIVE_SPECTRAL_EXTENSIONS
    | {".geojson", ".gpkg", ".shp", ".npz"}
)

PATCH_RE = re.compile(r"(rec|pet|ctb|cur)0*([0-9]{3,5})", re.IGNORECASE)
MAX_FILES = 4000


def _excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIR_PARTS for part in path.parts)


def _infer_patch_id(name: str) -> str:
    match = PATCH_RE.search(name)
    if not match:
        return ""
    prefix = match.group(1).upper()
    prefix = {"CUR": "CTB"}.get(prefix, prefix)
    return f"{prefix}_{match.group(2).zfill(5)}"


def _classify(path: Path, region: str, patch_id: str) -> dict[str, object]:
    ext = path.suffix.lower()
    is_visual = ext in VISUAL_RENDER_EXTENSIONS
    is_native = ext in NATIVE_SPECTRAL_EXTENSIONS

    evidence = []
    if region != "DESCONHECIDO":
        evidence.append("city_token_in_path")
    if patch_id:
        evidence.append("patch_token_in_filename")

    # Vínculo "forte" exigiria casar patch_id/asset_id com o contrato MV2-07
    # (que vive em outra branch). Sem esse cruzamento confirmado, jamais marcamos
    # link_strength=STRONG nem can_promote_to_canonical=true.
    if patch_id and region != "DESCONHECIDO":
        link_strength = "MEDIO"
        reason = "patch_token_presente_mas_sem_cruzamento_confirmado_com_contrato_mv2_07"
    elif region != "DESCONHECIDO":
        link_strength = "FRACO"
        reason = "apenas_token_de_cidade_no_caminho_sem_patch_id"
    else:
        link_strength = "NENHUM"
        reason = "sem_token_de_cidade_nem_patch_id"

    if is_native:
        recommended = "validar_como_raster_espectral_nativo_e_registrar_bandas_crs"
        reason_native = ""
    elif is_visual:
        recommended = "tratar_como_render_visual_NAO_espectral_candidato_a_rebalanceamento"
        reason_native = "render_visual_nao_substitui_raster_sentinel_espectral"
    else:
        recommended = "inspecionar_para_geometria_ou_metadata"
        reason_native = ""

    return {
        "extension": ext,
        "is_visual_render": is_visual,
        "is_native_spectral": is_native,
        "evidence_for_link": "|".join(evidence) if evidence else "none",
        "link_strength": link_strength,
        # Promoção a canônico SEMPRE bloqueada nesta etapa (sem vínculo forte).
        "can_promote_to_canonical": False,
        "reason_if_blocked": reason if not reason_native else f"{reason};{reason_native}",
        "recommended_action": recommended,
    }


def scan() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    count = 0
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if count >= MAX_FILES:
                break
            if not path.is_file() or _excluded(path):
                continue
            if path.suffix.lower() not in RELEVANT_EXTENSIONS:
                continue
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            count += 1
            region = infer_region_from_text(rel)
            patch_id = _infer_patch_id(path.name)
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            cls = _classify(path, region, patch_id)
            rows.append(
                {
                    "candidate_path": rel,
                    "filename": path.name,
                    "extension": cls["extension"],
                    "size_bytes": size,
                    "inferred_region": region,
                    "inferred_patch_id": patch_id,
                    "inferred_asset_id": "",
                    "evidence_for_link": cls["evidence_for_link"],
                    "link_strength": cls["link_strength"],
                    "can_promote_to_canonical": str(cls["can_promote_to_canonical"]).lower(),
                    "reason_if_blocked": cls["reason_if_blocked"],
                    "recommended_action": cls["recommended_action"],
                }
            )
    return rows


COLUMNS = [
    "candidate_path",
    "filename",
    "extension",
    "size_bytes",
    "inferred_region",
    "inferred_patch_id",
    "inferred_asset_id",
    "evidence_for_link",
    "link_strength",
    "can_promote_to_canonical",
    "reason_if_blocked",
    "recommended_action",
]


def main() -> None:
    ensure_output_dir()
    rows = scan()
    out = OUTPUT_DIR / "mv2_12_local_recovery_candidates.csv"
    write_csv(out, COLUMNS, rows)
    n_native = sum(1 for r in rows if r["extension"] in NATIVE_SPECTRAL_EXTENSIONS)
    n_visual = sum(1 for r in rows if r["extension"] in VISUAL_RENDER_EXTENSIONS)
    print(f"[mv2_12_scan] candidatos indexados: {len(rows)}")
    print(f"[mv2_12_scan] render visual (NAO espectral): {n_visual}")
    print(f"[mv2_12_scan] raster espectral nativo: {n_native}")
    print(f"[mv2_12_scan] saida: {out.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

"""MV2-13 Tarefa 16 — Validador de guardrails do Sentinel Binding.

Falha (exit != 0) se qualquer trava fail-closed do marco for violada.
"""

from __future__ import annotations

import json
import sys

from mv2_13_sentinel_binding_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    clean,
    read_csv_rows,
)

HEAVY_EXT = {".tif", ".tiff", ".jp2", ".safe", ".img", ".cog", ".npy", ".npz",
             ".png", ".jpg", ".jpeg", ".zip", ".gpkg", ".shp", ".nc", ".hdf"}
ALLOWED_EXT = {".csv", ".json", ".md", ".txt"}
MAX_BYTES = 2_000_000

REAL_STAC_OK = {"STAC_REAL_FUTURE_ELIGIBLE"}
BAD_BINDINGS = {"BINDING_WEAK", "BINDING_NONE", "BINDING_CONFLICT", "BINDING_INVALID"}


def find_violations() -> list[str]:
    v: list[str] = []
    if not OUTPUT_DIR.exists():
        return [f"diretório ausente: {OUTPUT_DIR}"]

    # 1-2. Só leve; nenhum bruto pesado.
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        ext = p.suffix.lower()
        if ext in HEAVY_EXT:
            v.append(f"bruto pesado em outputs_public: {rel}")
        elif ext not in ALLOWED_EXT:
            v.append(f"extensão não permitida: {rel} ({ext})")
        if p.stat().st_size > MAX_BYTES:
            v.append(f"arquivo acima de {MAX_BYTES} bytes: {rel}")

    summary_path = OUTPUT_DIR / "mv2_13_binding_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        # 3-12. Contadores fail-closed.
        for key in [
            "stac_real_executed", "downloads_executed", "crops_created",
            "spectral_features_created", "labels_created", "silver_created",
            "gold_created", "negatives_created", "native_rasters_created",
        ]:
            if s.get(key, 0) != 0:
                v.append(f"guardrail violado: {key}={s.get(key)} (deve ser 0)")
        if s.get("can_train") is not False:
            v.append("guardrail violado: can_train deve ser false")
        if s.get("sandbox_status") != "bloqueado":
            v.append("guardrail violado: sandbox_status deve ser bloqueado")
        if s.get("can_unlock_day10_now") is not False:
            v.append("guardrail violado: can_unlock_day10_now deve ser false (sem raster nativo)")
    else:
        v.append("summary JSON ausente")

    binding = read_csv_rows(OUTPUT_DIR / "mv2_13_binding_matrix.csv")
    gate = read_csv_rows(OUTPUT_DIR / "mv2_13_stac_gate_matrix.csv")
    assets = read_csv_rows(OUTPUT_DIR / "mv2_13_asset_inventory_normalized.csv")
    day10 = read_csv_rows(OUTPUT_DIR / "mv2_13_day10_spectral_gate.csv")
    gate_by_id = {clean(g.get("binding_id")): g for g in gate}

    for b in binding:
        strength = clean(b.get("binding_strength"))
        g = gate_by_id.get(clean(b.get("binding_id")), {})
        # 13. Binding fraco/none/conflito/inválido nunca STAC real.
        if strength in BAD_BINDINGS:
            if clean(b.get("stac_status")) in REAL_STAC_OK or clean(g.get("can_stac_real_next_step")) == "true":
                v.append(f"binding {clean(b.get('binding_id'))} fraco com STAC real")
        # 14. Sem patch e asset -> sem STAC real.
        if clean(g.get("can_stac_real_next_step")) == "true":
            if not clean(b.get("patch_id")) or not clean(b.get("asset_id")):
                v.append(f"STAC real sem patch/asset: {clean(b.get('binding_id'))}")
            # 15-16. Sem geometria/CRS/cena/datetime -> sem STAC real.
            if clean(g.get("geometry_ok")) != "true":
                v.append(f"STAC real sem geometria/crs: {clean(b.get('binding_id'))}")
            if not clean(b.get("scene_id")) and clean(b.get("acquisition_datetime")) == "":
                v.append(f"STAC real sem scene_id/datetime: {clean(b.get('binding_id'))}")

    # 17. PNG/NPZ/render/DINOv2 nunca native raster.
    for a in assets:
        if clean(a.get("is_native_sentinel_raster")) == "true":
            if clean(a.get("is_visual_render")) == "true" or clean(a.get("is_npz_visual")) == "true" or clean(a.get("is_dinov2_embedding")) == "true":
                v.append(f"asset render marcado como native raster: {clean(a.get('asset_id'))}")

    # 18. Dia 10 não desbloqueia sem raster nativo.
    for d in day10:
        if clean(d.get("can_unlock_day10_now")) == "true" and clean(d.get("has_native_raster")) != "true":
            v.append(f"day10 desbloqueado sem raster nativo: {clean(d.get('day10_gate_id'))}")

    # 19-20. Ausência não vira negativo; cidade/região não vira label.
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if summary.get("negatives_created", 0) != 0:
        v.append("ausência tratada como negativo (negatives_created != 0)")
    if summary.get("labels_created", 0) != 0:
        v.append("label criado a partir de cidade/região (labels_created != 0)")

    # 20b. Sem caminho absoluto privado nos outputs.
    for p in OUTPUT_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if "C:/Users" in txt or "C:\\Users" in txt:
                v.append(f"caminho absoluto privado em {p.name}")
    return v


def main() -> int:
    v = find_violations()
    if v:
        print("[mv2_13_validate] VIOLACOES:")
        for x in v:
            print(f"  - {x}")
        return 1
    n = sum(1 for p in OUTPUT_DIR.rglob("*") if p.is_file())
    print(f"[mv2_13_validate] OK — {n} arquivos leves; todos os guardrails fail-closed preservados")
    return 0


if __name__ == "__main__":
    sys.exit(main())

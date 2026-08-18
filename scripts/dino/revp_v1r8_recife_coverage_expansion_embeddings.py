"""REV-P v1r8 (SUSC-21b) -- Extracao de embedding DINOv2 para os patches de
Recife que JA TEM asset visual local mas nunca tiveram embedding rodado.

Diagnostico que motivou esta etapa (Tarefa 1 do SUSC-21b)
---------------------------------------------------------
O corpus visual de Recife em disco tem 52 patches
(`patches/thumbnails/recife/*__rgb_preview.png`, 256x256 RGB, git-ignored):
39 da grade oficial + 13 REC_NEG de evidencia negativa. Destes, apenas 23
tinham embedding (v1r3). Os outros 29 nunca foram processados -- e o asset
deles ja esta em disco, no MESMO formato e na MESMA raiz que alimentou os 23
originais (conferido contra `dino_local_asset_preprocessing_audit_v1qi.csv`:
`recife_00183` bate em dimensao 256x256, tamanho 111850 bytes e
sha256 curto f4bd7de0c36d6076). Portanto a ampliacao de cobertura NAO exige
baixar nada e NAO exige regerar preview.

Cuidado explicito com troca silenciosa de modalidade
----------------------------------------------------
Existe em `PROJETO/patches/thumbnails/recife/` um preview homonimo de 64x64
gerado por `generate_stacks.py` -- asset DIFERENTE do que o pipeline DINO
consumiu. Este script le exclusivamente a raiz de 256x256 e recusa qualquer
arquivo cuja dimensao nao bata com a auditada em v1qi.

Portao de reproducao (obrigatorio, fail-closed)
-----------------------------------------------
Antes de extrair qualquer vetor novo, o script RE-EXTRAI os embeddings de
patches que ja existem em v1r3 e compara com os vetores publicados. Se o
encoder local, o processador de imagem ou o pos-processamento divergirem, os
vetores novos nao seriam comparaveis com os antigos e toda a analise ampliada
estaria contaminada -- nesse caso o script sai FAIL_CLOSED sem escrever
expansao. Mesma disciplina que v1r5 usou para validar o Firth reimplementado
contra os coeficientes ja publicados.

Nao cria label. Nao treina nada. Nao baixa imagem. Nao gera imagem.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, EXPECTED_DINO_DIM, ROOT, SCHEMAS,
    _p, env_str, file_sha256_short, parse_embedding_from_row, read_csv,
    write_csv, write_schema, write_vector_csv_row,
)

IN_EMB = _p("REVP_V1R8_IN_EMB", DATASETS / "dino_recife_sedec_all_embeddings_v1r3.csv")
OUT_EMB = _p("REVP_V1R8_OUT_EMB", DATASETS / "dino_recife_coverage_expansion_embeddings_v1r8.csv")
OUT_AUDIT = _p("REVP_V1R8_OUT_AUDIT", DATASETS / "dino_recife_coverage_expansion_audit_v1r8.csv")
SCH_EMB = _p("REVP_V1R8_SCH_EMB", SCHEMAS / "dino_recife_coverage_expansion_embeddings_v1r8_schema.csv")

VISUAL_ROOT = _p("REVP_V1R8_VISUAL_ROOT", ROOT / "patches" / "thumbnails" / "recife")
MODEL_PATH_ENV = "REVP_DINO_MODEL_PATH"
DEFAULT_MODEL = ROOT / "local_only" / "models" / "dinov2-with-registers-base"

EXPECTED_PREVIEW_SIZE = (256, 256)
# Max absolute per-dimension deviation tolerated when reproducing a published
# vector. Anything above this means a different encoder/preprocessing path.
REPRO_ABS_TOL = 1e-4
REPRO_MIN_COS = 0.9999
N_REPRO_CHECKS = 3

META_FIELDS = ["embedding_id", "patch_id", "alias", "region", "relative_path", "path_hash",
               "asset_sha256_short", "image_width", "image_height", "model_name",
               "model_path_hash", "embedding_dim", "l2_normalized", "vector_norm",
               "dino_allowed_use", "review_only", "can_create_label", "can_train_model",
               "target_created", "source_stage", "new_imagery_downloaded",
               "preview_regenerated"]
AUDIT_FIELDS = ["audit_key", "audit_value"]


def _preview_paths() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(glob.glob(str(Path(VISUAL_ROOT) / "*__rgb_preview.png"))):
        stem = Path(p).name.replace("__rgb_preview.png", "")
        if not stem.startswith("recife_"):
            continue
        out[f"REC_{stem.split('recife_', 1)[1].upper()}"] = Path(p)
    return out


def _load_model(model_path: str) -> Any | None:
    try:
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        model.eval()
        return (processor, model)
    except Exception:
        return None


def _embed(bundle: Any, img_path: Path) -> tuple[list[float] | None, tuple[int, int]]:
    """Identical to the v1qj executor: CLS token of last_hidden_state, L2-normalized."""
    from PIL import Image
    import torch
    processor, model = bundle
    raw = Image.open(img_path)
    size = raw.size
    img = raw.convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    if not hasattr(outputs, "last_hidden_state"):
        return (None, size)
    vec = [float(v) for v in outputs.last_hidden_state[0, 0, :].tolist()]
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return (vec, size)


def _fail_closed(reason: str, detail: str) -> None:
    write_csv(OUT_EMB, [], META_FIELDS)
    write_csv(OUT_AUDIT, [{"audit_key": "status", "audit_value": reason},
                          {"audit_key": "detail", "audit_value": detail},
                          {"audit_key": "embeddings_written", "audit_value": "0"},
                          {"audit_key": "new_imagery_downloaded", "audit_value": "0"}],
              AUDIT_FIELDS)
    print(f"[v1r8] FAIL_CLOSED: {reason} -- {detail}")


def run() -> None:
    model_path = env_str(MODEL_PATH_ENV, "") or str(DEFAULT_MODEL)
    if not Path(model_path).is_dir():
        _fail_closed("FAIL_CLOSED_LOCAL_MODEL_MISSING", f"model_path={model_path}")
        return

    previews = _preview_paths()
    existing = read_csv(IN_EMB)
    known: dict[str, list[float]] = {}
    for r in existing:
        v = parse_embedding_from_row(r)
        if v is not None:
            known[r["patch_id"]] = v
    model_name = existing[0]["model_name"] if existing else "dinov2-with-registers-base"
    model_hash_published = existing[0]["model_path_hash"] if existing else ""

    missing = sorted(set(previews) - set(known))
    if not missing:
        _fail_closed("NO_EXPANSION_AVAILABLE", "every local preview already has an embedding")
        return

    bundle = _load_model(model_path)
    if bundle is None:
        _fail_closed("FAIL_CLOSED_MODEL_LOAD_FAILED", f"model_path={model_path}")
        return

    # ---- Reproduction gate ------------------------------------------------
    repro_targets = [p for p in sorted(known) if p in previews][:N_REPRO_CHECKS]
    repro_rows: list[tuple[str, float, float]] = []
    for pid in repro_targets:
        vec, size = _embed(bundle, previews[pid])
        if vec is None or len(vec) != len(known[pid]):
            _fail_closed("FAIL_CLOSED_REPRODUCTION_SHAPE_MISMATCH", f"patch={pid}")
            return
        ref = known[pid]
        max_abs = max(abs(a - b) for a, b in zip(vec, ref))
        cos = sum(a * b for a, b in zip(vec, ref))
        repro_rows.append((pid, max_abs, cos))

    worst_abs = max(r[1] for r in repro_rows)
    worst_cos = min(r[2] for r in repro_rows)
    if worst_abs > REPRO_ABS_TOL or worst_cos < REPRO_MIN_COS:
        _fail_closed(
            "FAIL_CLOSED_REPRODUCTION_GATE_FAILED",
            f"re-extracting published patches did not reproduce them "
            f"(max_abs_dev={worst_abs:.3e} tol={REPRO_ABS_TOL:.0e}; "
            f"min_cos={worst_cos:.6f} floor={REPRO_MIN_COS}) -- new vectors would not be "
            f"comparable with v1r3, expansion refused")
        return

    # ---- Expansion --------------------------------------------------------
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for i, pid in enumerate(missing, start=1):
        path = previews[pid]
        vec, size = _embed(bundle, path)
        if vec is None or len(vec) != EXPECTED_DINO_DIM:
            skipped.append(f"{pid}:embedding_failed")
            continue
        if size != EXPECTED_PREVIEW_SIZE:
            skipped.append(f"{pid}:unexpected_preview_size_{size[0]}x{size[1]}")
            continue
        rel = f"recife/{path.name}"
        base = {
            "embedding_id": f"V1R8_EMB_{i:05d}", "patch_id": pid, "alias": pid,
            "region": "RECIFE", "relative_path": rel,
            "path_hash": "", "asset_sha256_short": file_sha256_short(path),
            "image_width": size[0], "image_height": size[1],
            "model_name": model_name, "model_path_hash": model_hash_published,
            "embedding_dim": EXPECTED_DINO_DIM, "l2_normalized": "True",
            "vector_norm": round(sum(v * v for v in vec) ** 0.5, 6),
            "dino_allowed_use": "REVIEW_ONLY_REPRESENTATION", "review_only": "true",
            "can_create_label": "false", "can_train_model": "false",
            "target_created": "false", "source_stage": "V1R8_LOCAL_COVERAGE_EXPANSION",
            "new_imagery_downloaded": "false", "preview_regenerated": "false",
        }
        from revp_v1qg_v1qm_smoke_embedding_common import path_hash
        base["path_hash"] = path_hash(rel)
        rows.append(write_vector_csv_row(base, vec, EXPECTED_DINO_DIM))

    fields = META_FIELDS + [f"embedding_{i:03d}" for i in range(EXPECTED_DINO_DIM)]
    write_csv(OUT_EMB, rows, fields)
    write_schema(SCH_EMB, META_FIELDS, "v1r8_dino_recife_coverage_expansion_embeddings")

    audit = [
        ("visual_root_used", "patches/thumbnails/recife (256x256 rgb_preview, git-ignored)"),
        ("n_local_previews", str(len(previews))),
        ("n_with_embedding_before", str(len(known))),
        ("n_without_embedding_before", str(len(missing))),
        ("n_embeddings_written", str(len(rows))),
        ("n_with_embedding_after", str(len(known) + len(rows))),
        ("skipped", ";".join(skipped) if skipped else "none"),
        ("model_path_is_local_dir", "true"),
        ("model_name", model_name),
        ("reproduction_gate_patches", ";".join(p for p, _, _ in repro_rows)),
        ("reproduction_gate_max_abs_deviation", f"{worst_abs:.3e}"),
        ("reproduction_gate_min_cosine", f"{worst_cos:.8f}"),
        ("reproduction_gate_tolerance", f"abs<={REPRO_ABS_TOL:.0e} cos>={REPRO_MIN_COS}"),
        ("reproduction_gate_verdict", "PASS_NEW_VECTORS_COMPARABLE_WITH_V1R3"),
        ("preview_size_enforced", f"{EXPECTED_PREVIEW_SIZE[0]}x{EXPECTED_PREVIEW_SIZE[1]}"),
        ("new_imagery_downloaded", "0"),
        ("previews_regenerated", "0"),
        ("labels_created", "0"),
        ("classifier_trained", "false"),
        ("status", "COVERAGE_EXPANSION_COMPLETE_REVIEW_ONLY"),
    ]
    write_csv(OUT_AUDIT, [{"audit_key": k, "audit_value": v} for k, v in audit], AUDIT_FIELDS)
    print(f"[v1r8] previews={len(previews)} before={len(known)} written={len(rows)} "
          f"after={len(known)+len(rows)} repro_max_abs={worst_abs:.3e} "
          f"repro_min_cos={worst_cos:.8f}")


if __name__ == "__main__":
    run()

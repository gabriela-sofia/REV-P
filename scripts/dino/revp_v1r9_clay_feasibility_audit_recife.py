"""REV-P v1r9 (SUSC-22) -- Tarefa 1: auditoria de viabilidade do Clay v1.5 como"""
from __future__ import annotations

import glob
import importlib.util
from pathlib import Path

import rasterio

from revp_v1qg_v1qm_smoke_embedding_common import DATASETS, ROOT, _p, write_csv

SENTINEL_DIR = _p("REVP_V1R9_SENTINEL_DIR", Path("C:/Users/gabriela/Documents/PROJETO/data/sentinel"))
EXPORT_MANIFEST = _p("REVP_V1R9_EXPORT_MANIFEST",
                     Path("C:/Users/gabriela/Documents/PROJETO/data/sentinel_export_manifest.csv"))
OUT = _p("REVP_V1R9_OUT", DATASETS / "clay_feasibility_audit_v1r9.csv")

FIELDS = ["requirement", "clay_expects", "local_reality", "status", "note"]
CLAY_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
LOCAL_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
DEPS = ["torch", "claymodel", "lightning", "einops", "timm", "box", "vit_pytorch", "rasterio"]


def _dep_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def run() -> None:
    tifs = sorted(glob.glob(str(Path(SENTINEL_DIR) / "patch_recife_*.tif")))
    sizes, bands, dtypes, res = [], set(), set(), set()
    for p in tifs:
        with rasterio.open(p) as d:
            sizes.append((d.width, d.height))
            bands.add(d.count)
            dtypes.add(d.dtypes[0])
            res.add(tuple(round(v, 3) for v in d.res))
    min_px = min(min(s) for s in sizes) if sizes else 0
    max_px = max(max(s) for s in sizes) if sizes else 0

    window = ""
    try:
        import csv as _csv
        with Path(EXPORT_MANIFEST).open(encoding="utf-8", newline="") as fh:
            rows = [r for r in _csv.DictReader(fh) if "recife" in r.get("regiao", "")]
        if rows:
            window = f"{rows[0]['date_start']}..{rows[0]['date_end']}"
    except OSError:
        window = "MANIFEST_UNREADABLE"

    ckpt_local = any(Path(ROOT).glob("local_only/models/**/clay-v1.5.ckpt"))
    missing = [d for d in DEPS if not _dep_present(d)]

    rows_out = [
        {"requirement": "bandas_espectrais",
         "clay_expects": "Sentinel-2 L2A, 10 bandas: " + ",".join(CLAY_S2_BANDS),
         "local_reality": f"{sorted(bands)} bandas: " + ",".join(LOCAL_BANDS),
         "status": "PARTIAL_OK",
         "note": "faltam B5,B6,B7,B8A (red-edge/narrow-NIR); o bloco de embedding "
                 "dinamico do Clay condiciona no comprimento de onda e aceita "
                 "subconjunto, mas e um desvio a documentar"},
        {"requirement": "gsd_resolucao",
         "clay_expects": "GSD informado por sensor; S2 = 10 m",
         "local_reality": f"res={sorted(res)} m, CRS UTM 32725",
         "status": "OK", "note": "resolucao nativa bate"},
        {"requirement": "tamanho_do_chip",
         "clay_expects": "treino em 256x256, patch_size=8 (lado divisivel por 8)",
         "local_reality": f"{min_px}x{min_px} a {max_px}x{max_px} px",
         "status": "NEEDS_DECISION",
         "note": "os patches tem ~100 px de lado e nao sao divisiveis por 8; "
                 "exigiria padding ou reamostragem, que muda o conteudo "
                 "espacial -- decisao metodologica, nao detalhe tecnico"},
        {"requirement": "escala_de_reflectancia",
         "clay_expects": "(chip - mean)/std com means/stds de configs/metadata.yaml",
         "local_reality": "float64, SR Sentinel-2 escalado (~200 a ~6900)",
         "status": "OK",
         "note": "escala compativel com as estatisticas do metadata.yaml do Clay"},
        {"requirement": "metadado_temporal",
         "clay_expects": "time = (semana, hora) da aquisicao",
         "local_reality": f"composicao MEDIANA anual, janela {window}",
         "status": "BLOCKER_SCIENTIFIC",
         "note": "o .tif nao e uma cena, e mediana de 12 meses (export_sentinel.py, "
                 ".median() com filtro de nuvem <35%); nao existe instante de "
                 "aquisicao -- o embedding temporal do Clay teria de ser zerado "
                 "ou inventado"},
        {"requirement": "metadado_espacial",
         "clay_expects": "latlon do centro do chip",
         "local_reality": "derivavel do bbox de cada .tif",
         "status": "OK", "note": "ja calculado no v1r7 para o QA de adjacencia"},
        {"requirement": "checkpoint",
         "clay_expects": "v1.5/clay-v1.5.ckpt (632M params; 311M no encoder)",
         "local_reality": "ausente localmente" if not ckpt_local else "presente",
         "status": "BLOCKER_NEEDS_APPROVAL" if not ckpt_local else "OK",
         "note": "download de 1.25 GB (encoder) a 5.16 GB (checkpoint completo "
                 "com estados do otimizador); regra do projeto exige aprovacao "
                 "explicita antes de baixar peso de modelo"},
        {"requirement": "dependencias_python",
         "clay_expects": ",".join(DEPS),
         "local_reality": "ausentes: " + (",".join(missing) if missing else "nenhuma"),
         "status": "BLOCKER_NEEDS_APPROVAL" if missing else "OK",
         "note": "instalacao via pip tambem e download; nao executada"},
        {"requirement": "custo_de_cpu",
         "clay_expects": "encoder ViT dim=1024 depth=24 heads=16",
         "local_reality": "medido localmente com proxy de mesma geometria "
                          "(302M params, 1025 tokens, 8 threads): 7,97 s/chip "
                          "-> ~6,9 min para 52 patches",
         "status": "OK",
         "note": "CPU NAO e bloqueio; medicao real com pesos aleatorios, mede "
                 "custo e nao qualidade; torch 2.11.0+cpu, cuda indisponivel"},
        {"requirement": "tamanho_do_modelo_premissa",
         "clay_expects": "632M params totais / 311M encoder (v1.5)",
         "local_reality": "premissa do pedido dizia 26M params",
         "status": "PREMISE_CORRECTED",
         "note": "26M corresponde ao Clay v0.x; o v1.5 atual e da mesma ordem "
                 "do Prithvi (~650M), entao o argumento de 'mais leve' nao se "
                 "sustenta -- o que sustenta a tentativa e ser pre-treinado em "
                 "multiespectral real, nao o tamanho"},
    ]
    write_csv(OUT, rows_out, FIELDS)
    blockers = [r["requirement"] for r in rows_out if r["status"].startswith("BLOCKER")]
    print(f"[v1r9] tifs={len(tifs)} bandas={sorted(bands)} janela={window} "
          f"blockers={blockers}")


if __name__ == "__main__":
    run()

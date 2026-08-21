"""REV-P v2fg (E2/E3) -- constrói o corpus e os medoids da governança DINOv2.

Lê SÓ embeddings reais já persistidos no repositório
(`datasets/dino_*embeddings*.csv`, backbone `dinov2-with-registers-base`,
768D, L2-normalizados -- gerados pelos executores v1qj/v1r0-v1r8) e produz:

  * `datasets/dinov2_governance_corpus_v2fg.csv`   -- auditoria candidato-a-candidato
  * `datasets/dinov2_governance_medoids_v2fg.json` -- medoids + vetores + config do gate OOD
  * `datasets/dinov2_governance_summary_v2fg.csv`  -- contagens (candidatos/válidos/bloqueados)
  * `datasets/schemas/dinov2_governance_corpus_v2fg_schema.csv`

Regras que este pipeline não quebra:
  * nenhum vetor é gerado aqui -- só lido, validado e indexado;
  * linha `mock`/fixture/sintética nunca entra no corpus;
  * medoid é o patch de MAIOR similaridade de cosseno média dentro do
    recorte, mesma definição já publicada em
    `outputs_public/tables/table_dino_medoids.csv`;
  * o limiar OOD default é derivado do próprio corpus (percentil), não
    arbitrado -- e fica registrado com sua base de cálculo;
  * a divergência contra as tabelas públicas é medida e registrada, não
    escondida.

Uso:
    python scripts/dino/revp_v2fg_build_dinov2_governance_corpus.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from revp_v2fg_dinov2_governance_engine import (  # noqa: E402
    EMBEDDING_DIM, L2_TOLERANCE, MANIFEST_VERSION, METHODOLOGICAL_NOTE,
    cosine_similarity, parse_embedding_row, region_of_patch, validate_embedding,
)

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
PUBLIC_TABLES = ROOT / "outputs_public" / "tables"

OUT_CORPUS = DATASETS / "dinov2_governance_corpus_v2fg.csv"
OUT_MANIFEST = DATASETS / "dinov2_governance_medoids_v2fg.json"
OUT_SUMMARY = DATASETS / "dinov2_governance_summary_v2fg.csv"
OUT_SCHEMA = SCHEMAS / "dinov2_governance_corpus_v2fg_schema.csv"

SOURCE_GLOB = "dino_*embedding*.csv"
EXPECTED_BACKBONE = "dinov2-with-registers-base"

#: Percentil (do corpus) usado como limiar OOD default. Escolha explícita e
#: registrada no manifesto: 5% das amostras REAIS do corpus ficam abaixo dele.
OOD_PERCENTILE = 0.05

FIXTURE_TERMS = ("fixture", "synthetic", "test_only", "dummy", "mock", "sample_random")
FORBIDDEN_TRUE_FIELDS = ("can_create_label", "can_train_model", "target_created", "ground_truth")

#: DESVIO DELIBERADO de `is_fixture_patch()` do v1pg/v1pm: aquele helper usa
#: `^(REC|PET|CWB)_0{3}\d{2}$`, que casa com patches REAIS de numero < 100 --
#: entre eles `REC_00019` (patch de linhagem TP1, ver
#: docs/v2bd_REC_00019_lineage_findings.md) e `CUR_00038` (medoid de Curitiba
#: publicado em outputs_public/tables/table_dino_medoids.csv). Aplicar o
#: regex aqui excluiria dois patches reais e documentados do corpus de
#: governanca. Este pipeline usa apenas a triagem textual por termo de
#: fixture/mock, que nao tem esse falso positivo.

CORPUS_FIELDS = [
    "corpus_row_id", "patch_id", "region", "source_file", "source_row_index",
    "embedding_id", "model_name", "declared_embedding_dim", "declared_l2_normalized",
    "observed_dim", "observed_l2_norm", "status", "blocking_reason",
    "is_region_medoid", "cosine_to_region_medoid", "nearest_medoid_region",
    "cosine_to_nearest_medoid", "nearest_medoid_matches_own_region",
    "methodological_note",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def _sha256_short(path: Path, n: int = 16) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentil de sequencia vazia")
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_values[int(k)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def _is_fixture(patch_id: str, row: dict[str, str]) -> bool:
    blob = " ".join(str(row.get(k, "")) for k in ("alias", "relative_path", "source_stage", "smoke_id")).lower()
    return any(term in blob for term in FIXTURE_TERMS)


def discover_sources() -> list[Path]:
    """CSVs de embedding reais, em ordem determinística (lexicográfica)."""
    out = []
    for path in sorted(DATASETS.glob(SOURCE_GLOB)):
        with path.open(encoding="utf-8-sig", newline="") as fh:
            header = csv.DictReader(fh).fieldnames or []
        if any(re.match(r"^embedding_\d+$", c) for c in header):
            out.append(path)
    return out


def collect_candidates(sources: list[Path]) -> list[dict[str, Any]]:
    """Uma linha por candidato, já validada e desduplicada.

    Precedência em caso de `patch_id` repetido: primeiro arquivo em ordem
    lexicográfica. Se o vetor duplicado divergir do já aceito, a linha é
    bloqueada com motivo próprio (não é descartada em silêncio).
    """
    rows: list[dict[str, Any]] = []
    accepted: dict[str, list[float]] = {}
    seq = 0

    for path in sources:
        rel = _rel(path)
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for idx, raw in enumerate(csv.DictReader(fh)):
                seq += 1
                patch_id = (raw.get("patch_id") or "").strip().upper()
                vec = parse_embedding_row(raw)
                observed_dim = len(vec) if vec is not None else 0
                observed_norm = (
                    math.sqrt(sum(x * x for x in vec)) if vec else 0.0)
                entry: dict[str, Any] = {
                    "corpus_row_id": f"DGOV_V2FG_{seq:05d}",
                    "patch_id": patch_id,
                    "region": region_of_patch(patch_id),
                    "source_file": rel,
                    "source_row_index": idx,
                    "embedding_id": (raw.get("embedding_id") or "").strip(),
                    "model_name": (raw.get("model_name") or "").strip(),
                    "declared_embedding_dim": (raw.get("embedding_dim") or "").strip(),
                    "declared_l2_normalized": (raw.get("l2_normalized") or "").strip().lower(),
                    "observed_dim": observed_dim,
                    "observed_l2_norm": round(observed_norm, 8),
                    "status": "VALID",
                    "blocking_reason": "",
                    "methodological_note": METHODOLOGICAL_NOTE,
                    "_vector": vec,
                }

                def block(reason: str) -> None:
                    entry["status"] = "BLOCKED"
                    entry["blocking_reason"] = reason
                    entry["_vector"] = None

                if not patch_id:
                    block("PATCH_ID_AUSENTE")
                elif _is_fixture(patch_id, raw):
                    block("PATCH_FIXTURE_OU_SINTETICO")
                elif entry["region"] == "UNKNOWN":
                    block("REGIAO_NAO_RESOLVIDA_PELO_PATCH_ID")
                elif EXPECTED_BACKBONE not in entry["model_name"]:
                    block(f"BACKBONE_INESPERADO: {entry['model_name'] or 'ausente'}")
                elif any(str(raw.get(f, "false")).strip().lower() == "true"
                         for f in FORBIDDEN_TRUE_FIELDS):
                    block("GUARDRAIL_CAMPO_PROIBIDO_TRUE")
                else:
                    ok, reason = validate_embedding(vec, EMBEDDING_DIM, L2_TOLERANCE)
                    if not ok:
                        block(f"EMBEDDING_INVALIDO: {reason}")
                    elif patch_id in accepted:
                        prev = accepted[patch_id]
                        sim = cosine_similarity(vec, prev)  # type: ignore[arg-type]
                        if abs(sim - 1.0) <= 1e-6:
                            block("DUPLICATA_IDENTICA_DE_PATCH_JA_ACEITO")
                        else:
                            block(f"DUPLICATA_DIVERGENTE_DE_PATCH_JA_ACEITO: cos={sim:.6f}")
                    else:
                        accepted[patch_id] = vec  # type: ignore[assignment]
                rows.append(entry)
    return rows


def compute_medoid(ids: list[str], vectors: dict[str, list[float]]) -> tuple[str, float] | None:
    """Medoid = maior similaridade de cosseno média dentro do recorte.

    Mesma definição de `outputs_public/tables/table_dino_medoids.csv`.
    Recorte com um único elemento não tem medoid definido (média sobre
    conjunto vazio) -- retorna None em vez de inventar 1.0.
    """
    if len(ids) < 2:
        return None
    best: tuple[str, float] | None = None
    for i in ids:
        mean = sum(cosine_similarity(vectors[i], vectors[j]) for j in ids if j != i) / (len(ids) - 1)
        if best is None or mean > best[1] or (mean == best[1] and i < best[0]):
            best = (i, mean)
    return best


def cross_check_published(vectors: dict[str, list[float]]) -> dict[str, Any]:
    """Compara o corpus atual com as tabelas públicas já versionadas.

    Não corrige nem sobrescreve nada: apenas mede e registra. As tabelas
    públicas foram geradas por uma rodada local (`local_runs/`, não
    persistida no repo), então divergência aqui é informação real sobre
    reprodutibilidade, não erro deste pipeline.
    """
    out: dict[str, Any] = {"similarity_matrix": None, "medoids": None}

    matrix_path = PUBLIC_TABLES / "table_dino_similarity_matrix.csv"
    if matrix_path.exists():
        with matrix_path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = [c for c in (reader.fieldnames or []) if c != "patch_id"]
            diffs: list[float] = []
            overlap: set[str] = set()
            for row in reader:
                a = (row.get("patch_id") or "").strip().upper()
                if a not in vectors:
                    continue
                overlap.add(a)
                for b in cols:
                    key = b.strip().upper()
                    if key not in vectors or key == a:
                        continue
                    try:
                        published = float(row[b])
                    except (TypeError, ValueError):
                        continue
                    diffs.append(abs(cosine_similarity(vectors[a], vectors[key]) - published))
        if diffs:
            out["similarity_matrix"] = {
                "source": _rel(matrix_path),
                "overlapping_patches": len(overlap),
                "pairs_compared": len(diffs),
                "max_abs_diff": round(max(diffs), 6),
                "mean_abs_diff": round(sum(diffs) / len(diffs), 6),
                "note": (
                    "A matriz publica foi produzida por uma rodada local anterior (local_runs/, "
                    "nao persistida no repositorio). Divergencia > 0 significa que os vetores hoje "
                    "em datasets/ nao reproduzem bit-a-bit aquela rodada -- limitacao real de "
                    "reprodutibilidade, registrada e nao contornada."),
            }

    medoids_path = PUBLIC_TABLES / "table_dino_medoids.csv"
    if medoids_path.exists():
        published_rows = []
        with medoids_path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                published_rows.append({
                    "scope": (row.get("scope") or "").strip(),
                    "patch_id": (row.get("patch_id") or "").strip().upper(),
                    "mean_similarity_within_scope": (row.get("mean_similarity_within_scope") or "").strip(),
                    "patch_in_current_corpus": (row.get("patch_id") or "").strip().upper() in vectors,
                })
        out["medoids"] = {
            "source": _rel(medoids_path),
            "published": published_rows,
            "note": (
                "Os medoids publicados foram calculados sobre o recorte de 12 patches de "
                "table_dino_embedding_inventory.csv. O corpus de governanca v2fg cobre todos os "
                "embeddings reais persistidos em datasets/, logo os medoids podem divergir por "
                "recorte -- comparacao registrada para rastreabilidade, nenhuma tabela publica "
                "foi alterada."),
        }
    return out


def build(dry_run: bool = False) -> dict[str, Any]:
    sources = discover_sources()
    if not sources:
        raise SystemExit(
            f"nenhum CSV de embedding real encontrado em {_rel(DATASETS)}/{SOURCE_GLOB} -- "
            "o corpus de governanca nao pode ser construido a partir de nada.")

    rows = collect_candidates(sources)
    valid_rows = [r for r in rows if r["status"] == "VALID"]
    vectors: dict[str, list[float]] = {r["patch_id"]: r["_vector"] for r in valid_rows}

    by_region: dict[str, list[str]] = {}
    for pid in sorted(vectors):
        by_region.setdefault(region_of_patch(pid), []).append(pid)

    medoids: list[dict[str, Any]] = []
    source_of: dict[str, str] = {r["patch_id"]: r["source_file"] for r in valid_rows}
    for region in sorted(by_region):
        found = compute_medoid(by_region[region], vectors)
        if found is None:
            continue
        pid, mean_sim = found
        medoids.append({
            "scope": region,
            "scope_kind": "region",
            "region": region,
            "patch_id": pid,
            "n_in_scope": len(by_region[region]),
            "mean_cosine_within_scope": round(mean_sim, 6),
            "source_file": source_of[pid],
            "definition": "Maior similaridade cosseno media no recorte analisado.",
            "embedding": vectors[pid],
        })

    corpus_medoid = compute_medoid(sorted(vectors), vectors)
    if corpus_medoid is not None:
        pid, mean_sim = corpus_medoid
        medoids.append({
            "scope": "CORPUS",
            "scope_kind": "corpus",
            "region": region_of_patch(pid),
            "patch_id": pid,
            "n_in_scope": len(vectors),
            "mean_cosine_within_scope": round(mean_sim, 6),
            "source_file": source_of[pid],
            "definition": "Maior similaridade cosseno media no recorte analisado.",
            "embedding": vectors[pid],
        })

    region_medoids = {m["region"]: m for m in medoids if m["scope_kind"] == "region"}

    # -- diagnósticos sobre o próprio corpus (base do limiar OOD) -------- #
    nearest_sims: list[float] = []
    concordant = 0
    for r in valid_rows:
        pid = r["patch_id"]
        own = region_medoids.get(r["region"])
        r["cosine_to_region_medoid"] = (
            round(cosine_similarity(vectors[pid], own["embedding"]), 6) if own else "")
        r["is_region_medoid"] = "true" if own and own["patch_id"] == pid else "false"
        ranked = sorted(
            ((round(cosine_similarity(vectors[pid], m["embedding"]), 6), m["region"])
             for m in region_medoids.values()),
            key=lambda t: (-t[0], t[1]))
        top_sim, top_region = ranked[0]
        r["cosine_to_nearest_medoid"] = top_sim
        r["nearest_medoid_region"] = top_region
        matches = top_region == r["region"]
        r["nearest_medoid_matches_own_region"] = "true" if matches else "false"
        concordant += int(matches)
        nearest_sims.append(top_sim)

    nearest_sims.sort()
    threshold = round(_percentile(nearest_sims, OOD_PERCENTILE), 6) if nearest_sims else None
    concordance_rate = round(concordant / len(valid_rows), 6) if valid_rows else None

    blocked_by_reason: dict[str, int] = {}
    for r in rows:
        if r["status"] == "BLOCKED":
            key = r["blocking_reason"].split(":")[0]
            blocked_by_reason[key] = blocked_by_reason.get(key, 0) + 1

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": _now_iso(),
        "generated_by": _rel(Path(__file__)),
        "model": {
            "model_name": f"facebook/{EXPECTED_BACKBONE}",
            "embedding_dim": EMBEDDING_DIM,
            "l2_normalized": True,
            "l2_tolerance": L2_TOLERANCE,
        },
        "corpus": {
            "source_files": [
                {"path": _rel(p), "sha256_short": _sha256_short(p)} for p in sources],
            "candidates": len(rows),
            "valid": len(valid_rows),
            "blocked": len(rows) - len(valid_rows),
            "blocked_by_reason": dict(sorted(blocked_by_reason.items())),
            "patches_by_region": {k: len(v) for k, v in sorted(by_region.items())},
            "audit_csv": _rel(OUT_CORPUS),
        },
        "medoid_definition": (
            "Medoid = patch com maior similaridade de cosseno media dentro do recorte "
            "(mesma definicao de outputs_public/tables/table_dino_medoids.csv)."),
        "ood_gate": {
            "threshold_default": threshold,
            "threshold_basis": (
                f"percentil {OOD_PERCENTILE:.0%} da similaridade de cosseno de cada embedding "
                f"valido do corpus ao medoid regional mais proximo (n={len(nearest_sims)}). "
                "Derivado do corpus real, nao arbitrado."),
            "env_override": "REVP_DINOV2_OOD_THRESHOLD",
            "distribution": {
                "n": len(nearest_sims),
                "min": round(nearest_sims[0], 6) if nearest_sims else None,
                "p05": threshold,
                "p50": round(_percentile(nearest_sims, 0.50), 6) if nearest_sims else None,
                "p95": round(_percentile(nearest_sims, 0.95), 6) if nearest_sims else None,
                "max": round(nearest_sims[-1], 6) if nearest_sims else None,
            },
        },
        "medoids": medoids,
        "diagnostics": {
            "territorial_concordance": {
                "definition": (
                    "fracao dos embeddings validos cujo medoid regional mais proximo e o da "
                    "propria regiao do patch_id"),
                "concordant": concordant,
                "total": len(valid_rows),
                "rate": concordance_rate,
                "reading": (
                    "Concordancia parcial: o medoid territorial DINOv2 e evidencia estrutural "
                    "fraca sobre regiao, nao um classificador territorial. Divergencia entre "
                    "regiao solicitada e regiao visualmente sugerida deve ser lida como sinal "
                    "para revisao humana, nunca como veredito."),
            },
        },
        "cross_check_published": cross_check_published(vectors),
        "scientific_boundary": [
            "DINOv2 nao entra no modelo fisico de Firth nem em seus coeficientes.",
            "Nenhum embedding, similaridade ou medoid e rotulo, classe, alvo ou confirmacao de evento.",
            "Mock/fixture nunca entra neste corpus (ver blocked_by_reason).",
        ],
        "methodological_note": METHODOLOGICAL_NOTE,
    }

    summary = [
        {"stat_key": "sources", "stat_value": str(len(sources))},
        {"stat_key": "candidates", "stat_value": str(len(rows))},
        {"stat_key": "valid", "stat_value": str(len(valid_rows))},
        {"stat_key": "blocked", "stat_value": str(len(rows) - len(valid_rows))},
        {"stat_key": "regions", "stat_value": str(len(by_region))},
        {"stat_key": "region_medoids", "stat_value": str(len(region_medoids))},
        {"stat_key": "ood_threshold_default", "stat_value": "" if threshold is None else f"{threshold:.6f}"},
        {"stat_key": "territorial_concordance_rate",
         "stat_value": "" if concordance_rate is None else f"{concordance_rate:.6f}"},
    ] + [
        {"stat_key": f"blocked__{k}", "stat_value": str(v)} for k, v in sorted(blocked_by_reason.items())
    ]

    if not dry_run:
        OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
        with OUT_CORPUS.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CORPUS_FIELDS)
            writer.writeheader()
            for r in rows:
                writer.writerow({f: r.get(f, "") for f in CORPUS_FIELDS})

        OUT_MANIFEST.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        with OUT_SUMMARY.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary)

        SCHEMAS.mkdir(parents=True, exist_ok=True)
        with OUT_SCHEMA.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["field", "description"])
            writer.writeheader()
            for field in CORPUS_FIELDS:
                writer.writerow({
                    "field": field,
                    "description": f"dinov2_governance_corpus_v2fg: {field}.",
                })

    return {"manifest": manifest, "rows": rows, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="calcula tudo e imprime o resumo, sem escrever artefato.")
    args = parser.parse_args()

    result = build(dry_run=args.dry_run)
    manifest = result["manifest"]
    corpus = manifest["corpus"]
    print(f"fontes reais lidas .......... {len(corpus['source_files'])}")
    print(f"candidatos .................. {corpus['candidates']}")
    print(f"validos ..................... {corpus['valid']}")
    print(f"bloqueados .................. {corpus['blocked']}")
    for reason, count in corpus["blocked_by_reason"].items():
        print(f"    {reason}: {count}")
    print(f"medoids regionais ........... "
          f"{[(m['region'], m['patch_id']) for m in manifest['medoids'] if m['scope_kind'] == 'region']}")
    print(f"limiar OOD default .......... {manifest['ood_gate']['threshold_default']}")
    print(f"concordancia territorial .... "
          f"{manifest['diagnostics']['territorial_concordance']['rate']}")
    if args.dry_run:
        print("[dry-run] nenhum arquivo escrito.")
    else:
        for path in (OUT_CORPUS, OUT_MANIFEST, OUT_SUMMARY, OUT_SCHEMA):
            print(f"escrito: {_rel(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

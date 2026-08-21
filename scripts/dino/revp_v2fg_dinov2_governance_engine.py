"""REV-P v2fg -- `Dinov2GovernanceEngine`: camada de governança visual DINOv2.

O que esta camada É: validação de domínio (gate OOD), similaridade
territorial contra medoids reais persistidos, e produção de evidência
auditável.

O que esta camada NÃO É (fronteira dura, herdada de
`revp_fase1_conclusao_dino_ab_test.md` e das LIMITATIONS do SUSC-20D):
  * NÃO entra no modelo físico de Firth nem em seus coeficientes.
  * NÃO altera `score.value` nem `score.confidence_interval`.
  * NÃO é rótulo, classe, alvo ou confirmação de evento observado.
  * NÃO bloqueia inferência em silêncio: todo caminho termina num estado
    explícito e carimbado na resposta.

Insumos (reais, versionados no repo -- ver
`scripts/dino/revp_v2fg_build_dinov2_governance_corpus.py`):
  * `datasets/dinov2_governance_medoids_v2fg.json`  -- medoids + vetores + config do gate
  * `datasets/dinov2_governance_corpus_v2fg.csv`    -- auditoria candidato-a-candidato
Sem esses artefatos a engine reporta `governance_unavailable` -- nunca
inventa medoid, embedding ou similaridade.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"

MANIFEST_PATH = DATASETS / "dinov2_governance_medoids_v2fg.json"
CORPUS_PATH = DATASETS / "dinov2_governance_corpus_v2fg.csv"

MANIFEST_VERSION = "dinov2_governance.v2fg.1"
EMBEDDING_DIM = 768
L2_TOLERANCE = 1e-4

# Estados possíveis do gate. Nenhum deles é silencioso.
STATUS_IN_DOMAIN = "in_domain"
STATUS_OUT_OF_DOMAIN = "out_of_domain"
STATUS_NO_VISUAL_EVIDENCE = "no_visual_evidence"
STATUS_INVALID_EMBEDDING = "invalid_embedding"
STATUS_UNAVAILABLE = "governance_unavailable"

TERRITORIAL_MATCH = "match"
TERRITORIAL_MISMATCH = "mismatch"
TERRITORIAL_NA = "not_applicable"

#: `region_registry.py` usa nomes minúsculos; o corpus DINO usa o código de
#: região do projeto (`normalize_region` em v1pg/v1pm).
REGION_ALIASES = {
    "recife": "RECIFE", "rec": "RECIFE",
    "curitiba": "CURITIBA", "cwb": "CURITIBA", "cur": "CURITIBA",
    "petropolis": "PET", "petrópolis": "PET", "pet": "PET",
}

_DIM_COL_RE = re.compile(r"^embedding_(\d+)$")

METHODOLOGICAL_NOTE = (
    "Resultado estrutural destinado a revisao. Nao constitui ground truth operacional, "
    "confirmacao de evento observado, classe, label, predicao ou treinamento supervisionado."
)


def normalize_region(raw: str | None) -> str:
    key = (raw or "").strip().lower()
    return REGION_ALIASES.get(key, (raw or "").strip().upper() or "UNKNOWN")


def region_of_patch(patch_id: str) -> str:
    pid = (patch_id or "").strip().upper()
    if pid.startswith("REC"):
        return "RECIFE"
    if pid.startswith("PET"):
        return "PET"
    if pid.startswith("CUR") or pid.startswith("CWB"):
        return "CURITIBA"
    return "UNKNOWN"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosseno clássico. Não assume normalização prévia -- se os dois vetores
    já são unitários (caso do corpus v2fg) o resultado é o produto interno."""
    if len(a) != len(b):
        raise ValueError(f"dimensoes incompativeis: {len(a)} != {len(b)}")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 0.0 or nb <= 0.0:
        raise ValueError("vetor de norma zero nao tem cosseno definido")
    return dot / (na * nb)


def parse_embedding_row(row: dict[str, str]) -> list[float] | None:
    """Lê colunas `embedding_000..embedding_767` de uma linha de CSV."""
    dims = sorted(((int(m.group(1)), c) for c in row if (m := _DIM_COL_RE.match(c))))
    if not dims:
        return None
    try:
        return [float(row[c]) for _, c in dims]
    except (TypeError, ValueError):
        return None


def validate_embedding(vec: Any, expected_dim: int = EMBEDDING_DIM,
                       tolerance: float = L2_TOLERANCE) -> tuple[bool, str]:
    """(ok, motivo). Motivo vazio quando ok."""
    if vec is None:
        return False, "embedding_ausente"
    try:
        values = [float(x) for x in vec]
    except (TypeError, ValueError):
        return False, "embedding_nao_numerico"
    if len(values) != expected_dim:
        return False, f"dimensao_invalida: {len(values)} != {expected_dim}"
    if not all(math.isfinite(x) for x in values):
        return False, "embedding_com_valor_nao_finito"
    norm = math.sqrt(sum(x * x for x in values))
    if norm <= 0.0:
        return False, "embedding_de_norma_zero"
    if abs(norm - 1.0) > tolerance:
        return False, f"embedding_nao_l2_normalizado: norma={norm:.6f}"
    return True, ""


class Dinov2GovernanceEngine:
    """Governança visual: gate OOD + medoid territorial + evidência de auditoria.

    Carrega apenas artefatos reais persistidos. Se o manifesto não existir,
    a engine fica `available == False` e todo `evaluate()` devolve
    `governance_unavailable` com o motivo -- comportamento fail-open para o
    score físico (que nunca depende do DINO) e fail-closed para a evidência
    visual (que nunca é fabricada).
    """

    def __init__(self, manifest_path: Path | None = None, corpus_path: Path | None = None,
                 ood_threshold: float | None = None, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else ROOT
        self.manifest_path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
        self.corpus_path = Path(corpus_path) if corpus_path is not None else CORPUS_PATH
        self.manifest: dict[str, Any] = {}
        self.medoids: list[dict[str, Any]] = []
        self.unavailable_reason: str = ""
        self._patch_index: dict[str, dict[str, str]] = {}
        self._vector_cache: dict[str, list[float]] = {}
        self._source_cache: dict[str, dict[str, list[float]]] = {}
        self._threshold_override = ood_threshold
        self._load_manifest()

    # -- carga ---------------------------------------------------------- #

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            self.unavailable_reason = (
                f"manifesto_de_medoids_ausente: {self._rel(self.manifest_path)} "
                "(rode scripts/dino/revp_v2fg_build_dinov2_governance_corpus.py)")
            return
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            self.unavailable_reason = f"manifesto_ilegivel: {type(exc).__name__}: {exc}"
            return
        if manifest.get("manifest_version") != MANIFEST_VERSION:
            self.unavailable_reason = (
                f"manifesto_de_versao_incompativel: {manifest.get('manifest_version')!r} "
                f"!= {MANIFEST_VERSION!r}")
            return
        medoids = [m for m in manifest.get("medoids", []) if m.get("scope_kind") == "region"]
        valid_medoids: list[dict[str, Any]] = []
        for m in medoids:
            ok, reason = validate_embedding(m.get("embedding"))
            if ok:
                valid_medoids.append(m)
            else:
                self.unavailable_reason = (
                    f"medoid_invalido_no_manifesto: {m.get('patch_id')} -> {reason}")
                return
        if not valid_medoids:
            self.unavailable_reason = "manifesto_sem_medoid_regional_valido"
            return
        self.manifest = manifest
        self.medoids = valid_medoids

    @property
    def available(self) -> bool:
        return bool(self.medoids)

    @property
    def ood_threshold(self) -> float | None:
        """Precedência: argumento do construtor > env > manifesto."""
        if self._threshold_override is not None:
            return float(self._threshold_override)
        env = os.environ.get("REVP_DINOV2_OOD_THRESHOLD", "").strip()
        if env:
            try:
                return float(env)
            except ValueError:
                pass
        gate = self.manifest.get("ood_gate") or {}
        value = gate.get("threshold_default")
        return float(value) if value is not None else None

    @property
    def threshold_source(self) -> str:
        if self._threshold_override is not None:
            return "argumento_explicito"
        if os.environ.get("REVP_DINOV2_OOD_THRESHOLD", "").strip():
            return "env:REVP_DINOV2_OOD_THRESHOLD"
        return "manifesto:ood_gate.threshold_default"

    def _rel(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.root)).replace(os.sep, "/")
        except ValueError:
            return Path(path).name

    # -- resolução de embedding por patch ------------------------------- #

    def _load_patch_index(self) -> dict[str, dict[str, str]]:
        if self._patch_index or not self.corpus_path.exists():
            return self._patch_index
        with self.corpus_path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") != "VALID":
                    continue
                self._patch_index[row["patch_id"].strip().upper()] = row
        return self._patch_index

    def _vectors_of_source(self, rel_path: str) -> dict[str, list[float]]:
        if rel_path in self._source_cache:
            return self._source_cache[rel_path]
        table: dict[str, list[float]] = {}
        path = self.root / rel_path
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    vec = parse_embedding_row(row)
                    pid = (row.get("patch_id") or "").strip().upper()
                    if vec is not None and pid and pid not in table:
                        table[pid] = vec
        self._source_cache[rel_path] = table
        return table

    def embedding_for_patch(self, patch_id: str) -> list[float] | None:
        """Vetor real do patch, lido do CSV de origem registrado na auditoria
        do corpus. None se o patch não estiver no corpus válido."""
        pid = (patch_id or "").strip().upper()
        if not pid:
            return None
        if pid in self._vector_cache:
            return self._vector_cache[pid]
        row = self._load_patch_index().get(pid)
        if row is None:
            return None
        vec = self._vectors_of_source(row.get("source_file", "")).get(pid)
        if vec is None:
            return None
        ok, _ = validate_embedding(vec)
        if not ok:
            return None
        self._vector_cache[pid] = vec
        return vec

    def known_patch_ids(self) -> list[str]:
        return sorted(self._load_patch_index())

    # -- avaliação ------------------------------------------------------ #

    def similarity_ranking(self, embedding: list[float]) -> list[dict[str, Any]]:
        """Similaridade de cosseno contra TODOS os medoids regionais,
        ordenada do mais próximo ao mais distante. É a evidência bruta --
        a resposta nunca mostra só o vencedor."""
        ranking = [
            {
                "region": m["region"],
                "medoid_patch_id": m["patch_id"],
                "cosine_similarity": round(cosine_similarity(embedding, m["embedding"]), 6),
            }
            for m in self.medoids
        ]
        ranking.sort(key=lambda r: (-r["cosine_similarity"], r["region"]))
        return ranking

    def audit_block(self) -> dict[str, Any]:
        """Rastro de auditoria: de onde veio cada número. Sem URL, sem
        identificador que não exista em disco."""
        corpus = self.manifest.get("corpus") or {}
        gate = self.manifest.get("ood_gate") or {}
        return {
            "manifest_path": self._rel(self.manifest_path),
            "manifest_version": self.manifest.get("manifest_version"),
            "manifest_generated_at": self.manifest.get("generated_at"),
            "corpus_path": self._rel(self.corpus_path),
            "corpus_candidates": corpus.get("candidates"),
            "corpus_valid": corpus.get("valid"),
            "corpus_blocked": corpus.get("blocked"),
            "model_name": (self.manifest.get("model") or {}).get("model_name"),
            "embedding_dim": (self.manifest.get("model") or {}).get("embedding_dim"),
            "medoid_definition": self.manifest.get("medoid_definition"),
            "ood_threshold_basis": gate.get("threshold_basis"),
            "ood_threshold_source": self.threshold_source,
            "territorial_concordance_in_corpus": (
                (self.manifest.get("diagnostics") or {}).get("territorial_concordance") or {}
            ).get("rate"),
            "methodological_note": METHODOLOGICAL_NOTE,
        }

    def _unavailable(self, requested_region: str | None, note: str) -> dict[str, Any]:
        return {
            "status": STATUS_UNAVAILABLE,
            "query_patch_id": None,
            "cosine_similarity": None,
            "nearest_medoid_patch_id": None,
            "suggested_region": None,
            "requested_region": normalize_region(requested_region) if requested_region else None,
            "territorial_match": TERRITORIAL_NA,
            "ood_threshold": self.ood_threshold,
            "ranking": [],
            "audit": self.audit_block() if self.manifest else {
                "manifest_path": self._rel(self.manifest_path),
                "methodological_note": METHODOLOGICAL_NOTE,
            },
            "notes": [note],
        }

    def evaluate(self, embedding: list[float] | None = None, patch_id: str | None = None,
                 requested_region: str | None = None) -> dict[str, Any]:
        """Executa a governança sobre um embedding (ou sobre o patch do
        corpus indicado). Nunca levanta exceção por falta de dado: devolve
        um estado explícito.

        `requested_region` é a região que a API resolveu pela geometria; a
        comparação com `suggested_region` é OBSERVAÇÃO estrutural, não
        veredito -- ver `diagnostics.territorial_concordance` no manifesto,
        que mede quanto essa concordância vale no próprio corpus.
        """
        req_region = normalize_region(requested_region) if requested_region else None

        if not self.available:
            return self._unavailable(requested_region, self.unavailable_reason or "governanca_indisponivel")

        resolved_patch = (patch_id or "").strip().upper() or None
        vec = embedding
        if vec is None and resolved_patch:
            vec = self.embedding_for_patch(resolved_patch)
            if vec is None:
                out = self._empty(req_region)
                out["status"] = STATUS_NO_VISUAL_EVIDENCE
                out["query_patch_id"] = resolved_patch
                out["notes"] = [
                    f"patch_id '{resolved_patch}' nao esta no corpus DINOv2 valido "
                    f"({self._rel(self.corpus_path)}) -- sem embedding real para governar."]
                return out
        if vec is None:
            out = self._empty(req_region)
            out["status"] = STATUS_NO_VISUAL_EVIDENCE
            out["notes"] = [
                "nenhum embedding visual pôde ser resolvido para esta requisição "
                "(sem patch_id no corpus e sem vetor fornecido) -- score fisico nao e afetado."]
            return out

        ok, reason = validate_embedding(vec)
        if not ok:
            out = self._empty(req_region)
            out["status"] = STATUS_INVALID_EMBEDDING
            out["query_patch_id"] = resolved_patch
            out["notes"] = [f"embedding rejeitado pela governanca: {reason}"]
            return out

        ranking = self.similarity_ranking(list(vec))
        top = ranking[0]
        threshold = self.ood_threshold
        in_domain = threshold is None or top["cosine_similarity"] >= threshold

        notes: list[str] = []
        territorial = TERRITORIAL_NA
        if req_region and req_region != "UNKNOWN":
            territorial = TERRITORIAL_MATCH if top["region"] == req_region else TERRITORIAL_MISMATCH
            if territorial == TERRITORIAL_MISMATCH:
                notes.append(
                    f"incompatibilidade territorial: regiao solicitada={req_region}, "
                    f"regiao visualmente mais proxima={top['region']} "
                    f"(medoid {top['medoid_patch_id']}, cos={top['cosine_similarity']:.4f}). "
                    "Observacao estrutural para revisao -- nao invalida o score fisico.")
        if not in_domain:
            notes.append(
                f"gate OOD acionado: similaridade ao medoid mais proximo "
                f"({top['cosine_similarity']:.4f}) abaixo do limiar {threshold:.4f}. "
                "O score fisico continua valido e inalterado; a evidencia visual e "
                "que fica fora do dominio observado do corpus DINOv2.")
        if threshold is None:
            notes.append("limiar OOD nao configurado no manifesto -- gate nao avaliado.")

        return {
            "status": STATUS_IN_DOMAIN if in_domain else STATUS_OUT_OF_DOMAIN,
            "query_patch_id": resolved_patch,
            "cosine_similarity": top["cosine_similarity"],
            "nearest_medoid_patch_id": top["medoid_patch_id"],
            "suggested_region": top["region"],
            "requested_region": req_region,
            "territorial_match": territorial,
            "ood_threshold": threshold,
            "ranking": ranking,
            "audit": self.audit_block(),
            "notes": notes,
        }

    def _empty(self, req_region: str | None) -> dict[str, Any]:
        return {
            "status": STATUS_NO_VISUAL_EVIDENCE,
            "query_patch_id": None,
            "cosine_similarity": None,
            "nearest_medoid_patch_id": None,
            "suggested_region": None,
            "requested_region": req_region,
            "territorial_match": TERRITORIAL_NA,
            "ood_threshold": self.ood_threshold,
            "ranking": [],
            "audit": self.audit_block(),
            "notes": [],
        }


__all__ = [
    "Dinov2GovernanceEngine", "MANIFEST_PATH", "CORPUS_PATH", "MANIFEST_VERSION",
    "STATUS_IN_DOMAIN", "STATUS_OUT_OF_DOMAIN", "STATUS_NO_VISUAL_EVIDENCE",
    "STATUS_INVALID_EMBEDDING", "STATUS_UNAVAILABLE",
    "TERRITORIAL_MATCH", "TERRITORIAL_MISMATCH", "TERRITORIAL_NA",
    "cosine_similarity", "validate_embedding", "parse_embedding_row",
    "normalize_region", "region_of_patch", "METHODOLOGICAL_NOTE",
]

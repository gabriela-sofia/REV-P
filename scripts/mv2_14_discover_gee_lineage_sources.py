"""MV2-14 Tarefa 1 — Descoberta de fontes GEE/export de lineage.

Varre a working tree e worktrees read-only por arquivos leves candidatos a conter
lineage GEE/Sentinel/export/scene. Não abre binário pesado. Indexa apenas arquivos
leves e pontua relevância.

Saída: outputs_public/mv2_gee_lineage_recovery/mv2_14_gee_source_discovery.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_14_gee_lineage_common import (
    ALLOWED_SCAN_EXT,
    ASSET_TERMS,
    EXPORT_TERMS,
    GEE_TERMS,
    OUTPUT_DIR,
    PATCH_TERMS,
    PROJECT_ROOT,
    S2_SCENE_RE,
    SENTINEL_TERMS,
    ensure_output_dir,
    is_excluded,
    iter_scannable_files,
    rel_to_root,
    resolve_scan_roots,
    write_csv,
)

COLUMNS = [
    "source_id", "source_path_or_ref", "source_origin", "source_type",
    "file_extension", "size_bytes", "contains_gee_terms", "contains_sentinel_terms",
    "contains_scene_id_like_terms", "contains_export_terms", "contains_asset_id_terms",
    "contains_patch_id_terms", "read_status", "relevance_score", "notes",
]

MAX_BYTES_READ = 262_144  # 256KB: termos GEE/scene aparecem no início de fontes pequenas
HARD_SKIP_BYTES = 5_000_000  # acima disto, não abre (ex.: registries de centenas de MB)
MAX_FILES = 8000


def _read_head(path: Path, limit: int) -> str:
    """Lê no máximo `limit` bytes via streaming, sem carregar o arquivo inteiro."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def _source_type(ext: str) -> str:
    return {
        ".py": "script", ".js": "script", ".ipynb": "notebook",
        ".json": "metadata", ".csv": "table", ".md": "doc", ".txt": "doc",
        ".yml": "config", ".yaml": "config",
    }.get(ext, "other")


def discover() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    idx = 0
    count = 0
    for root, origin in resolve_scan_roots():
        for path in iter_scannable_files(root):
            if count >= MAX_FILES:
                break
            if not path.is_file() or is_excluded(path):
                continue
            ext = path.suffix.lower()
            if ext not in ALLOWED_SCAN_EXT:
                continue
            ref = f"{origin}|{rel_to_root(path)}"
            if ref in seen:
                continue
            seen.add(ref)
            count += 1
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            if size > HARD_SKIP_BYTES:
                # Arquivos enormes (ex.: registries de negativos de centenas de MB)
                # não são fontes de export GEE; não abrir.
                continue
            txt = _read_head(path, MAX_BYTES_READ)
            read_status = "READ" if size <= MAX_BYTES_READ else "READ_HEAD_ONLY"

            has_gee = bool(GEE_TERMS.search(txt))
            has_sent = bool(SENTINEL_TERMS.search(txt))
            has_scene = bool(S2_SCENE_RE.search(txt))
            has_export = bool(EXPORT_TERMS.search(txt))
            has_asset = bool(ASSET_TERMS.search(txt))
            has_patch = bool(PATCH_TERMS.search(txt))

            score = sum([
                3 * has_scene, 2 * has_gee, 1 * has_sent, 1 * has_export,
                1 * has_asset, 1 * has_patch,
            ])
            if score == 0:
                continue  # só indexa fontes com algum sinal relevante

            idx += 1
            rows.append({
                "source_id": f"MV2_14_SRC_{idx:04d}",
                "source_path_or_ref": rel_to_root(path),
                "source_origin": origin,
                "source_type": _source_type(ext),
                "file_extension": ext,
                "size_bytes": size,
                "contains_gee_terms": str(has_gee).lower(),
                "contains_sentinel_terms": str(has_sent).lower(),
                "contains_scene_id_like_terms": str(has_scene).lower(),
                "contains_export_terms": str(has_export).lower(),
                "contains_asset_id_terms": str(has_asset).lower(),
                "contains_patch_id_terms": str(has_patch).lower(),
                "read_status": read_status,
                "relevance_score": score,
                "notes": "fonte leve indexada; bruto nunca aberto",
            })
    rows.sort(key=lambda r: (-int(str(r["relevance_score"])), str(r["source_path_or_ref"])))
    return rows


def main() -> None:
    ensure_output_dir()
    rows = discover()
    out = OUTPUT_DIR / "mv2_14_gee_source_discovery.csv"
    write_csv(out, COLUMNS, rows)
    scene = sum(1 for r in rows if r["contains_scene_id_like_terms"] == "true")
    gee = sum(1 for r in rows if r["contains_gee_terms"] == "true")
    print(f"[mv2_14_discovery] fontes relevantes: {len(rows)} | com scene_id: {scene} | com GEE: {gee}")
    print(f"[mv2_14_discovery] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

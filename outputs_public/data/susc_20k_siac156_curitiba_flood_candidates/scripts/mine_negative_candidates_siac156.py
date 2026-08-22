"""Minera candidatos a ponto negativo do SIAC 156 (Curitiba) -- espelha"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from negative_categories_curitiba import is_curitiba_negative_category  # noqa: E402


def normalize(text: str | None) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return ascii_only.upper().strip()


def stable_rank(*parts: str) -> int:
    """Hash estável -- usado pra ordenação determinística sem depender de seed aleatória."""
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def load_positive_bairros(positivos_path: Path | str) -> set[str]:
    with open(positivos_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {normalize(r["bairro"]) for r in rows}


def mine_year_file(csv_path: Path | str, positive_bairros: set[str], max_por_ano: int | None) -> list[dict]:
    source_year = None
    candidates = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            assunto = row.get("Assunto")
            subdivisao = row.get("Subdivisao")
            if not is_curitiba_negative_category(assunto, subdivisao):
                continue
            bairro = row.get("Bairro") or ""
            if normalize(bairro) not in positive_bairros:
                continue  # condicao 5: pareamento geografico
            logradouro = row.get("Logradouro") or ""
            data_criacao = row.get("DataCriacao") or ""
            if not logradouro.strip() or not bairro.strip():
                continue
            if source_year is None:
                source_year = data_criacao[-4:] if len(data_criacao) >= 4 else ""
            candidates.append(
                {
                    "source_year": source_year,
                    "data_criacao": data_criacao,
                    "assunto": assunto,
                    "subdivisao": subdivisao,
                    "situacao": row.get("Situacao"),
                    "logradouro": logradouro,
                    "bairro": bairro,
                    "regional": row.get("Regional"),
                    "origem": row.get("Origem"),
                    # SUSC-20N corrigiu um bug de reprodutibilidade: a versao anterior usava
                    # str(csv_path) aqui, entao a amostra dependia do caminho passado na linha
                    # de comando (mesmo arquivo, diretorio diferente = amostra diferente). Chave
                    # agora e so o conteudo do registro (source_year, nao o path), estavel entre
                    # maquinas e sessoes.
                    "rank_key": stable_rank(source_year, logradouro, bairro, data_criacao),
                }
            )
    candidates.sort(key=lambda r: r["rank_key"])
    if max_por_ano is not None:
        candidates = candidates[:max_por_ano]
    return candidates


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--positivos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-por-ano", type=int, default=None)
    p.add_argument("--summary-out")
    args = p.parse_args(argv)

    positive_bairros = load_positive_bairros(args.positivos)

    all_rows = []
    for csv_path in args.csv:
        rows = mine_year_file(csv_path, positive_bairros, args.max_por_ano)
        all_rows.extend(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_year", "data_criacao", "assunto", "subdivisao", "situacao", "logradouro", "bairro", "regional", "origem", "rank_key"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    from collections import Counter
    by_year = Counter(r["source_year"] for r in all_rows)
    summary = {"total": len(all_rows), "por_ano": dict(sorted(by_year.items())), "bairros_positivos_usados_no_pareamento": len(positive_bairros)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

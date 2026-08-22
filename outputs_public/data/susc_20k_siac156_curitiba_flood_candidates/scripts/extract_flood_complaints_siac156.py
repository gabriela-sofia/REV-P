"""Extrai reclamações de enchente/alagamento do SIAC 156 (Curitiba) pra uma janela de evento."""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

FLOOD_KEYWORDS = ("alagam", "enchente", "enxurrada", "transbord", "inund")


def normalize(text: str | None) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return ascii_only.upper().strip()


def is_flood_related(assunto: str | None, subdivisao: str | None) -> bool:
    hay = normalize(assunto) + " " + normalize(subdivisao)
    hay_lower = hay.lower()
    return any(kw in hay_lower for kw in FLOOD_KEYWORDS)


@dataclass
class EventWindow:
    window_id: str
    event_date_ddmmyyyy: str  # formato do campo DataCriacao no CSV: DD/MM/AAAA
    expected_bairros: set[str] = field(default_factory=set)  # já normalizados


def load_events(path: Path | str) -> list[EventWindow]:
    events = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            bairros = {
                normalize(b) for b in (row.get("bairros") or "").split(";") if b.strip()
            }
            events.append(
                EventWindow(
                    window_id=row["window_id"],
                    event_date_ddmmyyyy=row["event_date_ddmmyyyy"].strip(),
                    expected_bairros=bairros,
                )
            )
    return events


def extract_candidates(csv_paths: list[Path | str], events: list[EventWindow]) -> list[dict]:
    by_date = {ev.event_date_ddmmyyyy: ev for ev in events}
    seen = set()  # dedupe entre arquivos sobrepostos (extratos acumulados se repetem)
    out: list[dict] = []

    for csv_path in csv_paths:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                data_criacao = (row.get("DataCriacao") or "").strip()
                ev = by_date.get(data_criacao)
                if ev is None:
                    continue
                assunto = row.get("Assunto")
                subdivisao = row.get("Subdivisao")
                if not is_flood_related(assunto, subdivisao):
                    continue

                bairro = row.get("Bairro") or ""
                logradouro = row.get("Logradouro") or ""
                dedupe_key = (ev.window_id, normalize(bairro), normalize(logradouro), data_criacao, normalize(assunto))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                out.append(
                    {
                        "window_id": ev.window_id,
                        "data_criacao": data_criacao,
                        "tipo": row.get("Tipo"),
                        "orgao": row.get("Orgao"),
                        "assunto": assunto,
                        "subdivisao": subdivisao,
                        "situacao": row.get("Situacao"),
                        "logradouro": logradouro,
                        "bairro": bairro,
                        "regional": row.get("Regional"),
                        "origem": row.get("Origem"),
                        "bairro_esperado": normalize(bairro) in ev.expected_bairros
                        if ev.expected_bairros
                        else None,
                        "source_file": str(csv_path),
                    }
                )
    return out


def write_candidates_csv(rows: list[dict], out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_id", "data_criacao", "tipo", "orgao", "assunto", "subdivisao", "situacao",
        "logradouro", "bairro", "regional", "origem", "bairro_esperado", "source_file",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True, help="arquivos SIAC 156 (';'-delimitado)")
    p.add_argument("--events", required=True, help="CSV de janelas de evento")
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out")
    args = p.parse_args(argv)

    events = load_events(args.events)
    rows = extract_candidates(args.csv, events)
    write_candidates_csv(rows, args.out)

    summary = {"n_events": len(events), "n_candidates": len(rows)}
    by_window: dict[str, int] = {}
    for r in rows:
        by_window[r["window_id"]] = by_window.get(r["window_id"], 0) + 1
    summary["by_window"] = by_window
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""SUSC-15B step 2: document non-obvious precision source discovery."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import susc_15b_forensic_common as c  # noqa: E402


def main() -> int:
    rows = c.read_csv(c.HIDDEN_AUDIT)
    id_candidates = sum(1 for row in rows if row.get("field_kind") == "identifier_like" and int(row.get("non_empty_rows") or 0) > 0)
    coord_candidates = sum(1 for row in rows if row.get("precision_candidate") == "true")
    summary = c.summary_15a()
    c.write_markdown(c.DISCOVERY_REPORT, f"""# SUSC-15B - descoberta de camadas/eventos nao obvios

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

{c.MANDATORY}

## Fontes atacadas
- IDs/codigos internos em CSVs de ocorrencia: {id_candidates} campos com valor.
- Colunas escondidas de coordenada/UTM: {coord_candidates} campos coordinate-like com valor ja auditados ou bloqueados.
- Tabelas oficiais auxiliares: dependem de correspondencia manual por identificador.
- PDFs/anexos de pontos de alagamento: nao foram materializados em arquivo publico novo.
- ArcGIS/WFS/FeatureServer com nomes nao obvios: mantidos como alvo de aquisicao manual, sem download automatico nesta execucao.
- Intersecoes oficiais, numeracao predial e footprints Copernicus/EMS/SGB/INEA/APAC: requerem evidencia oficial precisa antes de T0-T4.

## Estado herdado do SUSC-15A
- Eventos avaliados: {summary['precision_events_total']}
- Eventos elegiveis para calibracao: {summary['calibration_eligible_events']}
- Links patch-evento precisos: {summary['precision_patch_links']}
- Patches observacionais precisos: {summary['unique_patches_observational']}

## Decisao
SUSC-15B permanece bloqueado para calibracao automatica ate que uma fonte oficial forneca ponto, poligono, lote, intersecao ou faixa numerica controlada. Bairro-only e segmento de rua sem numero/intersecao continuam excluidos.
""")
    print(f"discovery report written: {c.DISCOVERY_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

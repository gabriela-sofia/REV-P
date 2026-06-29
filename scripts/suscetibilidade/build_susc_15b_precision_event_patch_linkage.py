"""SUSC-15B step 7: build fail-closed precision event-patch linkage."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import susc_15b_forensic_common as c  # noqa: E402

FIELDS = [
    "linkage_id", "event_id", "patch_id", "region", "event_date", "precision_tier",
    "spatial_relation", "linkage_confidence", "can_be_used_for_observational_evaluation",
    "can_be_ground_truth", "can_be_used_as_ground_truth", "allowed_for_training",
    "review_only", "requires_manual_review", "interpretation", "limitations",
]


def main() -> int:
    rows = []
    for idx, event in enumerate(c.read_csv(c.CATALOG)):
        eligible = event.get("calibration_eligible") == "true"
        rows.append({
            "linkage_id": f"S15BLINK_{idx:06d}",
            "event_id": event.get("event_id", ""),
            "patch_id": "REGION_LEVEL_NO_PATCH_RESOLUTION",
            "region": event.get("region", ""),
            "event_date": event.get("event_date", ""),
            "precision_tier": event.get("precision_tier", ""),
            "spatial_relation": "blocked_no_new_precise_patch_link",
            "linkage_confidence": "very_weak" if not eligible else "manual_review_required",
            "can_be_used_for_observational_evaluation": "false",
            "can_be_ground_truth": "false",
            "can_be_used_as_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "requires_manual_review": "true",
            "interpretation": "SUSC-15B nao materializou novo vinculo patch-evento preciso.",
            "limitations": "Sem ponto/poligono/lote/intersecao/faixa numerica oficial adicional.",
        })
    c.write_csv(c.LINKAGE, rows, FIELDS)
    print(f"15B linkage rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""MV2 DATA-08 metadata-only replay and response normalisation.

Loads lightweight, fictional/empty public fixtures and normalises provider
responses (GEE / CDSE STAC / CDSE OData / Traceability) into the canonical
:class:`MetadataProviderResult` contract. Default behaviour is offline: no
network is ever touched here. Real responses, if ever captured, must be redacted
to light metadata before being published; tokens, signed URLs, local paths,
rasters and heavy payloads are stripped by :func:`redact_sensitive_fields`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mv2_data_08_metadata_provider_contracts import (
    MetadataProviderResult,
    MetadataQueryTarget,
)

REPLAY_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_metadata_only_replay"
FIXTURE_DIR = REPLAY_DIR / "fixtures"

FIXTURE_NAMES = {
    "GEE": "gee_empty_result.json",
    "CDSE_STAC": "stac_empty_result.json",
    "CDSE_ODATA": "odata_empty_result.json",
    "TRACEABILITY": "traceability_empty_result.json",
}

# Field names that must never reach a public artifact.
SENSITIVE_KEYS = {
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "bearer",
    "signed_url",
    "signedurl",
    "presigned_url",
    "secret",
    "secret_key",
    "access_key",
    "aws_secret_access_key",
    "password",
    "credential",
    "credentials",
    "local_path",
    "raster_path",
    "crop_path",
    "download_url",
    "href_download",
}


def hash_raw_response(raw: Any) -> str:
    """Stable sha256 of a raw response (canonical JSON)."""
    payload = json.dumps(raw, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def redact_sensitive_fields(value: Any) -> Any:
    """Recursively drop secrets / signed URLs / local paths / raster pointers."""
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).strip().lower() in SENSITIVE_KEYS:
                continue
            clean[key] = redact_sensitive_fields(item)
        return clean
    if isinstance(value, list):
        return [redact_sensitive_fields(item) for item in value]
    if isinstance(value, str) and value.lower().startswith(("http://", "https://")) and (
        "x-amz-" in value.lower() or "sig=" in value.lower() or "token=" in value.lower()
    ):
        return "<redacted_signed_url>"
    return value


def load_replay_fixture(provider: str, fixture_dir: Path = FIXTURE_DIR) -> dict[str, Any]:
    """Load a public empty fixture for a provider. Missing => empty payload."""
    name = FIXTURE_NAMES.get(provider)
    if not name:
        return {}
    path = fixture_dir / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _classify_item_status(product_id: str, datetime_utc: str, has_geometry: bool, has_tile: bool) -> str:
    if product_id and datetime_utc and has_geometry:
        return "MATCH_STRONG"
    if datetime_utc and (has_geometry or has_tile):
        return "MATCH_MEDIUM_REVIEW"
    return "MATCH_WEAK"


def _result(
    target: MetadataQueryTarget,
    provider: str,
    query_mode: str,
    raw_hash: str,
    *,
    collection: Any = "",
    scene_id: Any = "",
    product_id: Any = "",
    datetime_utc: Any = "",
    mgrs_tile: Any = "",
    datatake_identifier: Any = "",
    generation_time: Any = "",
    cloudy: Any = "",
    nodata: Any = "",
    cirrus: Any = "",
    geometry: Any = "",
    bbox: Any = "",
    odata_id: Any = "",
    odata_name: Any = "",
    odata_geofootprint: Any = "",
) -> MetadataProviderResult:
    has_geometry = bool(geometry) or bool(bbox) or bool(odata_geofootprint)
    status = _classify_item_status(product_id, datetime_utc, has_geometry, bool(mgrs_tile))
    return MetadataProviderResult(
        patch_id=target.patch_id,
        asset_id=target.asset_id,
        provider=provider,
        status=status,
        query_mode=query_mode,
        collection=collection or target.collection,
        scene_id=str(scene_id or ""),
        product_id=str(product_id or ""),
        datetime_utc=str(datetime_utc or ""),
        mgrs_tile=str(mgrs_tile or target.mgrs_tile or ""),
        datatake_identifier=str(datatake_identifier or ""),
        generation_time=str(generation_time or ""),
        cloudy_pixel_percentage=str(cloudy if cloudy != "" else ""),
        nodata_pixel_percentage=str(nodata if nodata != "" else ""),
        thin_cirrus_percentage=str(cirrus if cirrus != "" else ""),
        geometry=geometry or "",
        bbox=bbox or "",
        odata_id=str(odata_id or ""),
        odata_name=str(odata_name or ""),
        odata_geofootprint=odata_geofootprint or "",
        odata_s3path="",  # s3 path is never published; kept blank by policy
        source_response_hash=raw_hash,
    )


def normalize_gee_response(
    raw: dict[str, Any], target: MetadataQueryTarget, query_mode: str = "REPLAY"
) -> list[MetadataProviderResult]:
    raw_hash = hash_raw_response(raw)
    items = (raw or {}).get("features") or (raw or {}).get("items") or []
    results: list[MetadataProviderResult] = []
    for item in items:
        props = item.get("properties", {}) if isinstance(item, dict) else {}
        results.append(
            _result(
                target,
                "GEE",
                query_mode,
                raw_hash,
                collection=props.get("collection") or item.get("collection", ""),
                scene_id=item.get("id", ""),
                product_id=props.get("PRODUCT_ID") or props.get("product_id", ""),
                datetime_utc=props.get("datetime") or props.get("system:time_start", ""),
                mgrs_tile=props.get("MGRS_TILE") or props.get("mgrs_tile", ""),
                datatake_identifier=props.get("DATATAKE_IDENTIFIER", ""),
                generation_time=props.get("GENERATION_TIME", ""),
                cloudy=props.get("CLOUDY_PIXEL_PERCENTAGE", ""),
                nodata=props.get("NODATA_PIXEL_PERCENTAGE", ""),
                cirrus=props.get("THIN_CIRRUS_PERCENTAGE", ""),
                geometry=item.get("geometry", ""),
            )
        )
    return results


def normalize_stac_response(
    raw: dict[str, Any], target: MetadataQueryTarget, query_mode: str = "REPLAY"
) -> list[MetadataProviderResult]:
    raw_hash = hash_raw_response(raw)
    items = (raw or {}).get("features") or []
    results: list[MetadataProviderResult] = []
    for item in items:
        props = item.get("properties", {}) if isinstance(item, dict) else {}
        results.append(
            _result(
                target,
                "CDSE_STAC",
                query_mode,
                raw_hash,
                collection=item.get("collection", ""),
                scene_id=item.get("id", ""),
                product_id=props.get("s2:product_uri") or props.get("product_id", ""),
                datetime_utc=props.get("datetime", ""),
                mgrs_tile=props.get("grid:code") or props.get("s2:mgrs_tile", ""),
                datatake_identifier=props.get("s2:datatake_id", ""),
                generation_time=props.get("s2:generation_time", ""),
                cloudy=props.get("eo:cloud_cover", ""),
                nodata=props.get("s2:nodata_pixel_percentage", ""),
                cirrus=props.get("s2:thin_cirrus_percentage", ""),
                geometry=item.get("geometry", ""),
                bbox=item.get("bbox", ""),
            )
        )
    return results


def normalize_odata_response(
    raw: dict[str, Any], target: MetadataQueryTarget, query_mode: str = "REPLAY"
) -> list[MetadataProviderResult]:
    raw_hash = hash_raw_response(raw)
    items = (raw or {}).get("value") or []
    results: list[MetadataProviderResult] = []
    for item in items:
        attrs = {a.get("Name"): a.get("Value") for a in item.get("Attributes", []) if isinstance(a, dict)}
        results.append(
            _result(
                target,
                "CDSE_ODATA",
                query_mode,
                raw_hash,
                collection=item.get("Collection") or attrs.get("instrumentShortName", ""),
                product_id=attrs.get("PRODUCT_URI") or item.get("Name", ""),
                datetime_utc=item.get("ContentDate", {}).get("Start", "") if isinstance(item.get("ContentDate"), dict) else "",
                mgrs_tile=attrs.get("tileId", ""),
                datatake_identifier=attrs.get("datatakeIdentifier", ""),
                generation_time=attrs.get("processingDate", ""),
                cloudy=attrs.get("cloudCover", ""),
                geometry="",
                odata_id=item.get("Id", ""),
                odata_name=item.get("Name", ""),
                odata_geofootprint=item.get("GeoFootprint", ""),
            )
        )
    return results


def normalize_traceability_response(
    raw: dict[str, Any], target: MetadataQueryTarget, query_mode: str = "REPLAY"
) -> list[MetadataProviderResult]:
    raw_hash = hash_raw_response(raw)
    items = (raw or {}).get("items") or (raw or {}).get("value") or []
    results: list[MetadataProviderResult] = []
    for item in items:
        results.append(
            _result(
                target,
                "TRACEABILITY",
                query_mode,
                raw_hash,
                product_id=item.get("product_id") or item.get("productName", ""),
                datetime_utc=item.get("event_date") or item.get("eventDate", ""),
                datatake_identifier=item.get("datatake_id", ""),
                generation_time=item.get("processing_date", ""),
            )
        )
    return results


NORMALIZERS = {
    "GEE": normalize_gee_response,
    "CDSE_STAC": normalize_stac_response,
    "CDSE_ODATA": normalize_odata_response,
    "TRACEABILITY": normalize_traceability_response,
}


def normalize_response(
    provider: str, raw: dict[str, Any], target: MetadataQueryTarget, query_mode: str = "REPLAY"
) -> list[MetadataProviderResult]:
    normalizer = NORMALIZERS.get(provider)
    if normalizer is None:
        return []
    return normalizer(raw, target, query_mode)


def write_fixtures(fixture_dir: Path = FIXTURE_DIR) -> None:
    """Materialise empty, fictional public fixtures (idempotent)."""
    fixture_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "gee_empty_result.json": {"provider": "GEE", "features": []},
        "stac_empty_result.json": {"type": "FeatureCollection", "features": []},
        "odata_empty_result.json": {"value": []},
        "traceability_empty_result.json": {"provider": "TRACEABILITY", "items": []},
    }
    for name, payload in payloads.items():
        (fixture_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )


def write_readme(replay_dir: Path = REPLAY_DIR) -> None:
    replay_dir.mkdir(parents=True, exist_ok=True)
    (replay_dir / "README_REPLAY_METADATA_ONLY_PTBR.md").write_text(
        """# Replay metadata-only Sentinel-2

Esta pasta guarda fixtures publicas e leves usadas para exercitar o motor de
metadados sem tocar a rede. Por padrao o motor roda offline (`--replay-only`):
ele le estas fixtures vazias e nao executa nenhuma chamada real.

## Conteudo
- `fixtures/gee_empty_result.json`
- `fixtures/stac_empty_result.json`
- `fixtures/odata_empty_result.json`
- `fixtures/traceability_empty_result.json`

Todas sao ficticias e vazias (`features: []` / `value: []` / `items: []`).

## Regras de publicacao
- fixtures publicas sao sempre ficticias ou vazias;
- se no futuro houver resposta real, publicar somente campos leves e redigidos
  (`redact_sensitive_fields`);
- nunca publicar token, URL assinada, path local, raster ou payload pesado;
- `odata_s3path` e mantido em branco por politica.

## Como o replay e usado
1. `load_replay_fixture(provider)` carrega a fixture vazia;
2. `normalize_*_response(raw, target)` converte para o contrato canonico;
3. fixture vazia produz `NO_MATCH` (chamada simulada, nenhum item);
4. `hash_raw_response` registra um hash estavel da resposta bruta.
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    write_fixtures()
    write_readme()
    print("[mv2_data_08_metadata_replay] fixtures vazias e README publicados (offline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

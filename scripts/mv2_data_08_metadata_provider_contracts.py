"""MV2 DATA-08 metadata-only provider contracts.

Pure, dependency-free contracts for the metadata-only Sentinel-2 lineage engine.
These dataclasses describe *only* metadata: scene identifiers, product ids,
acquisition datetimes, MGRS tiles, cloud/nodata statistics and lightweight
geometry. No contract here references, requires or produces a raster, a crop, a
download or any private file. Everything is JSON/CSV serialisable so it can be
published as a redacted public artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

PROVIDERS = ["GEE", "CDSE_STAC", "CDSE_ODATA", "TRACEABILITY"]
OFFICIAL_PROVIDERS = set(PROVIDERS)
SENTINEL_2 = "SENTINEL_2"

# Query mode is metadata-only by construction; raster/crop modes are forbidden.
QUERY_MODES = ["NO_CALL", "REPLAY", "METADATA_ONLY"]

# Allowed result statuses (provider-level).
RESULT_STATUSES = [
    "NO_CALL",
    "BLOCKED_NO_CONFIG",
    "BLOCKED_BY_FLAGS",
    "BLOCKED_NO_TEMPORAL_WINDOW",
    "BLOCKED_NO_SENSOR_LINEAGE",
    "BLOCKED_NO_AOI",
    "QUERY_READY",
    "QUERY_FAILED",
    "NO_MATCH",
    "MATCH_WEAK",
    "MATCH_MEDIUM_REVIEW",
    "MATCH_STRONG",
    "CONFLICT",
]

# Statuses that mean "no provider call was actually executed".
NON_CALL_STATUSES = {
    "NO_CALL",
    "BLOCKED_NO_CONFIG",
    "BLOCKED_BY_FLAGS",
    "BLOCKED_NO_TEMPORAL_WINDOW",
    "BLOCKED_NO_SENSOR_LINEAGE",
    "BLOCKED_NO_AOI",
}

# Statuses that carry a confirmed product match.
MATCH_STATUSES = {"MATCH_WEAK", "MATCH_MEDIUM_REVIEW", "MATCH_STRONG"}

# Canonical, ordered field list for a serialised provider result.
RESULT_FIELDS = [
    "patch_id",
    "asset_id",
    "provider",
    "collection",
    "query_mode",
    "scene_id",
    "product_id",
    "datetime_utc",
    "mgrs_tile",
    "datatake_identifier",
    "generation_time",
    "cloudy_pixel_percentage",
    "nodata_pixel_percentage",
    "thin_cirrus_percentage",
    "geometry",
    "bbox",
    "odata_id",
    "odata_name",
    "odata_geofootprint",
    "odata_s3path",
    "status",
    "blocked_reason",
    "source_response_hash",
]


def _parse_iso(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "").strip())
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class MetadataQueryTarget:
    """Inputs describing *what* metadata to look for. Never carries raster."""

    patch_id: str
    asset_id: str
    sensor_family: str = ""
    temporal_window_start: str = ""
    temporal_window_end: str = ""
    aoi_wgs84: Any = None
    mgrs_tile: str = ""
    collection: str = ""
    source_ref: str = ""

    def has_valid_temporal_window(self) -> bool:
        start = _parse_iso(self.temporal_window_start)
        end = _parse_iso(self.temporal_window_end)
        return bool(start and end and start <= end)

    def is_sentinel_2(self) -> bool:
        return (self.sensor_family or "").strip().upper() == SENTINEL_2

    def has_valid_aoi(self) -> bool:
        aoi = self.aoi_wgs84
        if isinstance(aoi, dict):
            if aoi.get("type") and aoi.get("coordinates"):
                return True
            bbox = aoi.get("bbox")
            return isinstance(bbox, (list, tuple)) and len(bbox) == 4
        if isinstance(aoi, (list, tuple)):
            return len(aoi) == 4 and all(isinstance(v, (int, float)) for v in aoi)
        return False

    def is_query_ready(self) -> bool:
        return self.has_valid_temporal_window() and self.is_sentinel_2() and self.has_valid_aoi()

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "asset_id": self.asset_id,
            "sensor_family": self.sensor_family,
            "temporal_window_start": self.temporal_window_start,
            "temporal_window_end": self.temporal_window_end,
            "aoi_wgs84": self.aoi_wgs84,
            "mgrs_tile": self.mgrs_tile,
            "collection": self.collection,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class MetadataProviderResult:
    """One provider's metadata answer for one target. Metadata-only."""

    patch_id: str
    asset_id: str
    provider: str
    status: str = "NO_CALL"
    query_mode: str = "NO_CALL"
    collection: str = ""
    scene_id: str = ""
    product_id: str = ""
    datetime_utc: str = ""
    mgrs_tile: str = ""
    datatake_identifier: str = ""
    generation_time: str = ""
    cloudy_pixel_percentage: str = ""
    nodata_pixel_percentage: str = ""
    thin_cirrus_percentage: str = ""
    geometry: Any = ""
    bbox: Any = ""
    odata_id: str = ""
    odata_name: str = ""
    odata_geofootprint: Any = ""
    odata_s3path: str = ""
    blocked_reason: str = ""
    source_response_hash: str = ""

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"invalid result status: {self.status}")
        if self.query_mode not in QUERY_MODES:
            raise ValueError(f"invalid query mode: {self.query_mode}")

    def is_call_executed(self) -> bool:
        return self.status not in NON_CALL_STATUSES

    def has_product_match(self) -> bool:
        return bool((self.product_id or "").strip()) and self.status in MATCH_STATUSES

    def to_row(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in RESULT_FIELDS}


@dataclass(frozen=True)
class MetadataProviderError:
    """A non-fatal provider error captured without raising."""

    patch_id: str
    asset_id: str
    provider: str
    error_kind: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "asset_id": self.asset_id,
            "provider": self.provider,
            "error_kind": self.error_kind,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class MetadataConsensusRecord:
    """Cross-provider lineage consensus for one target."""

    patch_id: str
    asset_id: str
    consensus_status: str
    product_id: str = ""
    datetime_utc: str = ""
    mgrs_tile: str = ""
    collection: str = ""
    providers_considered: list[str] = field(default_factory=list)
    providers_agreeing: list[str] = field(default_factory=list)
    conflict_reason: str = ""
    evidence_count: int = 0

    def to_row(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "asset_id": self.asset_id,
            "consensus_status": self.consensus_status,
            "product_id": self.product_id,
            "datetime_utc": self.datetime_utc,
            "mgrs_tile": self.mgrs_tile,
            "collection": self.collection,
            "providers_considered": ";".join(self.providers_considered),
            "providers_agreeing": ";".join(self.providers_agreeing),
            "conflict_reason": self.conflict_reason,
            "evidence_count": self.evidence_count,
        }


@dataclass(frozen=True)
class MetadataReplayRecord:
    """A replay/fixture trace: what would (or did) come back, redacted."""

    provider: str
    fixture_name: str
    query_mode: str
    item_count: int
    source_response_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "fixture_name": self.fixture_name,
            "query_mode": self.query_mode,
            "item_count": self.item_count,
            "source_response_hash": self.source_response_hash,
        }


def no_call_result(patch_id: str, asset_id: str, provider: str, blocked_reason: str = "") -> MetadataProviderResult:
    """Convenience constructor for the default fail-closed result."""
    return MetadataProviderResult(
        patch_id=patch_id,
        asset_id=asset_id,
        provider=provider,
        status="NO_CALL",
        query_mode="NO_CALL",
        blocked_reason=blocked_reason,
    )

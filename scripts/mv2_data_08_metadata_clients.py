"""MV2 DATA-08 mockable metadata-only provider clients.

Four clients mirror the providers we may eventually consult for Sentinel-2
lineage: Google Earth Engine, CDSE STAC, CDSE OData and CDSE Traceability. They
are *metadata-only* and *offline by default*:

  - ``live=False`` (default): no network, returns ``NO_CALL`` (or replay).
  - ``replay=True``: reads light public fixtures, never touches the network.
  - ``live=True``: only permitted when every gate passes (config present,
    metadata calls enabled, raster/canary downloads disabled, valid temporal
    window, Sentinel-2 lineage, valid AOI). Even then, no real transport is wired
    by default: a real call only happens if an explicit ``transport`` callable is
    injected. No download, export or raster access is implemented anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_data_08_metadata_replay as replay
from mv2_data_08_metadata_provider_contracts import (
    MetadataProviderResult,
    MetadataQueryTarget,
    no_call_result,
)

Transport = Callable[[MetadataQueryTarget], dict[str, Any]]
FixtureLoader = Callable[[str], dict[str, Any]]


def evaluate_call_gate(
    config: dict[str, Any] | None, target: MetadataQueryTarget, live: bool
) -> tuple[str, str]:
    """Return ``(status, reason)`` deciding whether a live call may happen."""
    if not live:
        return "NO_CALL", "live_disabled"
    if config is None:
        return "BLOCKED_NO_CONFIG", "no_config"
    if not config.get("allow_network") or not config.get("allow_metadata_calls"):
        return "BLOCKED_BY_FLAGS", "metadata_calls_disabled"
    if config.get("allow_raster_download") or config.get("allow_canary_download"):
        return "BLOCKED_BY_FLAGS", "download_flags_enabled"
    if not target.has_valid_temporal_window():
        return "BLOCKED_NO_TEMPORAL_WINDOW", "no_temporal_window"
    if not target.is_sentinel_2():
        return "BLOCKED_NO_SENSOR_LINEAGE", "not_sentinel_2"
    if not target.has_valid_aoi():
        return "BLOCKED_NO_AOI", "no_aoi"
    return "QUERY_READY", ""


class _BaseMetadataClient:
    """Shared metadata-only client behaviour. Subclasses set ``provider``."""

    provider = ""

    def __init__(
        self,
        live: bool = False,
        replay_mode: bool = False,
        config: dict[str, Any] | None = None,
        transport: Transport | None = None,
        fixture_loader: FixtureLoader | None = None,
    ) -> None:
        self.live = bool(live)
        self.replay_mode = bool(replay_mode)
        self.config = config
        self.transport = transport
        self.fixture_loader = fixture_loader or replay.load_replay_fixture
        self.call_count = 0

    def _normalize(self, raw: dict[str, Any], target: MetadataQueryTarget, query_mode: str) -> list[MetadataProviderResult]:
        return replay.normalize_response(self.provider, raw, target, query_mode)

    def _no_item_result(self, target: MetadataQueryTarget, query_mode: str, raw: dict[str, Any]) -> MetadataProviderResult:
        return MetadataProviderResult(
            patch_id=target.patch_id,
            asset_id=target.asset_id,
            provider=self.provider,
            status="NO_MATCH",
            query_mode=query_mode,
            source_response_hash=replay.hash_raw_response(raw),
        )

    def _replay(self, target: MetadataQueryTarget) -> list[MetadataProviderResult]:
        raw = self.fixture_loader(self.provider)
        results = self._normalize(raw, target, "REPLAY")
        return results or [self._no_item_result(target, "REPLAY", raw)]

    def query(self, target: MetadataQueryTarget) -> list[MetadataProviderResult]:
        # Replay short-circuits before any live consideration.
        if self.replay_mode and not self.live:
            return self._replay(target)

        status, reason = evaluate_call_gate(self.config, target, self.live)
        if status != "QUERY_READY":
            # Fail-closed: no call performed. Optionally fall back to replay.
            if self.replay_mode and status == "NO_CALL":
                return self._replay(target)
            return [no_call_result(target.patch_id, target.asset_id, self.provider, reason) if status == "NO_CALL"
                    else MetadataProviderResult(
                        patch_id=target.patch_id,
                        asset_id=target.asset_id,
                        provider=self.provider,
                        status=status,
                        query_mode="NO_CALL",
                        blocked_reason=reason,
                    )]

        # Gate passed. A real call only happens if a transport is injected.
        if self.transport is None:
            return [MetadataProviderResult(
                patch_id=target.patch_id,
                asset_id=target.asset_id,
                provider=self.provider,
                status="QUERY_READY",
                query_mode="METADATA_ONLY",
                blocked_reason="no_transport_wired",
            )]
        try:
            raw = self.transport(target)
            self.call_count += 1
        except Exception as exc:  # offline-safe: never raise out of a client
            return [MetadataProviderResult(
                patch_id=target.patch_id,
                asset_id=target.asset_id,
                provider=self.provider,
                status="QUERY_FAILED",
                query_mode="METADATA_ONLY",
                blocked_reason=f"transport_error:{type(exc).__name__}",
            )]
        results = self._normalize(raw, target, "METADATA_ONLY")
        return results or [self._no_item_result(target, "METADATA_ONLY", raw)]


class GeeMetadataClient(_BaseMetadataClient):
    provider = "GEE"


class CdseStacMetadataClient(_BaseMetadataClient):
    provider = "CDSE_STAC"


class CdseODataMetadataClient(_BaseMetadataClient):
    provider = "CDSE_ODATA"


class CdseTraceabilityClient(_BaseMetadataClient):
    provider = "TRACEABILITY"


CLIENT_CLASSES = {
    "GEE": GeeMetadataClient,
    "CDSE_STAC": CdseStacMetadataClient,
    "CDSE_ODATA": CdseODataMetadataClient,
    "TRACEABILITY": CdseTraceabilityClient,
}


def build_clients(
    providers: list[str],
    *,
    live: bool = False,
    replay_mode: bool = False,
    config: dict[str, Any] | None = None,
    transports: dict[str, Transport] | None = None,
    fixture_loader: FixtureLoader | None = None,
) -> dict[str, _BaseMetadataClient]:
    transports = transports or {}
    clients: dict[str, _BaseMetadataClient] = {}
    for provider in providers:
        cls = CLIENT_CLASSES.get(provider)
        if cls is None:
            continue
        clients[provider] = cls(
            live=live,
            replay_mode=replay_mode,
            config=config,
            transport=transports.get(provider),
            fixture_loader=fixture_loader,
        )
    return clients

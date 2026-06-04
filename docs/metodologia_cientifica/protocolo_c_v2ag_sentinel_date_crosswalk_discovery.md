# Protocolo C v2ag Sentinel Date Crosswalk Discovery

## Scope

v2ag scans versionable registries for explicit crosswalk evidence between event-patch package v2 IDs and Sentinel-dated anchor namespaces.

## Allowed Evidence

- Same-row event patch plus reference patch, anchor patch, scene id, or source hash.
- Documented lineage candidates are recorded without date application.
- Region-only, row-order, name-only, visual similarity, and date-only joins are rejected.

## Guardrails

- No overlay execution.
- No coordinate invention.
- No package v2 date update.
- No ground reference or label creation.

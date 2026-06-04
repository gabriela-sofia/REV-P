# Protocolo C v2aa - Sentinel Date Recovery for Event-Patch Packages

- registries scanned: `123` (parseable for dates: `23`)
- filename/scene-id dates extracted: `4`
- sidecar explicit-date resolutions: `15`
- patches consolidated: `8` (recovered: `7`, missing/blocked: `1`)
- usable (HIGH/MEDIUM) dates: `7`
- event-patch candidates with placed temporal distance: `0` of `172`
- readiness update rows: `1032`
- ground-reference blocker rows: `21`
- selected next target: `EVENT_PATCH_PACKAGE_SCHEMA_HARDENING`
- suggested next version: `v2ab — Event-Patch Package Schema Hardening`

v2aa recovered Sentinel scene dates only from existing filenames, scene ids and explicit sidecar date fields. It never downloaded data, queried the web, inferred a date, used an approximate date as real, used created_at/modified_at or file mtimes, executed overlay, or created ground truth, ground reference or labels.

# Protocolo C v1us - Event-Patch Package Linkage Engine

## Engineering Scope
- Builds auditable, non-operational event-patch packages for Recife, Petropolis and Curitiba.
- Uses only consolidated evidence and existing registries; no new download and no web search.
- Does not execute overlay, infer coordinates, geocode localities, or create ground truth, ground reference, or labels.

## Components
- patch registry resolver (real registries only; no invented patch_id or Sentinel date)
- event-patch candidate builder (region-only candidate linkage, no spatial distance)
- event temporal window linker (records temporal class; never invents Sentinel date)
- external evidence attacher (Recife locality-only; Petropolis document-only)
- phenomenon status attacher (contextual support, never a label)
- geometry blocker attacher (overlay and ground reference blocked)
- event-patch readiness matrix builder
- DINO review support attacher (review-only)
- next action ranker (programming value from real blockers)

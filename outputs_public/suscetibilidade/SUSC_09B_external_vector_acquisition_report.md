# SUSC-09B — Aquisição Vetorial Externa Oficial

> O SUSC-09B amplia a aquisição vetorial externa oficial review-only. Não cria ground truth, não desbloqueia treinamento e não transforma geometria candidata em footprint validado por patch.

## Registry e download
- Fontes oficiais no registry: ver `susc_09b_external_source_registry_v1.csv`.
- Tentativas de download: **13** — status: {'not_attempted_no_direct_url_or_no_network': 13}.
- Downloads bem-sucedidos: **0** (offline-safe; sem URL direta → not_attempted).

## Parsing e integração
- Geometrias externas parseadas: **0**.
- Candidatos de geometria de evento: **0**.
- Overlay candidates (bbox): **0**.

## Regras respeitadas
- Sem raster, sem imagem Sentinel, sem arquivo >100MB, sem API com chave, sem geocoding.
- Todo download registrado com URL, status, tamanho e SHA256 (quando houver).
- Tudo `requires_manual_review=true`, `can_be_ground_truth=false`, `review_only=true`.

## Limitações
Offline nesta passada: a aquisição oficial direta é manual. Geometrias candidatas, quando adquiridas, são review-only e exigem validação humana — nunca footprint validado por patch.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.

# Protocolo C — resumo de estado atual

**Recife** tem referência candidata validada pelo Protocolo C (pontuação 0.76, incerteza moderada). O evento de referência é `REC_2022_05_24_30`. O produto cartográfico de entrada é o raster Charter 758 de 2022-06-02 (cicatrizes de deslizamento), aceito como evidência de referência candidata. A limitação técnica é a ausência de vetor com CRS documentado. Evidência temporal de apoio: séries APAC mensais e estágios ANA do Rio Capibaribe. Lacuna local: série de precipitação da estação A301 vazia; dados Cemaden pendentes.

**Estado metodológico**: nenhum rótulo operacional, negativo formal ou amostra de treinamento foi criado. O gate C7 (rótulo operacional ou ground truth supervisionado final) permanece não criado e bloqueado para treinamento. Código interno: `C7 = NOT_CREATED_BLOCKED_FOR_TRAINING`.

A política do Protocolo C calibrada em Recife foi reaplicada a Curitiba (data do asset Sentinel como critério) e a Petrópolis (proxy regional vs. contexto). Ver `protocol_c_cross_region_reapplication_report.md` e a matriz de aprendizado.

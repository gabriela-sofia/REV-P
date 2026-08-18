# SUSC-17C33 - Event-Anchored Canary Patch Set e Validacao Observacional Review-Only

## Objetivo
Criar uma familia NOVA de patches canarios ancorados em ocorrencias oficiais hidrologicas (geocodificadas no 17C32), SEM substituir o patch original. Nao e mover o patch antigo para dar certo: e ancorar AOIs no evento observado e contrastar review-only com o patch original que falhou no vinculo espacial.

## Patches canarios ancorados
- Patches canarios (namespace event_anchored_canary_patch): 11.
- Bairros cobertos: Afogados, Areias, Imbiribeira, Ipsep, Iputinga, Pina, Varzea.
- Distancia ao patch original: 3142.1 m (min) a 12641.9 m (max).
- Cada patch canario: bbox centrado no ponto geocodificado (street-level) +- incerteza; ocorrencia oficial hidrologica na janela 2022-05-24..30.

## Patch original como controle
- Patches originais marcados spatial_mismatch_control_candidate: 5 (preservados, nao substituidos).
- Controles com ocorrencia oficial proxima (buffer 1500 m): 0.
- Ausencia de ocorrencia proxima e mismatch espacial, NAO negativo verdadeiro.

## Features pre-evento (honesto)
- CHIRPS region-level herdado do 17C25 (mesmo evento): 11 patches. CHIRPS ~5km NAO discrimina local.
- Features locais (HAND/slope/drenagem/Sentinel-2/urban_prop) disponiveis: 0 (indisponiveis offline; requerem extracao).

## Score v6 nos patches canarios
- Score v6 computavel: 0 de 11.
- score_v6_status = not_computable_pre_event_features_unavailable. Score v6 NAO recomputado, NAO alterado; score v7 NAO criado.

## Resposta observacional
As areas com ocorrencia oficial hidrologica sao observacionalmente MAIS relevantes ao evento (ancora de ocorrencia documentada) do que o patch original, que nao tem ocorrencia proxima (mismatch espacial, NAO negativo). Porem a comparacao QUANTITATIVA de score v6 fica DEFERIDA: o gatilho de chuva (CHIRPS) foi region-wide (identico); as features discriminantes locais (HAND/slope/drenagem/Sentinel-2/urban_prop) nao estao disponiveis offline. Conjunto canario materializado para extracao futura.

## Guardrails
- Patch original preservado (vira spatial_mismatch_control_candidate); patches novos em namespace proprio; nenhum vira ground truth/treino/substitui SUSC original; score v6 intacto; score v7 inexistente; ocorrencia sem hidrologico nao entra; centroide de bairro nao usado sem incerteza (ponto geocodificado + incerteza_m); ausencia de ocorrencia NAO vira negativo verdadeiro.

## minimum_success_achieved: True | result_class: A_observational_anchor_B_score_deferred

## Proximo marco recomendado
SUSC-17C34 Extracao pre-evento de features locais (HAND/slope/distance_to_drainage/Sentinel-2/urban_prop) para os patches canarios ancorados e calculo de score v6 review-only por patch canario, para comparar quantitativamente com o patch original; manter score v6 original intacto e 17B fail-closed.

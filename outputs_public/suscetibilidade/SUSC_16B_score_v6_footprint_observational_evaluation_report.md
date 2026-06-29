# SUSC-16B - avaliacao observacional review-only do score v6 contra footprints elegiveis

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`; `score_v7_created=false`.

O SUSC-16B avalia o score v6 contra footprints observacionais elegíveis em modo review-only. A etapa não cria ground truth, não libera treino supervisionado, não cria score v7 e trata controles como ausência documentada, não como negativos verdadeiros.

## 1. O que o SUSC-16A desbloqueou
O SUSC-16A desbloqueou 65 links footprint-patch elegiveis, 62 patches observacionais unicos e 12 footprints elegiveis para avaliacao observacional.

## 2. Por que 16B e avaliacao, nao calibracao
O 16B mede concordancia/divergencia entre score v6, features e footprints. Ele nao altera pesos, nao cria score v7, nao cria ground truth e nao treina modelo.

## 3. Dataset evento-controle
O dataset usa `observed_footprint_patch` para links elegiveis e `no_documented_footprint_control` para patches sem footprint elegivel documentado na mesma regiao quando disponivel. Controles nao sao negativos verdadeiros.

## 4. Metricas score v6 contra footprints
- Score medio evento: 0.457698
- Score medio controle: 0.608525
- Diferenca evento-controle: -0.150827

## 5. Resultado hit@top-k
hit@10=0.0; hit@20=0.0; hit@30=0.0. Baixa presenca em top-k foi registrada como possivel subestimacao do score v6, sem correcao automatica.

## 6. Contraste de features
Features que concordam: urban_prop.
Features que divergem: hand_mean, slope_mean, distance_to_water_mean, twi_mean, flow_accumulation_mean, ndbi_mean, ndvi_mean, mndwi_mean, chirps_7d_mm, chirps_30d_mm, runoff_context_7d.

## 7. Divergencias principais
Divergencias sao registradas como diagnostico review-only. Se controles tiverem score medio superior ou eventos aparecerem pouco no top-k, isso indica conflito de direcao ou possivel subestimacao.

## 8. Auditoria da qualidade dos footprints
Distribuicao dos tiers elegiveis: {"B_official_or_technical_flood_footprint": 6, "E_insufficient": 6}. Tiers D/E nao sustentam conclusao forte.

## 9. Casos de footprint com score baixo
Os casos baixo/medio estao em `SUSC_16B_low_score_footprint_case_audit.csv` com possiveis razoes auditaveis.

## 10. Recomendacoes de calibracao review-only
Recomendacoes candidatas: {"block_change_insufficient_evidence": 13}. Nenhuma recomendacao altera pesos neste marco.

## 11. Readiness para 16C
O 16B esta pronto para revisao de calibracao proxy 16C quando os criterios de links, footprints, contraste de features e auditoria de qualidade estao completos.

## 12. Por que score v7 ainda nao e criado
Score v7 permanece bloqueado neste marco por governanca: 16B e avaliacao, nao calibracao. Tambem ha divergencias/limitacoes que exigem revisao antes de qualquer candidato.

## 13. Limitacoes
Footprints permanecem candidatos observacionais, muitos sem data explicita. Controles representam ausencia documentada no SUSC-16A, nao ausencia real. A avaliacao e exploratoria e review-only.

## 14. Proximo marco
Executar SUSC-16C como revisao de calibracao proxy, mantendo fail-closed, sem treino e sem score v7 automatico ate que a evidencia seja suficiente e auditavel.

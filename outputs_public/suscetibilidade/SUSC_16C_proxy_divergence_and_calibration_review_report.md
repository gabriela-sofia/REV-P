# SUSC-16C - diagnostico profundo de divergencia e calibracao review-only do proxy

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`; `can_be_used_as_ground_truth=false`; `score_v7_created=false`.

O SUSC-16C transforma a divergencia entre score v6 e footprints observacionais em diagnostico auditavel de calibracao. A etapa permanece review-only, nao cria ground truth, nao libera treino supervisionado, nao altera o score v6 oficial e nao cria score v7 automatico.

## 1. Estado herdado do 16A/16B

O SUSC-16A desbloqueou links footprint-patch elegiveis e o SUSC-16B confirmou avaliacao observacional do score v6 contra esses footprints. O estado herdado do 16B registra score medio de evento menor que controle e hit@top-k nulo, indicando divergencia util para diagnostico.

## 2. Por que o resultado ruim do score v6 e util

O resultado ruim nao foi escondido. Ele foi usado para isolar casos em que footprints observacionais aparecem em patches low/medium, identificar features que ainda sustentam suscetibilidade e apontar componentes que podem estar subestimados.

## 3. Auditoria do worktree antes do 16C

A auditoria esta em `SUSC_WORKTREE_AUDIT_BEFORE_16C.md`. Nenhum arquivo SUSC-16A/SUSC-16B de entrada apareceu modificado localmente; a sujeira fora do escopo foi preservada.

## 4. Dataset analitico unificado

Linhas totais: 103. Links evento: 65. Patches observacionais unicos: 62. Casos low/medium: 57.

## 5. Decomposicao do score v6

Distribuicao do componente mais baixo: {"documentary_component": 65}.

## 6. Auditoria individual dos 65 links

Foram gerados arquivos individuais em `SUSC_16C_individual_cases/`, um por link evento elegivel.

## 7. Features que sustentam suscetibilidade apesar do score baixo

As features de suporte sao registradas caso a caso em `SUSC_16C_individual_case_audit.csv` e agregadas na estabilidade de direcao. Elas incluem sinais hidrologicos, urbanos, espectrais e de chuva/runoff quando os limiares review-only foram satisfeitos.

## 8. Features que contradizem os footprints

Contradicoes possiveis foram registradas como HAND alto, distancia alta da agua, TWI baixo, baixa exposicao urbana, vegetacao alta ou chuva fraca, sempre como diagnostico e nao como negacao do evento.

## 9. Estabilidade das direcoes por regiao/qualidade

Resumo: {"contradictory": 15, "stable_support": 1}.

## 10. Cenarios de sensibilidade de pesos

Foram simulados 8 cenarios review-only. Melhor hit@30 simulado: `increase_spectral_water_weight` com 0.366667. Nenhum score v7 foi persistido.

## 11. Modos de falha do proxy

Distribuicao primaria: {"rainfall_trigger_underweighted": 5, "urban_flash_flood_underrepresented": 60}.

## 12. Matriz de design para calibracao futura

A matriz possui 22 linhas; 7 itens ficam elegiveis para discussao no 16D. `eligible_for_score_v7_future` permanece false por desenho fail-closed.

## 13. O que pode ser melhorado no proxy

Os candidatos de melhoria incluem calibracao regional, reavaliacao de hidrologia/topografia, chuva/runoff, exposicao urbana e sinal espectral umido. Essas acoes sao apenas candidatas.

## 14. O que nao pode ser afirmado ainda

Nao se pode afirmar ground truth, negativo verdadeiro, prontidao para treino, causalidade ou score v7 pronto. Footprints permanecem evidencias observacionais candidatas.

## 15. Por que score v7 nao foi criado

O 16C e diagnostico e desenho review-only. Criar score v7 exigiria decisao metodologica posterior, mais evidencia e validacao especifica.

## 16. Proximo marco recomendado

Executar SUSC-16D como desenho controlado de calibracao candidata, ainda sem treino e sem ground truth, priorizando itens elegiveis da matriz 16C.

## Distribuicao score v6 nos links evento

{"high": 8, "low": 31, "medium": 26}

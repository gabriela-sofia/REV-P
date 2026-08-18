# SUSC-10D — Diagnóstico DINO × Score v6 (ausência documentada)

> O SUSC-10D usa DINO apenas como camada representacional diagnóstica. Ele não treina classificador, não cria ground truth e não transforma embeddings em prova de ocorrência de enchente.

## Status: DINO AUSENTE (sem vetores carregáveis por patch)

A descoberta (`SUSC_10D_dino_embedding_discovery.csv`) encontrou 349 arquivos relacionados a DINO, mas **nenhum vetor por patch carregável**: os registries/feature-store estão vazios (scaffolds) e os vetores brutos `.npz/.npy` são git-ignored / externos (PROJETO).

## Consequência
- Nenhuma similaridade/PCA/centróide/outlier foi computada (sem vetores reais).
- Tabelas de alinhamento/região/outliers foram geradas vazias (apenas cabeçalho).
- Isso **não** bloqueia a linha SUSC: DINO é camada complementar diagnóstica, não requisito.

## O que seria feito se houvesse vetores
Alinhamento por `patch_id`, cosine ao centróide regional, PCA (se sklearn), distância ao centróide e outliers representacionais — sempre review-only, nunca como classificador.

## Limitações
Os vetores DINO reais (referenciados por URI/hash em manifests v1ph/v1pq) não estão no REV-P por política de peso/governança. Integração futura exige trazer vetores leves por patch de forma rastreável.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.

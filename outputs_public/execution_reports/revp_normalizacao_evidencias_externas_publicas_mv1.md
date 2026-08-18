# Normalização das evidências externas públicas — MV1

> Pacote review-only da frente de evidências externas. Remove a noção de "licença como bloqueador" e a substitui por bloqueadores técnicos/metodológicos reais. Não cria label, não altera a decisão review-only, não libera treino, não declara ground truth operacional, não copia bruto para `outputs_public`. Artefatos anteriores são preservados; as versões normalizadas recebem sufixo `_normalizado_mv1`/`_normalizada_mv1`.

## 1. Escopo

Esta normalização cobre toda a frente de evidências externas do marco MV1: curadoria original, fechamento de downloads, navegação/downloads, integração com o marco label-free, auditoria de reprodutibilidade externa, manifesto público URL+SHA256+tamanho, recomendações de pacote externo e a checagem de que nenhum bruto pesado foi para `outputs_public`.

A varredura foi feita em `outputs_public/`, `docs/`, `scripts/curadoria_externa/` e testes relacionados. Os bloqueadores de licença em campos estruturados estavam concentrados em três tabelas de auditoria de fontes (`revp_auditoria_fontes_externas_mv1`, `_downloads_mv1`, `_navegacao_mv1`) e na lógica do script `revp_curadoria_evidencias_externas_mv1.py`.

## 2. Por que a normalização foi necessária

Os artefatos anteriores tratavam licença como critério de auditoria (`licenca_ou_uso_claro`) e, quando a licença não estava explícita, emitiam um bloqueador de uso público (`uso_publico_direto_bloqueado` / `bloquear_para_uso_publico_ate_confirmar_licenca`). Isso classificava fontes públicas/institucionais como inutilizáveis por ausência de licença explícita — uma premissa incorreta para esta frente, em que todos os dados são públicos/institucionais.

## 3. Diferença entre dado público e reprodutibilidade técnica

**Dado público/institucional** = origem oficial rastreável (IBGE, ANA, SGB/CPRM, IPPUC, MapBiomas, CEMADEN, APAC, DRM-RJ, Copernicus/NHESS). Isto já está estabelecido para todas as fontes.

**Reprodutibilidade técnica** = uma pessoa externa consegue obter o mesmo arquivo (URL responde) e conferir a integridade (SHA256 + tamanho + parseabilidade). É isto, e não licença, que condiciona o uso operacional.

## 4. Por que licença não é bloqueador nesta frente

Os dados são públicos/institucionais. Ausência de licença explícita não impede auditabilidade nem reprodutibilidade. Por isso, nenhum artefato novo usa licença como bloqueador. O que efetivamente condiciona o uso é técnico/metodológico: URL, hash, formato, tamanho, disponibilidade, necessidade de navegação manual ou solicitação formal, ausência de geometria/evento, suscetibilidade vs evento, landslide vs flood, circularidade e reprodutibilidade pública.

## 5. Bloqueadores técnicos/metodológicos usados

Vocabulário fechado aplicado nos artefatos normalizados: `portal_sem_arquivo_direto`, `requer_solicitacao_formal`, `arquivo_pesado_para_git`, `sem_geometria`, `sem_crs`, `fonte_contextual_nao_evento`, `suscetibilidade_nao_evento_observado`, `catalogo_nao_serie_filtrada`, `fonte_academica_secundaria`, `risco_landslide_vs_flood`, `risco_circularidade`, `dependencia_local_only`, `dependencia_download_externo`, `html_de_portal_nao_dado_final`, `sem_produto_vetorial_publico`, `download_massa_pesada_nao_recomendado`.

Mapeamento dos bloqueadores reais por fonte (que antes eram "licença"):
- **CEMADEN (FONTE_NAC_001)** → `requer_solicitacao_formal` (formulário/e-mail).
- **APAC (FONTE_NAC_005)** → `portal_sem_arquivo_direto` (sem PDF/CSV estático; PCD em tempo real).
- **Copernicus EMS (FONTE_INT_001)** → `sem_produto_vetorial_publico` (sem produto rapid mapping para Petrópolis).
- **DRM-RJ (FONTE_NAC_006)** → `requer_solicitacao_formal` (portal 503; alternativa SGB usada).
- Demais fontes com arquivo verificado → sem bloqueador (`manter_referencia_url_hash`).

## 6. Manifesto público URL + SHA256 + tamanho

Foi gerado `revp_manifesto_publico_arquivos_externos_url_hash_mv1.csv` com os 8 arquivos externos verificados. Todos: existem em `local_only`, SHA256 confere com o registrado, tamanho confere, parseiam. `pode_ir_para_git=false` (bruto externo de quarentena / arquivo pesado), `pode_ir_para_pacote_externo=sim`. Este manifesto é o artefato leve que torna a frente reprodutível por terceiros: re-download pela URL oficial e conferência de hash/tamanho.

## 7. O que é verificável por pessoa externa

Os 8 arquivos: URLs oficiais respondem (verificadas em 2026-06-19), SHA256 e tamanho conferem, formatos parseiam (GeoJSON como `FeatureCollection`, PDFs com `%PDF`/`%%EOF`, XLSX como OOXML). Logo, qualquer pessoa externa pode reproduzir e auditar a integridade sem acesso privilegiado.

## 8. O que depende de `local_only`

Os binários permanecem em `local_only/evidencias_externas_quarentena/` (quarentena, git-ignored). O repositório público traz apenas métricas derivadas, hashes e o manifesto URL+hash. Bloqueador técnico aplicável: `dependencia_local_only` — resolvido na prática pelo manifesto público (re-download).

## 9. O que depende de re-download oficial

Todos os 8 arquivos são reproduzíveis por re-download direto das URLs oficiais. Bloqueador técnico: `dependencia_download_externo` (esperado e aceitável; URLs verificadas respondendo).

## 10. O que depende de solicitação formal

- **CEMADEN** — formulário com envio de link por e-mail.
- **DRM-RJ** — sem URL direta; portal 503; papel de suscetibilidade coberto por alternativa oficial (SGB/CPRM).

## 11. O que não deve ir para Git por motivo técnico

Nenhum dos 8 binários externos deve entrar no Git/`outputs_public` (bruto externo de quarentena / arquivos de 4–14 MB). O SIG vetorial completo da carta SGB (~1,8 GB) é `download_massa_pesada_nao_recomendado`. Confirmação: **0 brutos pesados em `outputs_public`** (varredura por extensão e por tamanho > 5 MB). O motivo é sempre técnico (peso/quarentena), nunca licença.

## 12. Quais artefatos anteriores foram normalizados

- `revp_auditoria_fontes_externas_mv1.csv`, `_downloads_mv1.csv`, `_navegacao_mv1.csv` — 33 ocorrências de licença como bloqueador localizadas e substituídas por bloqueadores técnicos (versão canônica normalizada: `revp_auditoria_fontes_externas_normalizada_mv1.csv`).
- Manifestos de fontes externas (coluna `licenca_ou_uso`) — substituída por `origem_publica_institucional` + `reprodutibilidade_tecnica` (`revp_manifesto_evidencias_externas_normalizado_mv1.csv`).
- Recomendações de pacote externo — reemitidas sem licença (`revp_recomendacoes_pacote_externo_normalizada_mv1.csv`).
- `scripts/curadoria_externa/revp_curadoria_evidencias_externas_mv1.py` — origem do bloqueador registrada na tabela de substituições como recomendação de refactor; **não editado destrutivamente** (preservação de histórico).

A rastreabilidade completa está em `revp_normalizacao_bloqueadores_evidencias_externas_mv1.csv` (antigo → normalizado, com justificativa).

## 13. Guardrails preservados

- evidência externa não vira label;
- download não vira validação operacional;
- suscetibilidade não vira evento observado;
- landslide scar não vira flood extent;
- texto sem geometria não fecha patch-level;
- Curitiba não vira negativo formal;
- não liberar treino;
- não declarar ground truth operacional;
- `pode_virar_label_agora=false`.

## 14. Conclusão

Os dados externos são públicos/institucionais; o que bloqueia o uso operacional não é licença, mas sim reprodutibilidade técnica, ausência de geometria/evento patch-level, ausência de CRS/overlay quando aplicável, necessidade de solicitação formal ou risco metodológico. A frente de evidências externas fica coerente, auditável e reprodutível: 8 arquivos verificáveis por URL+SHA256+tamanho, 0 bruto pesado em `outputs_public`, manifesto público leve disponível, e nenhum bloqueador de licença remanescente nos artefatos normalizados. Ground truth operacional permanece ausente e treino supervisionado permanece bloqueado.
</content>

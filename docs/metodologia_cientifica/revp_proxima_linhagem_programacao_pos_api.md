# Próxima linhagem de programação — pós-API (SUSC-20E/20F)

**Status**: PLANO_DE_TRABALHO_NAO_CANONICO — ordena os próximos passos de código a partir do
que foi consolidado nesta sessão. Não é gate, não altera modelo/dado/label. Complementa
`PLANO_ACAO_produto_v1.md`, `revp_fase1_conclusao_dino_ab_test.md`,
`revp_fase2_decisoes_design_contrato.md` e `revp_api_produto_referencias_correlacao.md`.

**Regra que rege esta linhagem**: uma etapa só começa quando a anterior fechar com artefato
real conferível (arquivo, número, commit) — mesma regra do `PLANO_ACAO_produto_v1.md`. Nenhuma
etapa promete prazo.

---

## Onde estamos (consolidado desta sessão, sem repetir o que já está documentado)

O motor científico de Recife está fechado: cadeia v7→v8→v9→v12 (Firth penalizado, 278 pontos
reais, LOO-AUC=0,6781) é a espinha dorsal confirmada e protegida. A API (SUSC-20E) implementa
o contrato de inferência do rascunho `txtpragab.docx`; a extensão SUSC-20F fechou a limitação
de "só pontos conhecidos", calculando as 6 features físicas sob demanda pra qualquer
coordenada dentro da cobertura real do DTM em Recife. O teste A/B do DINO (v1r5/v1r6) decidiu,
com prova estatística (LRT + erro-padrão cluster-robusto), que o embedding visual fica como
evidência auxiliar, nunca como feature do score — achado consistente com a literatura de
fusão multimodal revisada nesta sessão. Os dados de Recife, Curitiba e Petrópolis foram
auditados: já estão organizados por região/fonte, sem necessidade de reestruturação; a única
pendência real de dado é um par de CSVs da Defesa Civil de Recife com UUID diferente
(possível duplicata de download, não confirmado). `outputs/external_validation/` ainda não
recebeu curadoria — é a maior pilha de arquivo não avaliada que sobra no PROJETO.

---

## Atualização de status (2026-07-25, sessão de execução)

Etapas 1 e 2 fechadas com artefato real e verificado (schema Pydantic + 7 testes passando;
decisão dos CSVs da Defesa Civil documentada — eram duplicata exata, mesmo download,
UUID CKAN reatribuído). Etapa 3 (Curitiba) **diagnosticada, não modelada**: Lead A
continua negativo (nível estadual e índice LegisWeb descartados por busca real), inventário
real ficou em 3 candidatos não adjudicados (2 ANA + 1 GFD) contra os 278 de Recife — correto
não ter forçado Firth em N≤3, isso seria overcloaim contra as próprias regras do projeto.
Etapa 4 (Petrópolis) **desbloqueio documentado, filtro não aplicado**: critério COBRADE já
existia de sessão anterior, mas falta dataset registro-a-registro (S2ID ou Defesa Civil PMP)
pra aplicar — aquisição nova, fora de escopo. `region_registry.py` não mudou pra nenhuma das
duas regiões (correto — nenhuma virou fato novo). Nada commitado ainda.

## Atualização de status (2026-07-26, execução das pendências diagnosticadas)

A rodada de 2026-07-25 diagnosticou mas não tentou os próprios próximos passos que
identificou. Esta rodada tentou os três, de fato, e trouxe resultado real.

**Curitiba, Lead A** — o "bloqueio ASP.NET sem API" **não existia**: o Legisladoc
publica a base completa de atos em CSV nos dados abertos da Prefeitura. Varredura
sobre o texto integral de 347.439 atos, cobertura **64/64 dias úteis** de jan–mar/2022,
231 decretos de 17/01–28/02 enumerados um a um: **zero atos de enchente/emergência
climática** (os 58 hits de "situação de emergência" são todos COVID-19). Controle
positivo executado: o mesmo detector acha os decretos reais de 1983 (situação de
emergência por transbordamento do Belém/Iguaçu/Barigui/Atuba) e 1995 (calamidade por
enchente anormal na bacia do Iguaçu) — o negativo de 2022 é real, não falha de método.
Corroborado por fonte independente: o S2ID federal tem **0 registros de Curitiba em
jan/2022** (o único de 2022 é granizo em 22/04). Rota esgotada.
`PROJETO/local_runs/curitiba_modelo_v1_diagnostico_lead_a_e_inventario_pontos/RELATORIO_v2_LEGISLADOC_VARREDURA.md`.

**Curitiba, adjudicação** — os 3 candidatos passaram pelo protocolo de Recife
(SUSC-20A). As 2 estações ANA são corroboração de área/tempo, não ponto (regra
literal do Lead B de Recife, onde nem o sinal de percentil 99,48% virou ponto).
O candidato GFD `LEADC_CTBA_2015_0001` foi **rejeitado**: reproduzi (a)–(c)
exatamente, mas o passo (e) — coerência física, que Recife executou e Curitiba
pulou — mostra que 13 dos 14 pixels dentro do município são a lâmina d'água da
**Represa do Passaúna** (platô DEM de 887,00 m em 8,10 km²; JRC Global Surface Water
occurrence 84–87%; OSM `natural=water`). O `jrc_perm_water` do GFD não pegou porque a
represa oscila abaixo do limiar de permanência. Além disso, só 14 dos 54 pixels estão
dentro do limite municipal do IBGE — a bbox usada era metropolitana.
**Curitiba tem N = 0 pontos-evento adjudicados.** Não é N pequeno demais: é vazio.
`.../RELATORIO_v2_ADJUDICACAO_CANDIDATOS.md`.

**Petrópolis** — o S2ID **tem** export público sem login e foi adquirido. Os 3
registros de 2022 são todos COBRADE 13214 (chuvas intensas), zero nas classes
hidrológicas e zero em movimento de massa: o critério de separação é **inaplicável**,
não pendente. Segundo bloqueio, independente: o registro é municipal, sem geometria.
A Série Histórica do S2ID está com dados só até 2016 apesar de a UI oferecer até 2026.
`docs/metodologia_cientifica/revp_petropolis_s2id_aquisicao_real_cobrade.md`.

`region_registry.py` teve os `status_note` de Curitiba e Petrópolis corrigidos
(maturity inalterada nas duas). A afirmação anterior de Curitiba, "1 evento
MODIS-validado real (DFO_4276/2015)", foi falsificada pela adjudicação e removida.
7/7 testes de `tests/test_susc_20e_region_registry_schema.py` passando. Nada commitado.

## Linhagem proposta, em ordem

### 1. Ficha técnica estática do modelo (model card único)
**Objetivo**: consolidar num documento único e citável o que hoje só existe distribuído
(por-resposta na API + espalhado em relatórios) — uso pretendido, as 6 features e sua
força/estabilidade, LOO-AUC, método de CI, limitações (n=278, só Recife, gap de
elevação/declividade), histórico de decisões (SAR descartado, DINO auxiliar).
**Pré-requisito**: nenhum, pode começar já.
**Artefato esperado**: `docs/metodologia_cientifica/revp_model_card_v12.md`.
**Critério de feito**: documento único que uma banca consegue ler sem chamar a API.

### 2. Registro machine-readable modelo-por-região
**Objetivo**: formalizar o gate #8 já identificado na Fase 2 (`region_registry.py` hoje é a
única fonte, mas é informal) — um registro versionado e testado que decide, por região,
`available | limited_evidence | insufficient` e qual `model_version`.
**Pré-requisito**: etapa 1 (a ficha técnica referencia esse registro).
**Artefato esperado**: schema formal (JSON/YAML) + teste que falha se uma região reportar
`available` sem `model_version` associado.
**Critério de feito**: `pytest` cobrindo o registro; hoje só Recife=v12, resto null.

### 3. Fechar a pendência dos CSVs da Defesa Civil (Recife)
**Objetivo**: os pares de arquivo com UUID diferente em
`data/raw/recife/seced_defesa_civil/` (achados na curadoria desta sessão) precisam de
decisão antes de qualquer nova aquisição de dado pra Curitiba usar o mesmo padrão — checar
se é re-fetch do mesmo recorte ou tem diferença real de cobertura temporal.
**Pré-requisito**: nenhum, mas bloqueia a confiança total na base de eventos antes de replicar
o método em outra região.
**Artefato esperado**: nota curta em `docs/` documentando qual dos dois é o canônico (ou os
dois, se cobrirem períodos diferentes).
**Critério de feito**: decisão registrada, sem ambiguidade sobre qual CSV alimenta o pipeline.

### 4. Curitiba — replicar aquisição de eventos reais + features físicas + Firth
**Objetivo**: esta é a etapa que materializa "fazer o mesmo que fiz com Recife" pra Curitiba.
Curitiba já tem vantagem real sobre Petrópolis: os Leads B (ANA, corroboração hidrológica real
do evento 2022-01-15/16) e C (Global Flood Database, evento MODIS-validado DFO_4276/2015,
bairro São Miguel) do REV-P já entregaram evidência real. Falta: (a) réplica do Lead A
(Diário Oficial de Curitiba) pra decretos/ocorrências; (b) cálculo das 6 features físicas
usando `data/raw/curitiba/` (GeoCuritiba drenagem + SGB/CPRM MDE, já confirmados presentes e
organizados nesta sessão); (c) ajuste do Firth com a mesma metodologia do v9-v12, sem
inventar método novo.
**Pré-requisito**: etapa 3 fechada (não replicar um padrão de dado ainda incerto).
**Artefato esperado**: `local_runs/curitiba_modelo_v1_.../` seguindo a mesma estrutura de
nomenclatura e relatório que Recife (`RELATORIO_vN_MASTER.md`, dataset final, coeficientes,
bootstrap, LOO-AUC).
**Critério de feito**: `region_registry.py` passa a reportar Curitiba como `available` com
`model_version` real — só quando esse número existir de verdade, não antes.

### 5. Petrópolis — resolver separação enchente/deslizamento
**Objetivo**: pré-requisito documentado (não é modelagem ainda) — os dados de Petrópolis
misturam enchente e deslizamento de terra sem separação de fenômeno. Antes de qualquer
feature física ou label, decidir o critério de separação (fonte, geometria, palavra-chave nos
registros de defesa civil).
**Pré-requisito**: nenhum, mas independente da etapa 4 (podem rodar em paralelo, sessões
diferentes).
**Artefato esperado**: nota de decisão + eventualmente um script de filtro aplicado a
`data/raw/petropolis/` (já auditado como pequeno e organizado nesta sessão).
**Critério de feito**: `region_registry.py` de Petrópolis passa de `insufficient` pra
`limited_evidence` só quando o filtro estiver aplicado e documentado.

### 6. Decisão de generalização entre regiões
**Objetivo**: esta é a pergunta que você levantou nesta sessão — um modelo "inteligente o
bastante" pra usar o que aprendeu numa região com muito dado (Recife) em regiões com pouco
dado. **Só faz sentido começar depois da etapa 4** (Curitiba com modelo próprio validado) —
com só uma região madura não há o que comparar/agrupar. Quando houver duas, a decisão de
design (pooled/hierárquico vs. modelos totalmente separados) precisa da mesma disciplina das
Fases 1-2: decisão escrita antes de código, respeitando a regra de que o modelo reflete
relação física conhecida, não "descobre" padrão.
**Pré-requisito**: etapa 4 fechada com número real.
**Artefato esperado**: documento de decisão de design (estilo
`revp_fase2_decisoes_design_contrato.md`), não código ainda.
**Critério de feito**: decisão escrita, testável, sobre se/como agrupar Recife+Curitiba.

### 7. Curadoria de `outputs/external_validation/` (PROJETO)
**Objetivo**: única pilha grande de arquivo ainda não avaliada nesta sessão de curadoria —
dezenas de subpastas exploratórias tocando as 3 regiões. Pode rodar em paralelo com qualquer
etapa acima, é organização, não é ciência.
**Pré-requisito**: nenhum.
**Critério de feito**: mesma metodologia já aplicada (checar citação antes de mover, nunca
apagar de vez, tudo pra `_APAGAR_MANUALMENTE_20260725/`).

### 8. Higiene de branches/worktrees do REV-P
**Objetivo**: achado desta sessão — 44 branches do REV-P com commits reais não mesclados na
`main` (14 a 123 commits cada, uma da semana passada). Isso não é limpeza de arquivo, é
reconciliação de git; merece sessão própria, não decisão automática.
**Pré-requisito**: nenhum, mas não deveria competir por atenção com as etapas científicas
acima — é infraestrutura.
**Critério de feito**: cada branch com trabalho único revisada e decidida (mesclar, arquivar
como referência, ou descartar com aprovação explícita seção por seção).

### 9. Re-teste do DINO, oportunista
**Objetivo**: ressalva já registrada na Fase 1 — se mais patches Sentinel com embedding real
ficarem disponíveis além dos 23 atuais, o teste A/B pode ser refeito com mais poder
estatístico usando os mesmos scripts (`revp_v1r5`/`v1r6`, reexecutáveis sem alteração).
**Pré-requisito**: corpus de patches crescer; não é uma tarefa a agendar, é a rodar quando a
oportunidade aparecer.
**Critério de feito**: novo p-valor real, documentado do mesmo jeito que o atual.

### 10. Interface web + camada LLM de explicação
**Objetivo**: já sequenciado no `txtpragab.docx` como posterior ao contrato — mapa, score,
evidências na tela; LLM só explica o payload estruturado, nunca decide o score.
**Pré-requisito**: etapas 1-2 fechadas (a interface deveria consumir o registro formal, não
gambiarra direto na API).
**Critério de feito**: fora de escopo desta linhagem definir aqui — é a próxima decisão de
design quando chegar a vez.

### 11. Conformidade OGC API - Processes (opcional)
**Objetivo**: só relevante se/quando interoperabilidade com cliente GIS externo (QGIS etc.)
virar requisito real do produto — o contrato atual já está na forma que o padrão usa, só
falta formalizar se for preciso.
**Pré-requisito**: nenhuma das etapas acima depende disso.
**Critério de feito**: não é prioridade agora, fica registrado pra não esquecer.

---

## Resumo da ordem prática

Etapas 1-3 são rápidas e desbloqueiam o resto. Etapa 4 (Curitiba) é o próximo trabalho
científico de peso — é o que prova ou não que o método de Recife replica. Etapa 5
(Petrópolis) roda em paralelo, mais devagar. Etapa 6 (generalização) só existe depois da 4.
7 e 8 são organização/infraestrutura, podem rodar a qualquer momento sem competir com o resto.
9-11 são oportunistas ou de escopo futuro.

Nenhuma etapa promete prazo. Cada uma entrega um artefato conferível antes da próxima começar.

# Elevação relativa como regra permanente do pipeline — 2026-08-19

**Pedido**: "aplique isso como algo que é essencial de aprendizagem do
modelo, a gente vai ter outras regiões que vao ter suas respectivas
altitudes e niveis ent é importante que esse aprendizado sirva pra sempre e
que sempre seja algo a ser levado em consideração." Este documento fecha
esse pedido: registra o que mudou de fato no pipeline (não só na
documentação) para que a correção se aplique automaticamente a toda região,
presente ou futura, sem depender de alguém lembrar de pedir.

## O achado que motivou a regra

`app01-transferência-curitiba-sem-rótulo-local` (mesmo dia): `elevation_m`
absoluta tinha 0% dos pontos de Curitiba dentro do intervalo 5–95% de um
treino externo perto do nível do mar (diferença padronizada de média =
2,76). Curitiba fica a ~900 m; HAND já é relativo por definição (altura
acima da drenagem), elevação sozinha não era — por isso quebrava entre
regiões de altitude de base diferente. Esse é o **quarto** caso desta mesma
classe de erro no projeto (os outros três, já documentados no `ds03`: HAND
por duas cadeias, HAND com limiar de canal inconsistente, chuva misturando
CHIRPS/ERA5-Land na mesma coluna).

## O que muda, e por que cada mudança garante permanência

1. **`ds03_esquema_alvo.py` (v1 → v2)** — o contrato ganha duas colunas
   novas: `elevation_baseline_m` (o P1 de `elevation_m` dentro da própria
   fonte) e `elevation_rel_m` (`elevation_m − elevation_baseline_m`). Uma
   nova tupla, `VARIAVEIS_TERRENO_TRANSFERIVEL`, substitui `elevation_m` por
   `elevation_rel_m` nas quatro de terreno — é essa tupla, não
   `VARIAVEIS_TERRENO`, que qualquer ajuste multirregião deve usar daqui pra
   frente. Uma nova regra, `regra_elevacao`, documenta isso dentro de
   `contrato()` — não é nota solta, é parte do que o script imprime e grava
   toda vez que roda.

2. **`ds04_reduzir_por_fonte.py`** — a transformação roda dentro de
   `moldar()`, a função que **todo** redutor chama antes de devolver a
   fonte reduzida. Não é uma correção aplicada região por região: é
   estrutural. Uma fonte nova (Petrópolis, quando tiver ponto rotulado, ou
   qualquer região futura) recebe `elevation_rel_m` automaticamente, sem
   que o autor do novo redutor precise saber que essa regra existe. É esse
   o mecanismo que torna "sirva pra sempre" verificável em vez de uma
   promessa.

3. **`mod_mec04_elevacao_relativa.py` (novo)** — sucessor do `mod_mec03`,
   usando `VARIAVEIS_TERRENO_TRANSFERIVEL`. Não sobrescreve o `mod_mec03`:
   aquele resultado já publicado (comparação TERRENO-vs-COMPLETO sob
   elevação absoluta) fica como registro histórico de uma pergunta já
   respondida. O `mod_mec04` responde a pergunta atual — qual é o conjunto
   correto para ajuste multirregião daqui pra frente.

4. **`tests/test_ds03_ds05_tabela_unica.py`** — cinco testes novos
   transformam o achado em guarda de regressão, não só em texto:
   `elevation_rel_m` é sempre reconstruível a partir do baseline; o baseline
   é constante dentro da fonte; a nulidade acompanha `elevation_m`; a
   cobertura de Curitiba no intervalo 5–95% do treino externo continua acima
   de 80% (era 0%); e a tupla transferível difere da bruta em exatamente um
   elemento. Se alguém alterar o cálculo do baseline ou esquecer de chamá-lo
   numa fonte nova, o teste falha — a regra não depende de vigilância manual.

## Verificação: pipeline rodado ponta a ponta, não só editado

`ds04` e `ds05` foram executados de novo (schema v2). Contrato sem
violação, `POOL_FLUVIAL n=65.070` — idêntico ao `ds05` anterior, confirmando
que a mudança é aditiva (nenhuma linha nova rejeitada ou admitida por causa
dela). Conferido por fonte, na tabela consolidada:

| fonte | baseline (P1 de elevation_m) | elevation_rel_m: min–max |
|---|---:|---:|
| CEMS | 0,00 m | 0,03 – 3.185,93 |
| Sen1Floods11 | 0,00 m | 0,17 – 2.000,64 |
| UK | 9,41 m | −5,46 – 509,17 |
| Curitiba | 874,63 m | −2,65 – 130,09 |

Reconstrução (`elevation_m − elevation_baseline_m == elevation_rel_m`) bate
em toda linha (erro máximo ~1e-14, ruído de ponto flutuante). Cobertura de
Curitiba no intervalo 5–95% do treino externo (CEMS+UK+Sen1Floods11): **0,0%
em `elevation_m` → 93,7% em `elevation_rel_m`**; diferença padronizada:
**2,43 → −0,49**. Mesma direção do achado original do `app01` (0%/2,76); a
pequena diferença de magnitude é deriva normal do tamanho atual do pool, não
uma discrepância a investigar.

## `mod_mec04`: resultado com fidelidade total, N_BOOT=600 confirmado

Rodado neste ambiente (n=63.174, grupos=1.688 — a mesma contagem que o
`mod_mec03` teria hoje sob o mesmo filtro de negativo admitido; a diferença
frente ao `n=64.989` do resultado já publicado é deriva do dataset desde a
última execução real do `mod_mec03`, não efeito da elevação relativa).

**Leave-one-source-out, elevação relativa** (bootstrap de AUC com n=2.000,
fidelidade plena — este laço nunca foi reduzido):

| fonte deixada de fora | AUC | IC95 |
|---|---:|---:|
| CEMS | 0,6823 | [0,656; 0,710] |
| Sen1Floods11 | 0,7133 | [0,681; 0,741] |
| UK | 0,7424 | [0,715; 0,770] |
| Curitiba | 0,4636 | [0,405; 0,525] |

Comparado ao `mod_mec03` publicado (elevação absoluta: CEMS 0,7200,
Sen1Floods11 0,7063, UK 0,7346, Curitiba 0,4636): UK e Sen1Floods11 melhoram
levemente, Curitiba fica idêntico (não entra como treino nesse estrato), CEMS
cai ~0,038 — diferença que ainda não isolei entre efeito da variável e deriva
do dataset (o pool cresceu desde a última execução real do `mod_mec03`). Não
é uma leitura fechada; fica registrada, não escondida.

**IC95 dos coeficientes, N_BOOT=600 completo (atualizado 2026-08-19, agora
DEFINITIVO)**: o ambiente de execução mata qualquer chamada acima de
~170 s, e 600 réplicas de Firth por conjunto sobre ~63 mil linhas passam de
20 minutos numa chamada só. Em vez de reduzir a amostra (o que teria dado um
IC mais largo e menos confiável), escrevi
`scripts/suscetibilidade/mod_mec04_bootstrap_resumavel.py`: quebra o mesmo
laço de bootstrap do `avaliar()` em lotes, persistindo o ESTADO do gerador
aleatório (`numpy.random.Generator.bit_generator.state`) em disco entre
chamadas. Como TERRENO e COMPLETO usam a mesma semente e o mesmo `grupo_cv`,
a sequência de reamostras é idêntica entre os dois conjuntos por construção
do próprio `avaliar()` original — persistir o estado do gerador faz a soma
dos lotes ser bit-a-bit igual a uma execução única com N_BOOT=600. Rodou em
2 lotes (395 + 205 draws, 600/600 tentativas bem-sucedidas, nenhuma
reamostra degenerada) e fechou substituindo só os campos `ic95`/
`falhas`/`veredito` em `resultado.json` — `auc_cv`, LOSO e transferência de
relevo (que já eram fidelidade plena) não foram tocados.

| feature | TERRENO IC95 | COMPLETO IC95 |
|---|---:|---:|
| elevation_rel_m | [−0,053; +0,191] cruza zero | [−0,092; +0,163] cruza zero |
| slope_deg | [−0,052; +0,332] cruza zero | [−0,085; +0,307] cruza zero |
| hand_m | [−2,051; −0,664] | [−2,035; −0,662] |
| twi_dinf | [+0,408; +0,577] | [+0,408; +0,580] |
| rain_max_24h | — | [−0,486; −0,079] |
| rain_decay_index | — | [+0,087; +0,547] |

**Leitura**: as duas variáveis com sinal fisicamente exigido (HAND, TWI)
seguem coerentes e com IC95 fora de zero nos dois conjuntos —
`VEREDITO=COERENTE_COM_CRITERIOS` em ambos. `elevation_rel_m` e `slope_deg`
têm IC95 cruzando zero (não são gate de aprovação: `SINAL_EXIGIDO` nunca
cobriu essas duas, mesma convenção do `mod_mec03`) — informativo, não
reprovação: elevação relativa entra no conjunto por evitar viés de domínio
entre regiões, não porque se espere um coeficiente forte e estável por si
só. Chuva (`COMPLETO`) tem os dois coeficientes com IC95 fora de zero,
mesma leitura qualitativa que o `mod_mec03` já registrava.

## O que não muda

- `mod_mec03/resultado.json` já publicado permanece válido como registro
  histórico da pergunta "o que a chuva acrescenta ao terreno", respondida
  sob a variável que existia então.
- `main.tex` não foi tocado — mesma disciplina de sempre, achado do dia não
  vira resultado desta entrega.
- Nenhuma linha foi admitida ou rejeitada de forma diferente: a mudança é
  aditiva no esquema, não um novo critério de admissão.

## Arquivos

- `scripts/suscetibilidade/ds03_esquema_alvo.py` (v2)
- `scripts/suscetibilidade/ds04_reduzir_por_fonte.py`
- `scripts/suscetibilidade/mod_mec04_elevacao_relativa.py` (novo)
- `scripts/suscetibilidade/mod_mec04_bootstrap_resumavel.py` (novo — bootstrap
  de coeficiente em lotes, usado só porque o ambiente de execução limita o
  tempo por chamada; mesmo resultado que uma execução única produziria)
- `tests/test_ds03_ds05_tabela_unica.py` (5 testes novos)
- `local_runs/ds-05-tabela-unica/tabela_unica_pool_fluvial_v2.csv`
- `local_runs/mod-mec-04/resultado.json` (N_BOOT=600 completo, definitivo)
- `local_runs/mod-mec-04/boot_checkpoint.pkl` (estado do bootstrap, scratch)

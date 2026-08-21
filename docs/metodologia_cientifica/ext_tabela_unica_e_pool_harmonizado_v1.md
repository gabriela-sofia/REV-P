# Tabela única de pontos e o pool fluvial depois da harmonização completa

**Data**: 2026-08-13
**Status**: infraestrutura de dado fechada e testada; três achados novos, um deles
corrige a leitura de um resultado anterior
**Depende de**: `ext_resolucao_e_mecanismo_decisao_v1.md`,
`ext_hand_incomparavel_entre_regioes_v1.md`,
`ext_modelo_fluvial_multirregiao_v1.md`
**Fecha**: os Blocos 1, 2 e 3 do `CHECKLIST_template.md` §10; a pendência de fonte
de chuva declarada em `ext_resolucao_e_mecanismo_decisao_v1.md` §5

---

## 1. O que existia e o que faltava

O trabalho de 12/08 fechou a cadeia de terreno harmonizada e a separação por
mecanismo. O que ficou sem resolver era a etapa entre uma coisa e outra: as seis
fontes continuavam em seis pipelines com esquemas próprios, e o conjunto de
treino era montado por um script que sabia o formato de cada uma.

Isso funciona e não é auditável. Um erro nessa camada — uma coluna com dois
significados, um negativo promovido de nível, um ponto contado duas vezes — não
levanta exceção e atravessa o modelo inteiro. Já aconteceu três vezes neste
projeto, e as três estão documentadas: `hand_m` derivado por duas cadeias,
`hand_m` com limiar de canal diferente por região, e agora uma terceira, na
seção 4.

A resposta é estrutural: congelar o contrato antes de popular, traduzir cada
fonte isoladamente, e só então admitir, deduplicar e consolidar. São quatro
programas em vez de um, e a razão de serem quatro é que cada um responde por uma
pergunta diferente quando um número não bate.

| etapa | programa | o que entrega |
|---|---|---|
| contrato | `ds03_esquema_alvo.py` | 24 colunas, 7 domínios fechados, **zero linhas de dado** |
| tradução | `ds04_reduzir_por_fonte.py` | um arquivo por fonte, no contrato, nada descartado |
| admissão | `ds05_admissao_consolidacao.py` | tabela única com motivo nomeado por rejeição, hash, relatório de contagem |
| auditoria de terreno | `ter04_registro_auditoria_regional.py` | 124 derivações num registro só |
| auditoria de chuva | `aud_chuva01_fontes_incompativeis.py` | o achado da seção 4 |

A regra de ouro do contrato: **nenhuma coluna de valor existe sem a coluna de
procedência que diz de onde aquele valor veio.** Onde a fonte não tem a
variável, o campo é nulo declarado — nunca estimado, nunca imputado, nunca zero.

---

## 2. A tabela única

116.992 pontos de seis fontes, 33.349 admitidos, 33.071 elegíveis ao pool
fluvial. Nenhuma violação de contrato, nenhum `ponto_id` repetido, nenhuma
célula de ~11 m aparecendo em duas fontes.

A admissão não é um booleano só, e isso é decisão, não conveniência. Um critério
único obrigaria a escolher entre derrubar Recife — que é pluvial e entra a 10 m
por decisão declarada — ou deixar passar o CEMS em cadeia global, que é outro
instrumento. São dois portões com nomes diferentes:

- **`admitido`**: tem AOI, tem grupo de validação, tem cadeia de terreno
  declarada e não-global, e tem rótulo de treino
- **`elegivel_pool_fluvial`**: pode entrar no *mesmo ajuste* que as outras
  regiões — cadeia `wbt30`, mecanismo fluvial, as quatro variáveis presentes

Recife é admitido e não é elegível ao pool. Não é falha: é o resultado correto da
separação por mecanismo.

### Contagem por fonte — o entregável de E2

| fonte | entrada | admitido | pool | grupos | principal motivo de saída |
|---|---|---|---|---|---|
| sen1floods11 | 45.340 | 0 | 0 | 0 | cadeia global e sem declividade/TWI |
| cems | 36.418 | 23.915 | 23.915 | 119 | 12.503 em cadeia global; 9.577 água permanente |
| ufo | 25.800 | 0 | 0 | 0 | cadeia global e sem declividade/TWI |
| uk | 7.476 | 7.476 | 7.476 | 401 | — |
| curitiba | 1.680 | 1.680 | 1.680 | 1.471 | — |
| recife | 278 | 278 | 0 | 278 | pluvial: fora do pool por mecanismo |

O Sen1Floods11 e o UFO saem inteiros, e isso é conhecido e declarado desde o
`ds01`: declividade e TWI são derivadas direcionais que exigem grade local
reprojetada, não calculada para 661 chips em seis continentes. Continuam úteis
para auditar o critério de negativo por concordância espacial; não para treinar.

---

## 3. Duas fontes estavam fora do pool sem que isso fosse decisão

### 3.1 As 97 AOIs de planície

O `ter02` casava ponto e raster comparando o `grupo_cv` com a lista de AOIs
**íngremes** — porque a pergunta de origem era a analogia com Petrópolis. A
consequência não estava escrita em lugar nenhum: as 97 AOIs de planície ficavam
derivadas pelo `ter01` e invisíveis na sobreposição, e o pool fluvial rodava sem
elas.

Derivadas as 119 e re-extraídos os pontos, a planície passou de 1.680 para
28.684 pontos. O acordo entre a cadeia própria e o produto global, medido nos
23.915 pontos: elevação 0,998, HAND 0,944, declividade 0,913, TWI 0,607.

### 3.2 O piloto inglês

Mesma causa, chave diferente: no UK o `grupo_cv` é o **evento** (`EV_1.0`, 401
deles), não a AOI, então nenhum raster tinha aquele nome e a sobreposição nunca
casava. O piloto inglês — 7.476 pontos, a única fonte com validação prospectiva
de 25 anos — estava fora do conjunto que ela deveria validar.

Derivado o recorte de ~100 × 50 km numa AOI só (`ter01 --regiao`, 11 s) e
amostrados os pontos (`ter05`), entraram 7.476 de 7.476, nenhum fora do recorte.
Acordo com a cadeia anterior: elevação 0,999, HAND 0,893, declividade 0,799,
TWI 0,428.

> **Nota sobre o TWI, que se repete em todas as comparações**: 0,607 no externo,
> 0,428 no UK, 0,293 em Recife, 0,205 em Curitiba. TWI não sobrevive a troca de
> cadeia nem a troca de resolução. A regra já fixada — ou todo o TWI vem de 30 m,
> ou TWI sai do conjunto compartilhado — continua sendo a única leitura
> defensável.

### 3.3 Efeito no pool

| | mec01 (12/08) | mec02 (13/08) |
|---|---|---|
| pontos | 5.834 | 33.071 |
| grupos | 1.492 | 1.991 |
| fontes | 2 | 3 |
| planície | 1.680 | 28.684 |

---

## 4. Achado novo: a coluna de chuva de Recife carrega duas fontes, e a fonte prediz o rótulo


> **Nota de 20/08/2026.** A pendência de fonte de chuva descrita aqui foi
> resolvida: o `chuva02` (Recife, 16/08) e o `chuva04` (base inteira, 16/08)
> deixaram as seis fontes em Open-Meteo/ERA5-Land com a mesma fórmula. A
> limitação que restou é de escala, não de procedência — ver
> `ext_chuva_estado_do_projeto_v1.md`.

A pendência declarada dizia que Recife usava CHIRPS e Curitiba usava ERA5-Land.
A mistura não é entre as duas cidades. **É dentro de Recife.** O campo
`rain_data_source` do próprio v12 registra 181 pontos em CHIRPS v2.0 e 97 em
Open-Meteo ERA5-Land, na mesma coluna `rain_max_24h_chirps`, no mesmo modelo.

Isso não seria grave se a atribuição fosse aleatória. Não é:

| fonte de chuva | n | positivos | prevalência | mediana `rain_max_24h` |
|---|---|---|---|---|
| CHIRPS v2.0 | 181 | 145 | 0,80 | 39,7 mm |
| Open-Meteo ERA5-Land | 97 | 9 | 0,09 | 8,1 mm |

A causa é rastreável na procedência: a base canônica `v4_canonical_base` — 141
positivos — é 100% CHIRPS, e os 124 negativos `real_clean_sedec_negative` se
dividem em 36 CHIRPS e 88 ERA5. A fonte da chuva acompanha a **campanha de
aquisição**, e a campanha acompanha o rótulo.

### O que isso custa, medido

AUC por posto, IC95 por bootstrap de 2.000 reamostragens, semente fixa:

| o que discrimina | AUC | IC95 | n | pos |
|---|---|---|---|---|
| `rain_max_24h`, como o modelo usa | 0,7377 | [0,679, 0,796] | 269 | 145 |
| **só o indicador de qual produto mediu** | **0,8256** | **[0,782, 0,870]** | 278 | 154 |
| `rain_max_24h` dentro do estrato CHIRPS | 0,5985 | [0,510, 0,687] | 172 | 136 |
| `rain_max_24h` dentro do estrato ERA5 | 0,9167 | [0,852, 0,970] | 97 | 9 |

**O indicador de fonte discrimina melhor (0,826) do que a variável que ele
rotula (0,738)**, e dentro do maior estrato de fonte única a chuva cai para
0,599, com o IC quase encostando em 0,50.

### O que isto significa e o que não significa

Significa que a discriminação da coluna de chuva de Recife **não pode ser
atribuída a precipitação** enquanto a fonte não for unificada: parte dela é a
fonte. Como a trilha pluvial declara a chuva antecedente como preditor principal
— coeficiente +0,9896 com p < 1e-4, contra HAND em −0,0001 com p = 0,978 — é o
preditor principal que está afetado.

**Não** significa que o v12 esteja invalidado, e não significa que chuva não
importe em Recife. O v12 é um resultado interno a uma região e continua sendo o
que sempre foi. O estrato ERA5 dá 0,9167, mas com 9 positivos em 97 pontos — não
sustenta conclusão. E não há aqui nenhuma medida de qual das duas fontes é a
correta: as duas medem precipitação, e nenhuma foi auditada contra pluviômetro.

### O que fica pendente, e agora com prioridade

Reamostrar uma fonte única para os 278 pontos de Recife. É aquisição de dado,
não auditoria, e por isso não foi feita aqui — a auditoria precisava existir
antes para dizer se valia a pena. Vale.

---

## 5. Achado novo: o pool fluvial do mec01 incluía negativo que a hierarquia exclui

A hierarquia de negativo do `ds01` é explícita: negativo por **ausência** não
entra em modelo, porque ausência de registro não é registro de ausência. Os 442
negativos de Curitiba são ausência — existe chamado no 156 e o assunto não era
hidrológico. No mec01 isso passou porque o `tipo_negativo` vinha como texto cru
da fonte, sem normalização contra a hierarquia: a regra nunca chegou a ser
avaliada.

Por isso o `mod_mec02` roda as duas versões e reporta lado a lado:

| | ESTRITO | AMPLIADO |
|---|---|---|
| negativo por ausência | excluído | incluído (2,5% do negativo) |
| n | 32.629 | 33.071 |
| grupos | 1.565 | 1.991 |
| AUC_CV | 0,7436 | 0,7318 |
| gap | +0,0017 | +0,0018 |
| `hand_m` | −1,1455 [−2,091; −0,724] | −1,0929 [−1,983; −0,654] |
| `twi_dinf` | +0,4010 [+0,293; +0,477] | +0,4135 [+0,307; +0,492] |
| veredito | COERENTE_COM_CRITERIOS | COERENTE_COM_CRITERIOS |

**As duas concordam.** O resultado sobrevive à aplicação da regra: HAND negativo
e TWI positivo em ambas, nenhum IC cruzando zero, gap desprezível, e a diferença
de AUC entre as versões é de 0,012. O negativo por ausência não estava
carregando o resultado — mas agora isso é uma medida, e não uma suposição.

A composição do negativo, que a hierarquia obriga a reportar: 76,2% observado,
21,3% por exclusão qualificada, 2,5% por ausência.

---

## 6. A pergunta do mec01 não tinha a resposta que o número sugeria

O mec01 registrou que o modelo treinado nas AOIs tropicais e aplicado a Curitiba
caía para 0,4880, e deixou duas hipóteses em aberto, dizendo explicitamente que
não dava para escolher entre elas: **(i)** amostra pequena do lado planície, ou
**(ii)** incompatibilidade real entre serra tropical e planalto subtropical. O
registro dizia que harmonizar as 97 AOIs de planície era o que separava as duas.

Harmonizadas — e com o UK dentro do treino — o valor foi de 0,4880 para
**0,5004**. Pela regra escrita antes, isso decidiria por (ii).

**Não decide.** A regra estava incompleta, e a checagem que faltava é barata:
AUC de 0,50 num conjunto de teste tem duas causas que produzem o mesmo número e
levam a conclusões opostas — o modelo não transfere, ou o conjunto de teste não
tem contraste que modelo nenhum acharia. Medido dentro de Curitiba, feature por
feature isolada:

| feature | AUC bruto | separação | mediana pos | mediana neg | direção |
|---|---|---|---|---|---|
| `elevation_m` | 0,4070 | 0,186 | 904,0 | 911,0 | esperada |
| `slope_deg` | 0,4592 | 0,082 | 2,24 | 2,64 | esperada |
| `hand_m` | 0,4704 | 0,059 | 5,70 | 6,88 | esperada |
| `twi_dinf` | 0,4731 | 0,054 | 7,59 | 7,86 | **invertida** |

A melhor separação por feature isolada é 0,186, e HAND separa 0,059. **Não há
contraste em Curitiba para modelo nenhum encontrar**, e o negativo de lá é
ausência. O 0,50 é propriedade do rótulo de Curitiba, não evidência sobre
transferência entre climas. Isto é consistente com o que o `susc_20m` já havia
medido internamente: LOO-AUC 0,605 carregado por chuva antecedente, terreno
indeterminado.

Vale registrar o que o teste *sim* mostrou: aumentar a planície de 1.680 para
28.684 pontos não mudou nada, o que é o comportamento esperado quando o problema
está do lado do teste e não do treino.

Um detalhe que separa "não contrasta" de "contradiz": em três das quatro
features a direção é a esperada pelo modelo — positivos mais baixos e mais
planos. Curitiba não contradiz a relação terreno-inundação; ela tem pouco
contraste dela. O TWI é a exceção, e o TWI é justamente a variável que não
sobrevive a troca de cadeia em nenhuma região.

### O teste que efetivamente responde a pergunta de generalização

Transferência entre classes de relevo, medida em conjuntos que *têm* contraste:

| treino | teste | n teste | AUC |
|---|---|---|---|
| toda a planície | serra (INGREME) | 4.387 | **0,7885** |
| toda a serra | planície | 28.684 | **0,7017** |

Um modelo que nunca viu serra acerta 0,79 na serra; um que nunca viu planície
acerta 0,70 na planície. É o objetivo de transferência do planejamento —
terreno não representado no treino — medido em vez de suposto.

---

## 7. O que isto não é

Nada aqui é rótulo de referência, *ground truth* ou autorização de uso
preditivo. A tabela única é uma tabela de pontos harmonizada com procedência
declarada por linha. Os modelos do `mod_mec02` são ajustes com validação
agrupada sobre ela, sujeitos aos critérios fixados antes em
`ext_criterios_de_acerto_v1.md`, e nenhum deles foi validado operacionalmente.

Petrópolis continua com N = 0: tem derivação de terreno na mesma convenção das
outras 123, e nenhum ponto rotulado. Aparece no relatório como fonte de contagem
zero para que a ausência seja um número, e não um esquecimento.

---

## 8. Reprodução

```
python scripts/terreno/ter01_cadeia_harmonizada.py --lote todas --teto 600
python scripts/terreno/ter01_cadeia_harmonizada.py \
    --regiao uk_noroeste_harmonizado --bbox -2.7625,53.0466,-2.0020,53.9460
python scripts/terreno/ter02_reextrair_e_comparar.py --todas
python scripts/terreno/ter05_harmonizar_uk.py
python scripts/terreno/ter04_registro_auditoria_regional.py
python scripts/suscetibilidade/ds03_esquema_alvo.py
python scripts/suscetibilidade/ds04_reduzir_por_fonte.py
python scripts/suscetibilidade/ds05_admissao_consolidacao.py
python scripts/suscetibilidade/aud_chuva01_fontes_incompativeis.py
python scripts/suscetibilidade/mod_mec02_fluvial_pool_expandido.py
python -m pytest tests/test_ds03_ds05_tabela_unica.py -q
```

Saídas em `local_runs/`, que é git-ignored. O consolidado
(`ds-05-tabela-unica/tabela_unica_v1.csv`) tem sha256 no manifesto, e o teste
`test_consolidacao_e_deterministica` confere que rodar de novo dá o mesmo hash.

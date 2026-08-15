# Resolução única de 30 m em todo o projeto

**Data**: 2026-08-14
**Status**: decisão adotada; **substitui** a regra "a resolução segue o mecanismo"
de `ext_resolucao_e_mecanismo_decisao_v1.md` §3
**Não substitui**: a separação por mecanismo daquele documento, §4 e §5, que
continua valendo integralmente

---

## 1. O que muda, em uma frase

Toda linha canônica da base passa a ter as quatro variáveis de terreno
derivadas pela mesma cadeia — WhiteboxTools a 30 m, limiar de canal em
0,1123 km² de área contribuinte — sem exceção por região nem por mecanismo.

A regra anterior era defensável e não estava errada no que afirmava: Recife é
pluvial urbano, lá o terreno não é o mecanismo, e refinar a medida de uma
variável que o modelo declaradamente não usa seria esforço no lugar errado.
O que aquela regra não considerou é que **a decisão não é sobre Recife**.

## 2. Por que a regra anterior não se sustenta numa base replicável

Com duas resoluções convivendo, a coluna `hand_m` significa coisas diferentes
conforme a linha. Isso é exatamente o erro que esta cadeia inteira existe para
eliminar, e o projeto já o cometeu três vezes, todas documentadas:

1. `hand_m` derivado por duas cadeias — uma calculada, outra baixada pronta do
   produto global (`ext_cadeia_de_terreno_harmonizada_v1.md`)
2. `hand_m` com limiar de canal diferente em cada região, todos rotulados
   "percentil 98" (`ext_hand_incomparavel_entre_regioes_v1.md`)
3. `rain_max_24h_chirps` com CHIRPS em 181 pontos e ERA5-Land em 97, dentro da
   mesma região (`ext_tabela_unica_e_pool_harmonizado_v1.md` §4)

Nos três casos o erro não levantou exceção e atravessou análises. Manter uma
exceção declarada de resolução é manter aberta a mesma classe de falha, agora
com a agravante de ela estar documentada — o que protege quem leu o documento e
não protege o dado.

Há também uma razão positiva, e é a que dá nome a este trabalho: uma base de
conhecimento geoespacial replicável não pode exigir que quem a usa saiba, linha
a linha, qual instrumento produziu aquele número. A procedência tem de estar na
tabela, e o valor tem de ser comparável.

## 3. O custo, medido e não estimado

A troca incide onde já se sabia que incidiria. Medido nos próprios pontos
(`ter03` para o Brasil, `ter02` para o externo, `ter05` para o UK, `ter06`
para o Nível 1):

| região | elevação | HAND | declividade | TWI |
|---|---|---|---|---|
| Recife (10 m → 30 m) | 0,970 | 0,928 | **0,518** | **0,293** |
| Curitiba (10 m → 30 m) | 0,997 | 0,881 | 0,701 | **0,205** |
| CEMS, 119 AOIs (global → própria) | 0,998 | 0,944 | 0,907 | 0,603 |
| UK, piloto (global → própria) | 0,999 | 0,893 | 0,799 | 0,428 |
| Nível 1, 661 chips (global → própria) | 1,000 | 0,946 | — | — |

Em Recife a declividade mediana cai de 7,20° para 2,65°: a microtopografia de
planície costeira tem escala menor que a célula de 30 m, e isso é uma perda
real. Ela é aceitável por uma razão específica e não por conveniência — as duas
variáveis que se degradam são as duas que o modelo pluvial de Recife
declaradamente não usa. O HAND do v12 tem coeficiente −0,0001 com p = 0,978, e
o preditor daquele modelo é a chuva.

**A versão de 10 m não é descartada.** Sai em
`ds-04-reducao/recife__variante_nativa_10m.csv`, no mesmo esquema, e existe um
teste que falha se ela desaparecer ou se passar a ser idêntica à canônica. O que
muda é qual das duas é a canônica, não quantas existem.

## 4. O que isso destravou: Sen1Floods11 e UFO

O `ds01` declarou, e a declaração estava correta para a época, que estas duas
fontes entravam com as colunas de feature vazias de propósito: preencher
exigiria raster para 661 locais em seis continentes, "custo alto para um uso que
ainda não foi decidido".

O uso foi decidido, e com isso o custo deixou de ser especulativo e virou uma
conta: 661 chips de 3 a 5 km e **82 tiles DEM distintos**, porque os chips se
agrupam geograficamente muito mais do que a dispersão continental sugere. O
`ter06` derivou 658 de 661, sem nenhum estouro de tempo.

Elas ganham `slope_deg` e `twi_dinf`, que **nunca existiram** nessas fontes, e
passam a ter `hand_m` na mesma definição de canal das demais regiões. O acordo
com o produto global que elas usavam antes: elevação 1,000, HAND 0,946.

### O limite que essa escala impõe

HAND é a altura acima do canal mais próximo. Numa janela de ~9 km o canal
relevante pode estar fora do recorte, e a cadeia então encontra um canal que não
é o certo e devolve um HAND menor que o real. Isso não é hipotético em planície
de inundação, que é onde estas fontes amostram.

O `ter01` já tinha a guarda — rede com menos de 50 células de canal é reportada
como degenerada — e o `ter06` a lê e **anula o HAND** dos chips nessa condição,
em vez de deixar passar um número plausível. Nesta execução nenhum chip caiu na
condição, e o pearson de 0,946 contra o produto global é evidência independente
de que a janela não comprometeu a variável. O limite continua declarado porque
ele volta a valer se o conjunto de chips mudar.

## 5. Efeito na base

| | 12/08 (mec01) | 13/08 | 14/08 (resolução única) |
|---|---|---|---|
| pontos admitidos | — | 33.349 | **89.065** |
| pool fluvial | 5.834 | 33.071 | **64.989** |
| grupos no pool | 1.492 | 1.991 | 2.002 |
| fontes no pool | 2 | 3 | **4** |
| fontes em cadeia global | 3 | 3 | **0** |

Nenhuma fonte permanece integralmente em cadeia global. O que sobra fora da
cadeia própria são pontos individuais que caíram em nodata do raster derivado:
6.959 no CEMS (quase todos água permanente, que a máscara de mar anula por
definição), 2.002 no UFO e 706 no Sen1Floods11.

O UFO entra na base com as quatro variáveis e **continua fora dos modelos por
mecanismo**: a própria fonte declara cobrir drivers pluvial, fluvial e maré de
tempestade sem separação por chip. Harmonizar o terreno não muda o que a fonte
declara sobre si. Ele fica disponível como conjunto de robustez.

## 6. O que o pool maior mudou no modelo, e o que não mudou

Com 63.421 pontos e quatro fontes, contra 33.071 e três:

| | pool de 33 mil | pool de 63 mil |
|---|---|---|
| AUC_CV (ESTRITO) | 0,7436 | 0,7241 |
| AUC_CV (AMPLIADO) | 0,7318 | 0,7198 |
| `hand_m` | −1,1455 [−2,091; −0,724] | **−1,3636** [−2,434; −0,799] |
| `twi_dinf` | +0,4010 [+0,293; +0,477] | **+0,4888** [+0,397; +0,557] |
| `slope_deg` | −0,3083 [−0,483; −0,103] | **+0,0310 [−0,200; +0,215]** |
| `elevation_m` | +0,5394 [+0,411; +0,693] | +0,3618 [+0,277; +0,492] |

**Os dois termos causais ficaram mais fortes e a declividade deixou de ser
distinguível de zero.** HAND e TWI mantêm sinal e nenhum IC cruza zero, com
magnitudes maiores; a declividade inverte o sinal e o intervalo passa a conter
zero. Isso não reprova o modelo — os critérios fixados em
`ext_criterios_de_acerto_v1.md` exigem sinal apenas de `hand_m` e `twi_dinf`,
e ambos os vereditos permanecem COERENTE_COM_CRITERIOS.

A leitura honesta é que a declividade era um efeito de composição do conjunto
menor. Com quatro fontes e seis continentes, ela não sustenta um sinal próprio.

A transferência entre classes de relevo praticamente não se moveu, o que é o
melhor indício de estabilidade que este conjunto de testes produz:

| | pool de 33 mil | pool de 63 mil |
|---|---|---|
| treinar em planície → testar em serra | 0,7885 | 0,8018 |
| treinar em serra → testar em planície | 0,7017 | 0,6881 |

E o LOSO de Curitiba continua em 0,4997 com um treino que passou de 1.680 para
56.573 pontos de planície, em três fontes. A leitura registrada em
`ext_tabela_unica_e_pool_harmonizado_v1.md` §6 se mantém e fica mais forte: o
número é propriedade do rótulo de Curitiba, cuja melhor separação por feature
isolada é 0,186, e não evidência sobre transferência entre climas.

## 7. O que continua valendo da decisão anterior

Tudo, exceto §3. A separação por mecanismo não é afetada: Recife continua
PLUVIAL_URBANO e fora do pool fluvial, agora **exclusivamente** por mecanismo,
já que a cadeia deixou de ser um diferenciador. Isso é uma melhora de nitidez —
antes a exclusão de Recife tinha duas causas sobrepostas e agora tem uma só.

A pendência de chuva de §5 daquele documento continua aberta e continua sendo a
prioridade da trilha pluvial: unificar a fonte de precipitação dos 278 pontos de
Recife. A resolução do MDT saiu do caminho; a fonte da chuva não.

## 8. Reprodução

```
python scripts/terreno/ter01_cadeia_harmonizada.py --lote todas --teto 600
python scripts/terreno/ter01_cadeia_harmonizada.py \
    --regiao uk_noroeste_harmonizado --bbox -2.7625,53.0466,-2.0020,53.9460
python scripts/terreno/ter02_reextrair_e_comparar.py --todas
python scripts/terreno/ter05_harmonizar_uk.py
python scripts/terreno/ter06_harmonizar_chips_nivel1.py --teto 180
python scripts/terreno/ter04_registro_auditoria_regional.py
python scripts/suscetibilidade/ds03_esquema_alvo.py
python scripts/suscetibilidade/ds04_reduzir_por_fonte.py
python scripts/suscetibilidade/ds05_admissao_consolidacao.py
python scripts/suscetibilidade/mod_mec02_fluvial_pool_expandido.py
python -m pytest tests/test_ds03_ds05_tabela_unica.py -q
```

O `ter06` é o passo caro: 661 derivações, dezenas de minutos na primeira
execução, e reaproveita o que já existe nas seguintes.

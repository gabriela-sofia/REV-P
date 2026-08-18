# Cadeia de terreno harmonizada — replicação da auditoria do Recife

**Data**: 2026-08-12
**Artefatos**: `local_runs/ter-01-cadeia-harmonizada/`, `local_runs/ter-02-comparacao/`, `local_runs/mod-serra-02/`
**Scripts**: `scripts/terreno/ter01_cadeia_harmonizada.py`, `ter02_reextrair_e_comparar.py`, `scripts/suscetibilidade/mod_serra02_instrumento.py`

---

## 1. O problema

O projeto tinha duas cadeias de derivação produzindo colunas com o mesmo nome.

| | Recife (auditada, 27/07/2026) | análogos CEMS (`cems02`) |
|---|---|---|
| entrada | MDT real de 10 m | Copernicus GLO-30, 30 m |
| HAND | derivado: fill → D-inf → acumulação → canais → altura acima do canal | **não derivado**: produto global `glo-30-hand` |
| declividade | WhiteboxTools v2.4.0 | gradiente do numpy |
| TWI | SCA D-infinito + slope do WBT | pysheds, `tan(slope)` truncado em 0,05° |
| prova | r=1,0 e `max_abs_diff`=0,0 | nenhuma |

HAND é, literalmente, a altura acima do canal mais próximo. Trocar a definição
de canal troca a variável. `hand_m` numa tabela e `hand_m` na outra não eram a
mesma grandeza.

**Onde isso era fatal**: Petrópolis já tem HAND derivado pela cadeia
WhiteboxTools. Treinar no produto global e predizer no WBT é aplicar o modelo a
uma variável diferente da que ele aprendeu. Harmonizar não era refino — era
pré-condição da propagação.

---

## 2. A decisão de resolução

MDT de 10 m só existe no Brasil. Haiti, Equador, Honduras e Sri Lanka têm 30 m
no melhor caso. Como coeficiente só é comparável entre regiões derivadas do
mesmo modo, **a cadeia roda a 30 m em todas as regiões, inclusive em Recife**.

Recife já tem a versão de 10 m auditada. Derivá-la também a 30 m e comparar as
duas mede o efeito da resolução — transformando "limitação de resolução" de
ressalva em quantidade. Isso está pendente (seção 7).

---

## 3. O erro do limiar, e a correção

A primeira versão do `ter01` herdou o `stream_percentile = 98,0` do
`recife_validation`. Testada na EMSR789_AOI06, produziu HAND com mediana de
**70,7 m** contra 36,4 m do produto global, correlação de apenas 0,815.

A causa não era resolução. **O percentil é relativo às células da janela:**

| | células | resolução | área contribuinte |
|---|---|---|---|
| Recife | 1.122,7 | 10 m | **0,112 km²** |
| EMSR789_AOI06 | 826,1 | 30 m | **0,743 km²** |

Seis vezes e meia de diferença. O mesmo "percentil 98" define uma rede de
drenagem diferente em cada AOI, porque depende do tamanho e do relevo da
janela. Percentil serve para uma região única; para comparar regiões, não.

**Correção**: o limiar passa a ser fixado em **área contribuinte**, grandeza
física independente de resolução e de janela:

```
AREA_CANAL_KM2 = 0,1123      # exatamente o limiar efetivo do recife_validation
```

Isso preserva continuidade com a cadeia auditada e a torna transferível.

### O efeito da correção, medido

| HAND na EMSR789_AOI06 | mediana | positivo | negativo | contraste | pearson vs global |
|---|---|---|---|---|---|
| percentil 98 (errado) | 70,69 | 16,97 | 155,95 | +138,98 | **0,815** |
| **área 0,1123 km² (correto)** | **38,99** | **13,34** | **84,54** | **+71,20** | **0,953** |
| produto global (referência) | 36,36 | 9,25 | 72,97 | +63,72 | — |

Duas implementações independentes — a cadeia WhiteboxTools deste projeto e o
produto global GLO-30 HAND, feitos por equipes diferentes com código diferente
— passam a concordar a **0,953**, com medianas de 39,0 contra 36,4.

Isso é validação cruzada da derivação, não consistência interna. E é o que
apanhou o erro: rodar a cadeia sozinha e aceitar o resultado teria produzido um
HAND com o dobro do valor correto, sem nenhum sinal de alarme.

O percentil equivalente passa a ser gravado como **diagnóstico**: variou entre
90,5 e 95,5 nas 21 AOIs, contra 98 em Recife. A densidade de drenagem realmente
difere entre regiões — que é exatamente por que percentil não transfere.

---

## 4. Execução

21 das 22 AOIs íngremes derivadas. A restante (EMSR847_AOI15) falhou por
arquivo residual bloqueado no sistema de arquivos, não por erro de cadeia.

Cada AOI produz `run_manifest.json` com versão do WBT, CRS, shape, pixel,
parâmetros, limiar em células e em km², percentil equivalente, tempo e
**sha256 de cada saída**.

---

## 5. Concordância entre as cadeias, no ponto amostrado

A comparação relevante não é no raster inteiro — o modelo só vê os pontos
amostrados. Uma divergência grande no raster é irrelevante se ocorrer onde não
há ponto; uma pequena é decisiva se estiver no fundo de vale, onde estão os
positivos.

Mediana das 21 AOIs, 4.235 pontos:

| feature | pearson | mediana global | mediana WBT | contraste global | contraste WBT |
|---|---|---|---|---|---|
| `elevation_m` | **0,999** | 636,94 | 635,85 | 94,71 | 78,40 |
| `hand_m` | **0,935** | 16,18 | 14,98 | 24,82 | 29,57 |
| `slope_deg` | **0,905** | 11,01 | 10,02 | 5,60 | 6,70 |
| `twi_dinf` | **0,633** | 6,63 | 6,57 | −1,28 | −1,72 |

**TWI é a feature menos reprodutível entre implementações** — e é justamente a
que mais pesou no modelo de serra. Elevação é praticamente idêntica, como
esperado: as duas cadeias partem do mesmo DEM.

---

## 6. O teste de robustez: mesmos pontos, dois instrumentos

Mesmos 4.154 pontos, mesmos 21 grupos, mesma especificação, mesma semente. Só
muda o instrumento. Qualquer diferença é atribuível à cadeia de derivação.

| | AUC CV agrupada | `hand_m` | `twi_dinf` |
|---|---|---|---|
| **global** | 0,7573 | −0,8410 [−1,8849; −0,4171] | +0,4603 [+0,2881; +0,6062] |
| **WBT** | **0,7673** | −0,6572 [−1,8834; −0,2809] | +0,6663 [+0,4112; +0,8588] |

```
delta AUC (wbt - global) = +0,0100
hand_m     IC sobrepõe = sim
twi_dinf   IC sobrepõe = sim
robusto_ao_instrumento = SIM
```

Os dois passam nos critérios de `ext_criterios_de_acerto_v1.md`. Os sinais
concordam e os intervalos se sobrepõem nas duas features.

**A conclusão do `mod-serra-01` sobrevive à troca de instrumento** — apesar de
TWI concordar a apenas 0,633 entre as cadeias, a ordenação que o modelo usa é
preservada. Isso é um resultado de robustez, não uma formalidade: significa que
"em serra o TWI pesa mais que na planície" é fenômeno, não artefato de software.

### Critério de adoção, declarado antes de rodar

**A cadeia não foi escolhida por AUC.** Foi escolhida por comparabilidade com
Petrópolis, que já tem derivação WhiteboxTools. Um AUC maior na cadeia global
seria irrelevante, porque o modelo não poderia ser aplicado ao alvo sem trocar
de variável no meio.

Que a WBT também tenha dado AUC ligeiramente maior (+0,010) é conveniente, não
é o argumento.

---

## 7. O que fica pendente

- **Recife a 30 m.** Derivar pela mesma cadeia e comparar com a versão auditada
  de 10 m. Converte o efeito de resolução em número medido. É o passo de maior
  valor científico restante nesta frente.
- **EMSR847_AOI15.** Remover o resíduo e re-derivar; recupera a 22ª AOI.
- **Petrópolis pela cadeia harmonizada.** A derivação existente é de 2026-07 com
  parâmetros próprios; refazer com `AREA_CANAL_KM2 = 0,1123` a 30 m fecha a
  cadeia entre treino e alvo.
- **Planície pela cadeia WBT.** Hoje só as AOIs íngremes foram harmonizadas. A
  comparação serra × planície do `ext_modelo_de_encosta_v1.md` continua válida
  (ambos os lados usam a cadeia global), mas para juntar as duas num só modelo é
  preciso harmonizar as 97 AOIs de planície também.

---

## 8. Correção que isto impõe a documento anterior

`ext_criterios_de_acerto_v1.md`, seção 4.3, fixa a régua de contraste de HAND
em 2–3 m para negativo realista e acima de 4,5 m para negativo fabricado.

**Essa régua vale para planície.** Em terreno íngreme o contraste medido é de
+29,6 m (WBT) e continua fisicamente correto — a encosta sobe rápido, então um
ponto não inundado está dezenas de metros acima da drenagem. A régua precisa
ser lida por classe de relevo, e está registrada assim em
`ext_modelo_de_encosta_v1.md`, seção 3.

# Validação prospectiva, e por que Recife não pode ser somada ao conjunto externo

**Data**: 2026-08-12
**Artefatos**: `local_runs/mod-prosp-01/`, `local_runs/ter-03-brasil-harmonizado/`
**Scripts**: `mod_prosp01_holdout_temporal.py`, `ter03_reextrair_brasil.py`

> **Nota de 20/08/2026 — a seção 1 foi refeita e continua valendo, mas não é
> mais a medida corrente.** O MOD-PROSP-01 rodou na `ds-01`, base anterior à
> harmonização, ordenou blocos negativos por uma data que eles não têm, e
> reportou AUC sem intervalo de confiança. A execução que fecha o E4 é o
> MOD-PROSP-02, sobre a tabela única, com IC de grupos em todo fold:
> `ext_holdout_temporal_e4_v1.md`. O veredito não mudou
> (`PROSPECTIVAMENTE_ESTAVEL`); o que mudou foi a base em que ele se apoia —
> oito folds com prevalência de teste 0,46–0,50, em vez de nove com
> prevalência derivando de 0,30 a 0,93. As seções 2 a 5 deste documento
> seguem correntes.

---

## 1. A validação prospectiva — o teste que faltava

Toda validação do projeto até aqui foi por grupo (evento ou AOI). Isso trata
autocorrelação espacial, mas não responde à pergunta operacional: **um modelo
ajustado com o que já aconteceu acerta o que ainda vai acontecer?**

Janela expansiva sobre 328 eventos ingleses entre 2000 e 2025. Cada corte treina
em todos os eventos anteriores e testa nos 32 seguintes. Trava de EPV: só entra
fold com pelo menos 40 eventos de treino (10 × 4 features).

| corte | treino | AUC prospectivo | AUC treino | gap |
|---|---|---|---|---|
| 2000-10-29 | 40 ev | 0,7381 | 0,8264 | +0,088 |
| 2001-05-14 | 72 ev | 0,7014 | 0,7812 | +0,080 |
| 2002-08-10 | 104 ev | 0,7476 | 0,7832 | +0,036 |
| 2008-01-21 | 136 ev | 0,8217 | 0,7802 | −0,042 |
| 2012-06-22 | 168 ev | 0,7448 | 0,7855 | +0,041 |
| 2016-09-13 | 200 ev | 0,7989 | 0,7686 | −0,030 |
| 2020-10-06 | 232 ev | 0,7434 | 0,7733 | +0,030 |
| 2024-11-23 | 264 ev | 0,8546 | 0,7725 | −0,082 |
| 2025-01-01 | 296 ev | 0,7671 | 0,7828 | +0,016 |

```
AUC medio 0,7686   mediana 0,7476   min 0,7014   max 0,8546   desvio 0,0450
tendencia (corr entre ordem do corte e AUC) = +0,521
folds na faixa [0,70; 0,88] = 9/9      folds abaixo de 0,60 = 0/9
VEREDITO = PROSPECTIVAMENTE_ESTAVEL
```

**Um modelo treinado só com eventos de 2000 e 2001 acerta 0,70 em eventos que
aconteceram até 25 anos depois.** Nenhum fold degrada. A tendência é levemente
positiva, o que reflete mais amostra de treino, não deriva.

### O que isso resolve

Isso ataca diretamente o colapso temporal de Curitiba (AUC 0,6459 → 0,5246 em
holdout 2026), que sete diagnósticos internos não explicaram.

**O colapso não é propriedade do método.** Se fosse, a Inglaterra colapsaria
também, com o mesmo modelo, as mesmas features e um horizonte muito mais longo.
Não colapsa. Logo o problema está nos dados de Curitiba — na amostra, no rótulo
ou na definição do negativo daquele ano — e não na hipótese de que terreno
prediz inundação.

É também a primeira evidência de que a relação terreno-inundação **não caduca**:
o que o modelo aprende é geometria de bacia, e geometria de bacia muda em escala
geológica, não em escala de década.

---

## 2. O achado que reorganiza o conjunto multirregião

Re-extraindo os 278 pontos de Recife da cadeia harmonizada, fui verificar o
modelo v12 publicado. Os coeficientes são estes:

| feature | coef. padronizado | IC95 | p |
|---|---|---|---|
| `rain_decay_index_api_chirps` | **+0,9896** | [+0,613; +1,423] | **<0,0001** |
| `twi_dinf` | +0,2786 | [+0,005; +0,569] | 0,046 |
| `elevation_m` | +0,2662 | [−0,321; +0,872] | 0,374 |
| `slope_deg` | −0,1698 | [−0,457; +0,102] | 0,224 |
| `rain_peak_residual_orthogonalized` | −0,1402 | [−0,422; +0,163] | 0,347 |
| **`hand_m_dinf`** | **−0,0001** | [−0,597; +0,586] | **0,978** |

**O coeficiente de HAND no v12 de Recife é −0,0001, com p = 0,978.** É zero.
Não é fraco: é ausente. O LOO-AUC de 0,678 é carregado quase inteiramente pela
chuva antecedente.

E nos dados brutos o contraste está invertido em relação à física:

| | HAND mediana |
|---|---|
| negativos (n=124) | 5,250 m |
| positivos (n=154) | **8,071 m** |

Os pontos que inundaram estão **mais altos acima da drenagem** que os que não
inundaram. Isso permanece após a re-extração a 30 m (contraste −3,17 m).

### Por que, e por que isso não é um erro

Recife é enchente **pluvial urbana costeira** — a água não vem do canal subindo,
vem da chuva excedendo a capacidade de drenagem. HAND mede a lâmina necessária
para o canal alcançar o ponto. **Num evento pluvial, HAND não é o mecanismo.**

O conjunto externo é o oposto: são eventos fluviais e de enxurrada em vale, onde
a água vem do canal e HAND é exatamente o mecanismo — e lá o coeficiente é
−3,30 na planície e −0,66 na serra, com IC longe de zero.

Não são resultados contraditórios. **São fenômenos diferentes com o mesmo nome.**

### A consequência prática

Somar Recife ao conjunto externo assume um único processo gerador. Não é o caso.
Um modelo conjunto tentaria conciliar um subconjunto em que HAND não importa com
outro em que HAND é a variável principal, e o resultado não descreveria nenhum
dos dois.

O conjunto multirregião deve ser montado **por mecanismo**, não por país:

- **fluvial / enxurrada em vale** — todo o conjunto externo, Curitiba (fluvial
  urbano em planalto) e Petrópolis (enxurrada em serra). HAND é causal aqui.
- **pluvial urbano** — Recife. Modelo próprio, dominado por chuva, com HAND
  entrando como controle e não como preditor.

Isso não rebaixa Recife. Explica por que ele nunca respondeu bem às features
topográficas e por que o v12 depende tanto de chuva — e transforma uma anomalia
em classificação.

---

## 3. O custo de harmonizar a 30 m, medido

Re-extração dos pontos brasileiros dos rasters de 30 m contra os valores
originais de 10 m:

| | Recife (278) | Curitiba (1.680) |
|---|---|---|
| `elevation_m` | 0,970 | 0,997 |
| `hand_m_dinf` | **0,928** | **0,881** |
| `slope_deg` | **0,518** | 0,701 |
| `twi_dinf` | **0,293** | **0,205** |

Elevação e HAND sobrevivem à mudança de resolução. **Declividade e TWI não.**

Em Recife a declividade mediana cai de 7,20° para 2,65° — fator 2,7. Recife é
planície costeira: a microtopografia que organiza sua drenagem tem escala menor
que 30 m e simplesmente desaparece. Como TWI = ln(SCA/tan β) depende de
declividade e de área contribuinte, ele herda a perda e piora: 0,205 em Curitiba
é praticamente ausência de relação.

**Isto é um custo, não um detalhe.** A harmonização a 30 m compra
comparabilidade entre regiões e paga com o sinal de declividade e TWI nas
regiões brasileiras. Para HAND — que é a variável causal central — a troca vale.
Para TWI, não vale em terreno plano.

Recomendação que decorre: em modelo multirregião a 30 m, **TWI derivado de MDT
fino não deve ser misturado com TWI de 30 m**. Ou se usa 30 m em todos, ou TWI
sai do conjunto de features compartilhadas.

---

## 4. Estado dos cinco passos

| passo | estado |
|---|---|
| 1. Re-extrair 278 + 1.680 dos rasters harmonizados | **feito** (`ter-03-brasil-harmonizado/`) |
| 2. Harmonizar as 97 AOIs de planície | **em execução** — mecânico e longo; 22 íngremes + 3 brasileiras prontas |
| 3. Conjunto multirregião | **reformulado**: montar por mecanismo, não por país (seção 2) |
| 4. Validação prospectiva | **feita** — `PROSPECTIVAMENTE_ESTAVEL` (seção 1) |
| 5. Camada de explicação | pendente — o EBM do v12 existe e serve de base |

Comando do passo 2:

```
python scripts/terreno/ter01_cadeia_harmonizada.py --lote todas
```

É resumível: interromper e repetir continua de onde parou.

---

## 5. Limitações declaradas

**A herança de data.** No holdout, o negativo herda a data do seu próprio
evento. Sem isso, só 428 dos 3.738 negativos teriam data e o teste ficaria
desbalanceado por construção. A herança é defensável — o negativo foi amostrado
para aquele evento — mas é uma decisão, não um dado.

**Janela fixa de 32 eventos.** Folds mais antigos têm menos treino. A trava de
EPV descarta os cortes sem amostra, mas o primeiro fold ainda opera no limite.

**Só a Inglaterra.** A estabilidade prospectiva foi medida num único conjunto
regional. Ela refuta "o colapso é do método", mas não prova estabilidade em
clima tropical de serra.

**A fonte de chuva difere entre as regiões brasileiras.** ~~Recife usa CHIRPS;
Curitiba usa Open-Meteo ERA5-Land, apesar de as colunas terem sufixo `_chirps`.
Isso não foi resolvido e impede tratar as duas chuvas como a mesma variável.~~

**Resolvido em 16/08/2026 e verificado em 20/08.** O `chuva02` padronizou
Recife e o `chuva04` reextraiu toda a base: as seis fontes usam Open-Meteo/
ERA5-Land, mesma janela de 14 dias e mesmo decaimento de 0,85/dia, com testes
que guardam o invariante. Em troca, a limitação mudou de lugar: a chuva é
medida em células de ~11 km enquanto o modelo compara pontos dentro do mesmo
evento, e não discrimina nessa escala em nenhuma das seis fontes. Ver
`ext_chuva_estado_do_projeto_v1.md`.

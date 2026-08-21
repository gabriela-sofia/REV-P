# O modelo de encosta na tabela única — E3/M2

**Data**: 2026-08-20
**Artefatos**: `local_runs/mod-serra-03/`
**Script**: `scripts/suscetibilidade/mod_serra03_relevo_ds05.py`
**Substitui**: `ext_modelo_de_encosta_v1.md` (MOD-SERRA-01, base `ds-01`)
**Critérios**: fixados em `ext_criterios_de_acerto_v1.md` (09/08), antes desta rodada

---

## 1. O que muda em relação à v1

O MOD-SERRA-01 respondeu a pergunta certa — a relação HAND→inundação se comporta
em encosta como em planície? — mas na `ds-01`, base anterior à harmonização, e
só com as AOIs do Copernicus EMS. Esta rodada refaz o desenho na tabela única.

E, ao refazer, apareceu um problema de método que muda o que se pode afirmar.

**O MOD-SERRA-01 declarou EPV 11,0 dividindo 22 grupos por 2 variáveis.** Contar
assim ignora de qual classe são os grupos. No pool harmonizado o estrato íngreme
tem 24 grupos, mas **só 19 têm positivo**. Pela regra como Peduzzi a enunciou —
eventos da classe minoritária por variável — o orçamento é 19/10 = 1,9, isto é,
**uma variável, não duas**.

É a mesma correção que a trava do MOD-PROSP-02 aplicou ao holdout temporal
(`ext_holdout_temporal_e4_v1.md` §4), e ela não pode valer lá e não valer aqui.
Como isso mexe num resultado já publicado, esta rodada mede **as duas leituras**
em vez de escolher uma no escuro.

---

## 2. Orçamento de variáveis por estrato

| estrato | n | grupos | grupos + | grupos − | leitura precedente | leitura estrita |
|---|---:|---:|---:|---:|---:|---:|
| ÍNGREME | 5.162 | 24 | 19 | 24 | 2 variáveis | **1 variável** |
| PLANO_OU_ONDULADO | 58.006 | 1.666 | 1.334 | 420 | 166 | 42 |
| NÃO_CLASSIFICADO | 6 | 1 | 0 | 1 | — | — (fora: uma classe só) |

A planície não tem restrição prática. **O gargalo do E3 é o estrato íngreme**, e
ele é de 19 grupos positivos — todos estrangeiros, 22 AOIs do Copernicus EMS mais
Sen1Floods11.

---

## 3. Coeficientes por classe de relevo

Firth padronizado, `GroupKFold` por evento/AOI, IC95 por bootstrap de grupos
(N=1000, regra U2).

### ÍNGREME

| conjunto | cabe em | AUC agrupada | `hand_m` | `twi_dinf` | veredito |
|---|---|---:|---|---|---|
| **1 variável** | precedente **e estrita** | **0,7916** | −1,4423 [−3,1126; −0,8283] | — | `COERENTE_COM_CRITERIOS` |
| 2 variáveis | só precedente | 0,7847 | −0,8221 [−2,1830; −0,3593] | +0,7226 [+0,4643; +0,9028] | `COERENTE_COM_CRITERIOS` |

### PLANO_OU_ONDULADO

| conjunto | AUC agrupada | `hand_m` | `twi_dinf` | veredito |
|---|---:|---|---|---|
| 1 variável | 0,7121 | −2,2536 [−2,9506; −1,7001] | — | `COERENTE_COM_CRITERIOS` |
| 2 variáveis | 0,7131 | −1,6325 [−2,2171; −1,1973] | +0,3636 [+0,3028; +0,4101] | `COERENTE_COM_CRITERIOS` |
| 4 variáveis (terreno) | 0,7245 | −2,0985 [−2,7783; −1,5612] | +0,3991 [+0,3267; +0,4507] | `COERENTE_COM_CRITERIOS` |
| 6 variáveis (com chuva) | 0,7234 | −2,0912 [−2,7752; −1,5696] | +0,4006 [+0,3264; +0,4514] | `COERENTE_COM_CRITERIOS` |

**Todos os ajustes passam em todos os critérios fixados antes**: AUC dentro de
0,70–0,88, abaixo de 0,95, `hand_m` negativo, `twi_dinf` positivo, IC95 sem
cruzar zero.

**E o mais importante: a conclusão sobrevive à leitura estrita.** Com uma
variável só — a que o orçamento honesto permite — os dois estratos são coerentes,
e o de serra tem AUC mais alto que o de planície.

---

## 4. O que os coeficientes dizem

**O peso troca entre as duas variáveis, e o achado da v1 se confirma na base
harmonizada.** Com duas variáveis:

| | serra | planície | razão |
|---|---|---|---|
| `hand_m` | −0,8221 | **−1,6325** | planície pesa 2,0× |
| `twi_dinf` | **+0,7226** | +0,3636 | serra pesa 2,0× |

Na planície o modelo é dominado por HAND; na serra a decisão se reparte, e TWI
pesa o dobro. A leitura física é a mesma da v1: numa planície de inundação quase
tudo converge, então TWI carrega pouca informação e o que decide é a altura acima
da drenagem; numa serra, qual talvegue recebe a água importa tanto quanto quão
fundo ele é.

A magnitude mudou em relação à v1 (lá era −0,8173 na serra contra −3,2979 na
planície, razão 4×). A razão cai para 2× na base harmonizada — o achado
qualitativo se mantém, o quantitativo não é transportável entre as duas bases, e
é a base harmonizada que vale.

---

## 5. Transferência entre classes de relevo

| treinado em | avaliado em | 1 variável | 2 variáveis |
|---|---|---|---|
| **planície** | **serra** | **0,7957** [0,7433; 0,8460] | **0,8017** [0,7505; 0,8524] |
| serra | planície | 0,7322 [0,7106; 0,7521] | 0,6831 [0,6615; 0,7036] |

**O modelo treinado em planície, aplicado à serra, atinge 0,80 — acima do que o
próprio modelo da serra consegue em validação cruzada (0,78–0,79).** Ele nunca viu
terreno íngreme e funciona lá.

A explicação honesta não é que planície ensine melhor: é que a planície tem 58 mil
pontos em 1.666 grupos contra 5 mil em 24. O modelo está muito melhor estimado. E
o fato de transferir sem perda para um domínio que não viu é a evidência de que a
relação HAND/TWI→inundação é a mesma nos dois terrenos.

O caminho inverso perde, e perde mais com duas variáveis (0,6831) do que com uma
(0,7322) — coerente com a mesma explicação: acrescentar variável a um modelo mal
estimado piora a transferência.

---

## 6. O que isto fecha, e o que não fecha

**Fecha o entregável do E3/M2**: tabelas de coeficientes e de desempenho por
classe de relevo, sobre a base congelada, com validação agrupada e aplicação
cruzada. Os artefatos estão em `local_runs/mod-serra-03/`.

**Não fecha a aplicação cruzada entre fontes de negativo**, que o texto do E3
também menciona — isso está no `mod-mec-03` (`leave_one_source_out`) e continua
válido lá; não foi refeito aqui porque não é por classe de relevo.

**Não resolve Petrópolis.** Os 19 grupos positivos íngremes são todos
estrangeiros, e Petrópolis não tem nenhuma linha na tabela única. O modelo de
serra existe e é coerente; o que falta é a região ter variáveis físicas
extraídas — e depois, validação.

---

## 7. Recomendação sobre a leitura da regra de EPV

**Adotar a leitura estrita como padrão do projeto**, pelos motivos:

1. É a regra como a literatura a enuncia — eventos da classe rara por variável.
2. Já está aplicada no E4 (`mod_prosp02`), onde derrubou folds degenerados que a
   contagem por grupos totais deixava passar. Ter duas leituras vigentes ao mesmo
   tempo é o tipo de inconsistência que produz número defensável em um lugar e
   indefensável em outro.
3. Neste caso ela **não custa a conclusão**: com uma variável, serra e planície
   passam em todos os critérios e a transferência se mantém.

O que muda ao adotá-la: o ajuste de serra com duas variáveis passa a ser
reportado como fora do orçamento — ainda publicável, com a ressalva escrita, mas
não como resultado principal. E o texto do manuscrito que cita coeficientes de
serra e planície precisa passar a citar os desta rodada.

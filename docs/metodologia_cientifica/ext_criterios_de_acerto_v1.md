# O que é acertar — critérios de leitura do modelo

**Data**: 2026-08-09
**Propósito**: fixar, ANTES de qualquer rodada nova, quais números indicam que
o modelo aprendeu o fenômeno e quais indicam que ele aprendeu outra coisa.

---

## 1. O fenômeno, em uma frase

Uma enchente acontece quando chega mais água num ponto do que ele consegue
escoar ou armazenar. Isso decompõe em três termos, e as seis features do v12
não são uma lista arbitrária — são exatamente esses três:

| Termo físico | Feature | O que representa |
|---|---|---|
| **Quanto chega** | `rain_max_24h`, `rain_decay_index_api` | o pulso e o estado antecedente do solo |
| **Para onde converge** | `twi_dinf` | área de contribuição a montante por unidade de contorno |
| **Quanto sobe até alcançar** | `hand_m` | altura do ponto acima da drenagem que o serve |

`elevation_m` e `slope_deg` não são termos do balanço. São **descritores de
contexto** que entram como controle. Essa distinção não é estilística: é ela
que explica por que os dois trocam de sinal entre regiões e HAND não.

### Por que HAND é a variável central

HAND é a diferença de elevação entre um ponto e o ponto da rede de drenagem que
o drena. Sua interpretação é direta: **é a lâmina d'água necessária para que
aquele ponto seja alcançado.** Um ponto com HAND de 0,8 m inunda quando o curso
sobe 0,8 m; um com HAND de 6 m precisa de uma cheia seis vezes maior.

A literatura de mapeamento de inundação usa HAND com limiares de **2, 4 e 10 m**
para delinear planície de inundação. Isso dá uma régua física para ler nossos
números, e não uma régua estatística.

---

## 2. O que os nossos dados dizem hoje

Medianas de `hand_m`, por fonte e por classe:

| | positivo | negativo | interpretação física |
|---|---|---|---|
| Inglaterra (exclusão) | **0,76 m** | **5,65 m** | positivo dentro da planície; negativo **fora** dela |
| CEMS (observação) | **0,35 m** | **2,50 m** | positivo dentro; negativo **na borda** |

Os positivos das duas fontes estão **abaixo do limiar de 2 m** — ou seja, dentro
da planície de inundação como a literatura a define. Isso é a confirmação mais
forte de que o rótulo positivo está certo: não foi imposto, saiu do dado, e cai
exatamente onde a hidrologia prevê.

**A diferença entre os negativos é o achado.** O negativo observado (2,50 m)
está no limiar de 2–4 m, isto é, **na borda da planície**: área que poderia ter
inundado numa cheia maior e não inundou naquela. O negativo por exclusão
(5,65 m) está acima de 4 m, **fora da planície por construção** — o buffer de
400 m do critério N3 o empurrou para lá.

É por isso que o coeficiente de HAND infla de −1,28 para −2,61 quando se troca
uma definição pela outra. O modelo treinado com exclusão não aprendeu melhor;
aprendeu um problema mais fácil.

---

## 3. A régua da literatura

O que é AUC defensável em suscetibilidade a inundação:

- Estudos com validação adequada ficam **acima de 0,80**; conjuntos por
  ensemble chegam a 0,94 ou mais.
- **Divisão aleatória 70/30 sem bloco espacial infla o AUC em 5–15%** por
  autocorrelação espacial.
- Trabalhos que reportam **0,95–0,99 com divisão aleatória devem ser lidos com
  cautela**; o avanço aparente da última década é parcialmente ilusório.
- Um benchmark realista com validação cruzada espacial fica em torno de
  **0,881** para ensemble.

### Onde nos colocamos

```
MOD-UK-01   AUC = 0,7999   GroupKFold por evento   201 eventos   6 features
```

Isso está **abaixo** dos 0,88 do benchmark — e isso é esperado, por dois
motivos legítimos:

Primeiro, o benchmark de 0,881 é de *ensemble* (empilhamento de modelos não
lineares). O nosso é regressão logística de Firth, linear e interpretável por
escolha declarada do projeto. Comparar os dois seria comparar categorias
diferentes.

Segundo, e mais importante: nossa validação é **agrupada por evento**, não por
bloco espacial nem aleatória. É mais dura que a régua. Se rodássemos divisão
aleatória, o mesmo modelo provavelmente passaria de 0,90 — e esse número seria
inflado exatamente pelo mecanismo que a literatura denuncia.

**Conclusão: 0,7999 com validação agrupada é um número saudável.** Um AUC de
0,95 neste desenho seria motivo de suspeita, não de comemoração.

---

## 4. Faixas de aceitação — fixadas antes de rodar

Estas faixas passam a valer como critério de leitura de qualquer rodada futura.
Um resultado fora delas exige investigação do pipeline antes de qualquer
interpretação científica.

### 4.1. Desempenho

| AUC com CV agrupada | Leitura |
|---|---|
| < 0,60 | feature quebrada, rótulo invertido ou sinal ausente |
| 0,60 – 0,70 | fraco; verificar cobertura de features e definição de negativo |
| **0,70 – 0,88** | **faixa esperada para modelo linear com validação agrupada** |
| 0,88 – 0,95 | possível, mas exige justificar por que superou o benchmark |
| > 0,95 | **suspeita de vazamento**: grupo mal definido, rótulo circular ou negativo fabricado |

### 4.2. Sinais físicos obrigatórios

Estes não são preferências. Um modelo que os viole está errado, tenha o AUC que
tiver:

| Feature | Sinal exigido | Razão |
|---|---|---|
| `hand_m` | **negativo** | mais alto acima da drenagem, menos inunda |
| `twi_dinf` | **positivo** | mais convergência, mais água chega |
| `rain_max_24h` | **positivo** | mais chuva, mais inundação |
| `rain_decay_index_api` | **positivo** | solo mais saturado, menos infiltra |
| `elevation_m` | livre | descritor de contexto, pode inverter |
| `slope_deg` | livre | idem; ver 4.4 |

Estado atual do MOD-UK-01: **os quatro sinais obrigatórios estão corretos.**

### 4.3. Magnitude de HAND — a régua física, **por classe de relevo**

> **Correção de 2026-08-12.** A régua abaixo foi fixada com dados de planície e
> foi indevidamente escrita como se fosse universal. Em terreno íngreme os
> valores corretos são uma ordem de grandeza maiores. Ver
> `ext_modelo_de_encosta_v1.md` seção 3 e `ext_cadeia_de_terreno_harmonizada_v1.md`.

**Em planície** (relevo local < 400 m), contraste negativo-positivo:

- entre **2 e 3 m** → negativo na borda da planície, situação realista;
- **acima de 4,5 m** → negativo fora da planície, provavelmente fabricado por
  buffer ou por ausência de registro.

Medido: observação +2,15 m (saudável), exclusão +4,89 m (**inflado**).

**Em serra** (relevo local ≥ 400 m e ≥ 25% da área acima de 15°):

- contraste de **20 a 35 m** é o esperado e fisicamente correto. A encosta sobe
  rápido, então um ponto não inundado está dezenas de metros acima da drenagem.

Medido nas 21 AOIs íngremes: **+29,6 m** pela cadeia WhiteboxTools, +24,8 m
pelo produto global. Na EMSR789_AOI06 isoladamente, +71,2 m.

Aplicar a régua de planície em serra levaria a rejeitar como "negativo
fabricado" um contraste que é apenas geometria de encosta.

### 4.4. O sinal de `slope_deg` e a inversão condicional

No univariado, os positivos ingleses são mais *planos* que os negativos
(2,29° contra 2,80°). No multivariado, `slope_deg` sai **positivo**.

Isso não é erro. É associação condicional: uma vez controlados HAND e elevação,
o que resta da declividade capta convergência local de encosta — água que desce
rápido e se acumula no fundo do vale. Fenômeno real, mas que **precisa de
parágrafo próprio no artigo**, porque um leitor desatento lê "declividade maior
aumenta enchente" e conclui que o modelo está errado.

### 4.5. Sinais de que o modelo aprendeu o problema errado

- gap treino-validação acima de **0,15** → sobreajuste
- qualquer fold com AUC ≥ 0,999 → partição degenerada
- IC de HAND cruzando zero → sinal não estabelecido
- AUC alto com HAND fraco → o modelo está usando contexto, não física
- **transferência que melhora ao sair do domínio** → o domínio de origem era
  artificialmente fácil (foi o que vimos: observação→exclusão sobe 0,116)

---

## 5. Diagnóstico dos problemas atuais

**O que está certo.** Os quatro sinais físicos, os positivos dentro da planície
nas duas fontes, o gap treino-validação de +0,009, o EPV de 33,5, os quatro
intervalos de confiança sem cruzar zero.

**O que está inflado.** O negativo por exclusão, e por consequência o
coeficiente de HAND do modelo inglês. O AUC de 0,7812 daquele modelo é
otimista; o valor honesto para o mesmo modelo em dado real é o **0,6222** da
aplicação cruzada.

**O que ainda não sabemos.** Se o modelo funciona em clima tropical úmido de
baixa latitude com relevo de serra — porque o único conjunto que testaria isso
(Petrópolis) não tem rótulo, e os análogos tropicais que temos ou são planície
(Vietnã, Equador) ou têm pouquíssimo movimento de massa.

---

## 6. Petrópolis — predição e validação são coisas diferentes

Houve um erro meu de enquadramento que precisa ser corrigido, porque ele
travava uma rota que está aberta.

Eu vinha afirmando que "o inventário CPRM/DRM-RJ é a única rota para
Petrópolis". Isso confunde duas coisas:

**Para PREDIZER em Petrópolis não falta nada.** Todas as seis features são
globais: Copernicus DEM, HAND global de 30 m, TWI derivável, CHIRPS v3. O
modelo treinado nos análogos pode ser aplicado a Petrópolis hoje, e produzir um
mapa de suscetibilidade. É exatamente para isso que os análogos por relevo
foram construídos, e você está certa: essa capacidade já existe.

**Para VALIDAR em Petrópolis falta o inventário.** Sem rótulo local não há como
medir se a predição acertou. Nenhum análogo resolve isso — validação exige
observação no lugar validado, por definição.

A distinção muda o que se pode escrever no artigo:

- ✅ "O modelo, treinado em N regiões com negativo observado, produz o seguinte
  mapa de suscetibilidade para Petrópolis" — legítimo, com a ressalva de que a
  transferibilidade foi medida entre as regiões de treino, não para esta.
- ❌ "O modelo acerta X% em Petrópolis" — não verificável.

O documento `ext_balanco_e_lacunas_por_regiao_v1.md` afirma que o CEMS "não
resolve Petrópolis". A afirmação correta é: **não resolve a validação de
Petrópolis; a predição está disponível.**

O pedido LAI continua valendo, mas deixa de ser bloqueador. Ele passa a ser o
que transforma uma predição declarada em predição verificada — e isso pode
acontecer depois da entrega.

---

## 7. O que precisa acontecer para dizer que o modelo aprendeu

Em ordem de força probatória:

1. **Sinais físicos corretos** — já temos.
2. **AUC na faixa 0,70–0,88 com CV agrupada** — já temos (0,7999).
3. **IC sem cruzar zero nas features causais** — temos para as 4 topográficas;
   falta o bootstrap das 6.
4. **Transferência entre regiões sem colapso** — o modelo de observação
   transfere a 0,7798. Este é o teste mais forte que temos.
5. **Holdout temporal com dezenas de eventos** — **feito** (20/08/2026,
   `ext_holdout_temporal_e4_v1.md`). São 201 eventos ingleses em **110 datas
   distintas** entre 2000 e 2025 — o "~158" desta lista era estimativa e
   nunca se confirmou. Janela expansiva sobre a tabela única, 8 folds com IC
   de grupos, AUC médio 0,7890, nenhum fold abaixo de 0,60:
   `PROSPECTIVAMENTE_ESTAVEL`. Ataca o colapso de Curitiba pelo lado do
   método e o descarta como explicação; não explica o dado de Curitiba.
6. **Validação em região tropical de serra** — não disponível hoje.

O item 5 deixou de ser o próximo passo. O de maior valor científico agora é o
item 6, e ele **depende de aquisição** — é o pedido LAI de Petrópolis.

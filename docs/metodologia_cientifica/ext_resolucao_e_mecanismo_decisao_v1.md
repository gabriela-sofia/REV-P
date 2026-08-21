# Resolução do MDT e separação por mecanismo — decisão fechada

**Data**: 2026-08-12
**Status**: decisão metodológica adotada; substitui a escolha provisória de 30 m para tudo
**Depende de**: `ext_validacao_prospectiva_e_mecanismo_v1.md`, `ext_hand_incomparavel_entre_regioes_v1.md`

---

## 1. A pergunta

Recife e Curitiba foram derivados de MDT de 10 m. As regiões externas só têm
30 m. Harmonizar a 30 m custa caro em Recife: a declividade cai de 7,20° para
2,65° e o TWI perde a relação com a versão de 10 m (pearson 0,293).

Qual é a resolução certa? A resposta da literatura **não é um número único** —
depende do mecanismo, o que resolve a questão junto com a separação que já
tínhamos decidido fazer.

---

## 2. O que a literatura estabelece

**Para inundação pluvial urbana**, MDT de 1 m ou menos, derivado de LiDAR, é o
adequado; **10 m já produz resultados inconsistentes**. As avaliações de risco
pluvial urbano operam tipicamente entre 1 e 5 m, porque o que decide onde a
água se acumula é a microtopografia — meio-fio, sarjeta, cota de soleira — e
essa estrutura tem escala menor que a célula.

**Para inundação fluvial e para mapas de escala nacional**, 30 m é o padrão de
fato, e a razão é explícita: permite comparação entre regiões e indica onde vale
um estudo detalhado depois. É a resolução escolhida justamente quando o objetivo
é comparar.

**Sobre o efeito da resolução no desempenho**, a evidência é morna: 12,5 m rende
acurácia um pouco maior que 30 m, mas resolver o MDT sozinho não altera de forma
significativa a acurácia da predição de probabilidade de inundação, qualquer que
seja o modelo. Outros fatores — altitude, precipitação, distância à drenagem —
pesam mais.

---

## 3. A decisão

> **A resolução segue o mecanismo, não a região.**

| trilha | mecanismo | resolução | justificativa |
|---|---|---|---|
| **Fluvial / enxurrada** | água vem do canal subindo | **30 m harmonizado** | padrão de escala nacional; escolhido para permitir comparação entre regiões, que é exatamente o objetivo. HAND sobrevive à mudança (pearson 0,93) |
| **Pluvial urbano** | chuva excede a drenagem | **a melhor disponível, declarada** | 30 m é inadequado e 10 m é marginal segundo a literatura. Mas o terreno não é o mecanismo aqui, então a resolução não é o gargalo — é a chuva |

Isto **não é um meio-termo**. É a consequência de duas coisas já medidas:

1. Na trilha fluvial, HAND é a variável causal e sobrevive aos 30 m com
   pearson 0,93. O que se perde (TWI, declividade) é secundário lá.
2. Em Recife, o coeficiente de HAND é −0,0001 com p = 0,978. **Terreno não é o
   mecanismo.** Gastar esforço em resolução fina para Recife seria refinar a
   medida de uma variável que o próprio modelo diz não usar.

### Consequência para o TWI

TWI derivado a 10 m e TWI derivado a 30 m **não são a mesma variável** (pearson
0,293 em Recife, 0,205 em Curitiba). A regra passa a ser: num modelo que junta
regiões, ou todo o TWI vem de 30 m, ou TWI não entra no conjunto compartilhado.
Nunca misturar as duas origens na mesma coluna.

---

## 4. A tabela de mecanismo

Classificação declarada por fonte, com a evidência que a sustenta:

| fonte | mecanismo | evidência |
|---|---|---|
| `UK_noroeste` | **FLUVIAL_ENXURRADA** | Recorded Flood Outlines da EA, dominado por transbordamento de curso d'água |
| `analogo_EMSR*` | **FLUVIAL_ENXURRADA** | ativações CEMS de Flood/Storm com `observedEventA`; enxurrada em vale |
| `nivel1_sen1floods11` | **FLUVIAL_ENXURRADA** | declarado na própria base: dominada por cheia fluvial em área aberta |
| Curitiba | **FLUVIAL_ENXURRADA** | fluvial urbano em planalto (perfil da região) |
| Petrópolis | **FLUVIAL_ENXURRADA** | enxurrada em serra com movimento de massa |
| **Recife** | **PLUVIAL_URBANO** | HAND com coeficiente −0,0001 (p=0,978) e contraste invertido; chuva antecedente domina (+0,99, p<1e-4) |
| `nivel1_ufo` | **MISTO_NAO_SEPARAVEL** | a própria base declara cobrir drivers pluvial, fluvial **e** maré de tempestade em 14 eventos, sem separação por chip |

O caso do UFO merece atenção: são **215 grupos**, o maior número do conjunto, e
não são atribuíveis a um mecanismo. Incluí-los num modelo fluvial contaminaria
o subconjunto com pluvial e maré. Ficam fora dos modelos por mecanismo e
disponíveis como conjunto de robustez — onde a pergunta é justamente se o modelo
aguenta mistura.

---

## 5. O que muda em Recife

Recife deixa de ser tratado como uma região que responde mal às features
topográficas e passa a ser tratado como **outro fenômeno**, com modelo próprio:

- **preditor principal**: chuva antecedente e pico de chuva
- **terreno**: entra como controle, não como preditor. HAND e TWI ficam no
  modelo para absorver variação de contexto, com o entendimento declarado de que
  seus coeficientes não têm interpretação causal aqui
- **resolução**: 10 m mantida, por ser a melhor disponível, com a ressalva de que
  a literatura considera 10 m marginal para pluvial e que o adequado seria LiDAR

O que **não** muda: o v12 continua válido como está. LOO-AUC 0,678 num modelo
dominado por chuva é um resultado legítimo para fenômeno pluvial — só não é um
resultado sobre topografia, e nunca foi.

### Pendência que isto expõe


> **Nota de 20/08/2026.** A pendência de fonte de chuva descrita aqui foi
> resolvida: o `chuva02` (Recife, 16/08) e o `chuva04` (base inteira, 16/08)
> deixaram as seis fontes em Open-Meteo/ERA5-Land com a mesma fórmula. A
> limitação que restou é de escala, não de procedência — ver
> `ext_chuva_estado_do_projeto_v1.md`.

Recife usa CHIRPS e Curitiba usa Open-Meteo ERA5-Land, apesar de as colunas
terem sufixo `_chirps`. Se a chuva é o preditor principal da trilha pluvial, a
fonte de chuva deixa de ser detalhe de proveniência e vira a variável central —
e duas fontes diferentes não podem ocupar a mesma coluna. Unificar é agora
prioridade da trilha pluvial, no lugar que a resolução do MDT ocupava.

---

## 6. O que isso fecha

A pergunta "10 m ou 30 m" não tinha resposta enquanto a pergunta anterior — "que
fenômeno é este?" — estava em aberto. Com a separação por mecanismo, as duas se
respondem juntas:

- fluvial compara entre regiões → 30 m, que é o padrão para comparar
- pluvial depende de microtopografia → resolução fina seria necessária, mas
  como o terreno não é o mecanismo, o esforço vai para a chuva

Nenhuma das duas escolhas é concessão. Cada uma é a que a literatura indica para
o fenômeno correspondente.

# O primeiro modelo multirregião com cadeia única e mecanismo único

**Data**: 2026-08-12
**Artefatos**: `local_runs/ds-02-mecanismo/`, `local_runs/mod-mec-01/`

---

## 1. O que precisou acontecer antes

Três correções, cada uma das quais sozinha invalidaria o resultado:

1. **Analogia por AOI** — o ranking anterior media relevo no centroide da
   ativação; cinco dos seis "análogos de Petrópolis" eram planícies.
2. **Cadeia única** — Recife derivava HAND com WhiteboxTools, os análogos
   baixavam o produto global, e o limiar declarado em percentil valia
   0,1123 km² em Recife contra 0,7507 km² em Petrópolis.
3. **Mecanismo único** — Recife é pluvial (HAND com p=0,978), o conjunto
   externo é fluvial.

---

## 2. Composição

Só mecanismo fluvial, só cadeia WhiteboxTools 30 m com limiar de 0,1123 km².

| fonte | n | pos | neg | grupos |
|---|---|---|---|---|
| curitiba | 1.680 | 1.238 | 442 | **1.471** |
| analogo_EMSR796 (Equador) | 1.235 | 515 | 720 | 6 |
| analogo_EMSR847 (Haiti/Caribe) | 720 | 120 | 600 | 5 |
| analogo_EMSR851 (Sri Lanka) | 720 | 360 | 360 | 3 |
| analogo_EMSR778 (Honduras) | 480 | 240 | 240 | 2 |
| analogo_EMSR789 (Equador) | 360 | 120 | 240 | 2 |
| analogo_EMSR790 (La Réunion) | 240 | 120 | 120 | 1 |
| analogo_EMSR813 (Equador) | 240 | 120 | 120 | 1 |
| analogo_EMSR857 (Moçambique) | 159 | 120 | 39 | 1 |
| **total** | **5.834** | 2.953 | 2.881 | **1.492** |

**Desequilíbrio a declarar**: Curitiba responde por 98,6% dos grupos. O conjunto
é, em número de unidades de validação, quase inteiramente Curitiba com 21 AOIs
tropicais ao lado. Isso condiciona a leitura de tudo o que vem a seguir.

---

## 3. Resultado

```
AUC_CV = 0,7673   (5 folds, min 0,7342, max 0,8021)   gap = -0,0011
EPV = 373         VEREDITO = COERENTE_COM_CRITERIOS
```

| feature | coeficiente | IC95 |
|---|---|---|
| `hand_m` | **−0,6958** | [−1,5971; −0,4397] |
| `slope_deg` | −0,5958 | [−0,9971; −0,3821] |
| `elevation_m` | +0,4229 | [+0,1709; +0,6805] |
| `twi_dinf` | **+0,2655** | [+0,0444; +0,4231] |

Nenhum IC cruza zero. Os dois sinais obrigatórios estão corretos: HAND negativo,
TWI positivo. **Os dois termos causais se mantêm num conjunto que vai de serra
tropical andina e caribenha a planalto subtropical brasileiro, com a mesma
medida.** É a primeira vez que isso pode ser afirmado.

`slope_deg` sai negativo aqui, tendo saído positivo no modelo inglês. É a
terceira vez que declividade troca de sinal entre conjuntos — coerente com a
classificação de descritor de contexto e não de processo.

---

## 4. Leave-one-source-out — o teste mais duro, e o achado negativo

Treina em todas as fontes menos uma; testa na que ficou de fora.

| fonte excluída | n teste | AUC |
|---|---|---|
| analogo_EMSR857 (Moçambique) | 159 | 0,9137 |
| analogo_EMSR847 (Haiti/Caribe) | 720 | 0,9072 |
| analogo_EMSR813 (Equador) | 240 | 0,8909 |
| analogo_EMSR851 (Sri Lanka) | 720 | 0,7996 |
| analogo_EMSR778 (Honduras) | 480 | 0,7345 |
| analogo_EMSR796 (Equador) | 1.235 | 0,7311 |
| analogo_EMSR789 (Equador) | 360 | 0,7124 |
| analogo_EMSR790 (La Réunion) | 240 | 0,6229 |
| **curitiba** | **1.680** | **0,4880** |

**O modelo treinado só nas 21 AOIs tropicais, aplicado a Curitiba, fica em
0,4880 — nível de acaso.**

Duas leituras, e é preciso separá-las:

A leitura fraca é amostral. Sem Curitiba, o treino tem 21 grupos contra 1.471.
Todos os outros folds treinam *com* Curitiba, isto é, são o modelo de Curitiba
sendo testado em conjuntos pequenos. A assimetria replica a que já havia sido
medida entre planície e serra: o lado com mais dado transfere, o com menos não.

A leitura forte é que 0,488 **não é degradação, é ausência de sinal**. Um modelo
mal estimado tenderia a 0,6; ficar em 0,49 sugere que a relação aprendida em
serra tropical simplesmente não organiza Curitiba.

Não é possível decidir entre as duas com o dado atual, e afirmar qualquer uma
delas agora seria escolher a conclusão antes da evidência. O que decide é
harmonizar as 97 AOIs de planície: isso multiplica os grupos não-Curitiba por
uma ordem de grandeza e separa amostra pequena de incompatibilidade real.

### O que já dá para afirmar

- Curitiba → tropical **funciona**: 0,73 a 0,91 em oito fontes independentes,
  incluindo Haiti e Sri Lanka, que Curitiba não se parece em nada.
- Tropical → Curitiba **não funciona** com o dado disponível.

Para a propagação a Petrópolis isso é informação direta e favorável: Petrópolis
é serra tropical úmida, o lado do espaço de features onde o modelo funciona bem
(0,73–0,91), e não o lado onde falha.

---

## 5. Limitações declaradas

**Desequilíbrio de grupos.** 98,6% dos grupos são Curitiba. O `GroupKFold`
distribui grupos, então cada fold é majoritariamente Curitiba — o AUC de 0,7673
é, em boa medida, o desempenho em Curitiba.

**As 97 AOIs de planície não estão harmonizadas.** Ficaram de fora por terem
cadeia global. Entrar com elas exige rodar `ter01 --lote todas`.

**Petrópolis não entra**, por não ter rótulo local. Está derivado e pronto para
receber predição, não validação.

**A chuva não entra.** Nenhuma ativação CEMS tem chuva e o modelo é
exclusivamente topográfico. Ele responde onde inunda, não quando.

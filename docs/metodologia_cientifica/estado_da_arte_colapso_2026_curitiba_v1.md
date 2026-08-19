# Estado da arte — o que a literatura diz sobre o colapso prospectivo de Curitiba (2026-08-16)

**Pedido**: estudar estado da arte/literatura pra achar como resolver de vez o
colapso de 2026. **Resposta honesta antes dos achados**: a literatura não tem
um truque que transforma 0,52 em 0,68. O que ela tem são duas explicações
concorrentes, cada uma com um caminho de ação diferente e nenhuma delas é
"ajustar o modelo até dar certo" — o que violaria a regra fixa do projeto de
não misturar validação científica com otimização de performance. As duas
explicações continuam vivas depois desta busca; não escolhi uma por conta
própria.

## Explicação A — pouco dado de verdade pra Curitiba sozinha (tem remédio)

Restrito à evidência real (114 pontos em exclusão qualificada, ver relatório
de 16/08 anterior), o EPV das 5 variáveis reprova. Isso é literalmente o
cenário descrito por King & Zeng (2001, *Political Analysis*), o artigo que
formalizou o problema de amostra pequena em regressão logística — e que os
próprios autores resolvem com penalização, o mesmo mecanismo do Firth já
usado aqui. King e Zeng não é uma alternativa ao Firth: é a razão pela qual o
Firth já é a escolha certa. O piso de EPV (Peduzzi et al. 1996, já citado no
`main.tex`) é a régua que decide quando mesmo o Firth não segura.

**O que a literatura aponta como próximo passo real, e que este projeto ainda
não testou**: modelo hierárquico bayesiano com *partial pooling* (Gelman &
Hill; ver também aplicações recentes em `brms`/PyMC). A ideia: em vez de
ajustar Curitiba isolada (77 negativos reais) ou jogar tudo junto num pool só
(o que apaga a diferença entre regiões), o modelo estima um coeficiente por
região que é *puxado* em direção à média das outras regiões na proporção
inversa de quanto dado aquela região tem. Curitiba, com pouco dado real,
pesaria mais a informação das regiões com mais evidência (UK, CEMS);
Recife, que já tem 278 pontos maduros, quase não seria puxado. Isso é
diferente de tudo que já foi tentado aqui (PU bagging, GBM monotônico, ENSO)
porque nenhum desses "empresta força" de outra região — todos rodam
Curitiba sozinha ou o pool inteiro sem hierarquia.

**Ressalva que a própria literatura levanta**: modelos hierárquicos
"funcionam bem com 5+ grupos" — aqui são 6 fontes (UK, CEMS, Sen1Floods11,
UFO, Recife, Curitiba), no limite inferior recomendado, não confortavelmente
acima dele. E é ferramenta nova neste projeto (fora do `firthlogist`/
`interpret` já validados) — entra como tarefa própria, com ambiente e testes
antes de qualquer número, não decisão de hoje.

## Explicação B — a relação físico-chuva de Curitiba mudou de verdade em 2026 (não tem remédio estatístico)

Milly et al., "Stationarity Is Dead: Whither Water Management?" (*Science*,
2008) é o artigo que formalizou, pra hidrologia inteira, que assumir que a
relação passado→futuro se mantém estável deixou de ser seguro sob mudança
climática e de uso do solo. O achado do `SUSC-20Q` (coeficiente de chuva
cai a zero especificamente em 2026, depois de 3 anos consistentes) é
exatamente o tipo de sintoma que essa literatura descreve como
não-estacionariedade real — não artefato de amostra. Se for isso, nenhum
método de reamostragem, regularização ou escolha de ano resolve, porque não
é um problema de como o modelo foi ajustado — é o fenômeno físico mudando.
A única forma de testar essa hipótese é dado novo ao longo do tempo (mais
anos de 2026+ pra ver se o padrão se firma ou se foi um ano fora da curva),
que não existe ainda.

## O que a busca confirma que já estava certo, sem precisar mudar nada

- **Under-reporting em canal de queixa cidadã é achado geral da literatura,
  não peculiaridade do 156.** Kontokosta et al., "Equity in 311 Reporting"
  (arXiv:1710.02452) e trabalho de acompanhamento sobre viés de reportagem
  em dado 311 confirmam, com outra cidade e outro canal, o mesmo padrão que
  motivou excluir "ausência" pura do negativo aqui: quem não liga não é
  prova de que não aconteceu. Reforça — não substitui — `li2022pul` e
  `agostini2024`, já citados no `main.tex`.
- **PU-learning genérico não resolve deslocamento de distribuição.** A
  literatura recente (ex. Kumagai et al., *ICML*/*ICLR* 2025, PU learning
  sob *distribution shift* via *importance weighting*) confirma exatamente o
  que o `SUSC-20S` já tinha achado sozinho: PU bagging padrão assume
  treino e teste da mesma distribuição, e falha quando não é o caso. Existem
  variantes de PU mais novas desenhadas pra deslocamento de distribuição,
  mas todas exigem alguns pontos rotulados do período-alvo (2026) pra
  calibrar — o que abre risco real de vazamento entre treino e teste se não
  for desenhado com o mesmo cuidado que os testes anteriores já tiveram.
  Fica registrado como opção existente, não recomendada agora pelo risco
  metodológico.
- **Transferência entre regiões via aprendizado profundo** é a linha
  dominante na literatura de "flood transferability" hoje (redes neurais,
  operadores de transferência) — mas conflita direto com a prioridade do
  projeto por interpretabilidade (coeficiente com direção física
  verificável). Não é candidata aqui; hierárquico bayesiano entrega o mesmo
  princípio ("emprestar força de outra região") mantendo coeficiente
  interpretável.

## Recomendação concreta, sem prometer resultado

Testar o modelo hierárquico bayesiano (Explicação A) como próxima rodada
dedicada — é o único caminho novo, citável e consistente com as regras do
projeto (interpretável, sem otimizar performance escolhendo o que "dá
certo"). **Pré-registrar antes de rodar**: se o coeficiente de chuva de
Curitiba, mesmo puxado pela força das outras regiões, continuar
indistinguível de zero em 2026, isso é evidência a favor da Explicação B
(não-estacionariedade real) — não fica sem resposta, os dois caminhos se
testam com o mesmo experimento. Preciso da sua confirmação antes de instalar
ferramenta nova (`pymc` ou `brms`/R) e gastar uma rodada nisso.

## Fontes

- King, G. & Zeng, L. (2001). "Logistic Regression in Rare Events Data."
  *Political Analysis*, 9(2), 137-163.
- Milly, P. C. D. et al. (2008). "Stationarity Is Dead: Whither Water
  Management?" *Science*, 319(5863), 573-574.
- Kontokosta, C. E. & Hong, B. (2017/2021). "Equity in 311 Reporting:
  Understanding Socio-Spatial Differentials in the Propensity to Complain."
  arXiv:1710.02452.
- Gelman, A. & Hill, J. *Data Analysis Using Regression and
  Multilevel/Hierarchical Models* — capítulos de *partial pooling*.
- Kumagai, A. et al. (2025). "Importance-weighted Positive-unlabeled
  Learning for Distribution Shift Adaptation." *ICML/ICLR* (OpenReview
  CTYgrczjj2).

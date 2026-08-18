# Ontologia de estados de label e ground truth MV1

## Escopo
Este documento define estados futuros de label para o REV-P em modo fail-closed. Ele não cria labels reais, positivos formais reais, negativos formais reais ou ground truth operacional.

## Estados

### `positivo_ouro`
Estado futuro para patch com evidência independente forte de inundação, geometria compatível, janela temporal fechada e revisão humana completa.

- Evidência mínima exigida: Evento observado independente, fonte de label independente, geometria compatível entre evento e patch, janela temporal fechada e revisão humana/adjudicação.
- Exige evento observado: `true`
- Exige geometria: `true`
- Exige janela temporal fechada: `true`
- Exige revisão humana: `true`
- Pode treinar no futuro, se todos os gates estiverem satisfeitos: `true`
- Observação: Só pode entrar em treino futuro quando todos os gates estiverem satisfeitos; hoje não há amostras nesse estado.

### `positivo_prata`
Estado futuro para candidato positivo com evidência observacional parcial ou fonte independente incompleta.

- Evidência mínima exigida: Evento observado ou documental forte, mas com alguma lacuna em geometria, janela temporal ou adjudicação.
- Exige evento observado: `true`
- Exige geometria: `true`
- Exige janela temporal fechada: `true`
- Exige revisão humana: `true`
- Pode treinar no futuro, se todos os gates estiverem satisfeitos: `false`
- Observação: Não pode entrar no treino principal; só pode aparecer em ablação futura separada e explicitamente marcada.

### `negativo_pareado`
Estado futuro para patch pareado com um positivo e com evidência explícita de não inundação na janela controlada.

- Evidência mínima exigida: Evidência explícita de não inundação, fonte independente, mesma política temporal do positivo e revisão humana.
- Exige evento observado: `false`
- Exige geometria: `true`
- Exige janela temporal fechada: `true`
- Exige revisão humana: `true`
- Pode treinar no futuro, se todos os gates estiverem satisfeitos: `true`
- Observação: Só pode existir com evidência explícita de não inundação; ausência de evidência não satisfaz o estado.

### `negativo_dificil`
Estado futuro para patch visual ou territorialmente similar ao positivo, mas com evidência explícita de não inundação e controle espacial/temporal.

- Evidência mínima exigida: Evidência explícita de não inundação, pareamento espacial/temporal controlado, revisão humana e aprovação anti-leakage.
- Exige evento observado: `false`
- Exige geometria: `true`
- Exige janela temporal fechada: `true`
- Exige revisão humana: `true`
- Pode treinar no futuro, se todos os gates estiverem satisfeitos: `true`
- Observação: Curitiba nunca vira negativo formal por default; o estado exige evidência negativa explícita.

### `desconhecido`
Estado para patch sem evidência suficiente para positivo, negativo ou exclusão metodológica.

- Evidência mínima exigida: Registro da lacuna e preservação do estado sem inferência.
- Exige evento observado: `false`
- Exige geometria: `false`
- Exige janela temporal fechada: `false`
- Exige revisão humana: `false`
- Pode treinar no futuro, se todos os gates estiverem satisfeitos: `false`
- Observação: Desconhecido nunca pode treinar e nunca vira negativo.

### `excluido`
Estado para patch removido do conjunto por problema de qualidade, escopo, duplicidade, licença ou inconsistência metodológica.

- Evidência mínima exigida: Motivo de exclusão documentado e rastreável.
- Exige evento observado: `false`
- Exige geometria: `false`
- Exige janela temporal fechada: `false`
- Exige revisão humana: `false`
- Pode treinar no futuro, se todos os gates estiverem satisfeitos: `false`
- Observação: Excluído nunca pode treinar.

### `bloqueado`
Estado para patch com evidência insuficiente, conflito de proveniência, risco de circularidade ou gate obrigatório não satisfeito.

- Evidência mínima exigida: Bloqueador explícito e ação de desbloqueio documentada.
- Exige evento observado: `false`
- Exige geometria: `false`
- Exige janela temporal fechada: `false`
- Exige revisão humana: `false`
- Pode treinar no futuro, se todos os gates estiverem satisfeitos: `false`
- Observação: Bloqueado nunca pode treinar.

## Regra fail-closed
`desconhecido`, `bloqueado` e `excluido` nunca podem treinar. `positivo_prata` não entra no treino principal. `positivo_ouro`, `negativo_pareado` e `negativo_dificil` só são estados futuros treináveis quando todos os gates forem satisfeitos por amostra.

# Política de evidência negativa MV1

## Princípio
Negativo formal só pode existir com evidência explícita de não inundação. Ausência de registro, ausência de evento, cidade de contraste, unknown ou lacuna documental não são evidência negativa.

## Requisitos para `negativo_pareado`
- Fonte independente que registre não inundação na janela controlada.
- Patch e asset com identificadores rastreáveis.
- Relação temporal compatível com o positivo pareado.
- Revisão humana completa.
- Auditoria anti-leakage aprovada.

## Requisitos para `negativo_dificil`
- Evidência explícita de não inundação.
- Similaridade espacial, temporal ou visual controlada sem uso circular da feature como label.
- Geometria e janela temporal auditáveis.
- Revisão humana e adjudicação quando houver dúvida.

## Regras proibitivas
- `unknown` nunca vira negativo.
- Ausência de evento nunca vira negativo formal.
- Curitiba nunca vira negativo formal por default.
- Evidência contextual nunca vira label.
- O contraste entre cidades serve para análise label-free e priorização de revisão humana, não para criar negativo formal.

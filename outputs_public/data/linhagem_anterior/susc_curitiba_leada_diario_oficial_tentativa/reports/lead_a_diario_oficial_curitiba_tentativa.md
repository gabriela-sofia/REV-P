# Lead A (Curitiba) -- Diário Oficial do Município: busca real concluída, resultado negativo

**Status**: BUSCA_REAL_CONCLUIDA_RESULTADO_NEGATIVO -- portal oficial real localizado,
automatizado (via browser, capturando a URL de download real por trás do postback
ASP.NET) e as 3 edições completas do dia do evento foram baixadas e varridas por
palavra-chave. Nenhum decreto de enchente/alagamento/emergência encontrado nessa data
(diferente do Lead A de Recife, que fechou com 2 decretos confirmados).

## Atualização -- varredura completa das 3 edições de 17/01/2022

Usando o browser para simular o clique real no botão "Visualizar" (captura da chamada
`window.open` disparada pelo postback ASP.NET, sem precisar reimplementar
`__VIEWSTATE`), foram baixadas as 3 edições reais do Diário Oficial do dia
17/01/2022 (o dia central da janela do evento já registrada,
S17C_REF_0060 = 2022-01-15/16):

- Edição Nº 11 (76 páginas, `DiarioConsultaExterna_Download.aspx?Id=4984`)
- Suplemento 1 (68 páginas, `DiarioSuplementoConsultaExterna_Download.aspx?Id=3035`)
- Suplemento 2 (249 páginas, `DiarioSuplementoConsultaExterna_Download.aspx?Id=3036`)

Total: **393 páginas**, texto extraído (pypdf) e varrido por
`enchente|alagament|inunda|chuva|emerg[eê]ncia|defesa civil|temporal|desastre|anormal`.

**Resultado real (não fabricado): zero decretos de enchente/emergência climática.** As
únicas 2 páginas com match (no Suplemento 1) são um serviço socioassistencial
padrão do SUAS ("Serviço de Proteção em Situações de Calamidades Públicas e de
Emergências") -- um item de cadastro de programa social, não um decreto de evento real.
O conteúdo das 393 páginas é 100% administrativo de rotina: licitações, portarias de
pessoal, decretos orçamentários, extratos de contrato.

**Implicação honesta**: o decreto/registro administrativo que sustenta
`S17C_REF_0060` (2022-01-15/16) não está no Diário Oficial do dia 17/01/2022 --
precisa ter vindo de outra fonte (talvez edição de dias/semanas depois, já que
decretos de emergência costumam ser publicados após avaliação de danos, não no dia
do pico da chuva) ou de nível estadual, não municipal. Isso não invalida o evento
já registrado no projeto (que tem outra fonte documentada), só significa que **esta
via específica (Diário Oficial municipal do dia exato) não o corrobora** -- um
resultado negativo real, não um "não tentei".

## O que foi encontrado de real

- Portal oficial real do Diário Oficial do Município de Curitiba (DOM-CTBA):
  `https://legisladocexterno.curitiba.pr.gov.br/DiarioConsultaExterna_Pesquisa.aspx`
  -- confirmado como o sistema real de consulta (Legisladoc), mas é uma página
  ASP.NET WebForms baseada em postback (`__VIEWSTATE`), não uma API REST simples --
  não é acessível por GET direto como o CKAN usado para Recife.
- Padrão real de PDF direto de decretos: `https://mid.curitiba.pr.gov.br/{ano}/{id}.pdf`
  (confirmado com `Decreto Nº 940` e `Decreto - Aviso de Publicação Nº 41`, ambos
  recuperáveis), mas o `{id}` é um identificador interno sequencial de documento, não o
  número do decreto -- não dá para "adivinhar" o PDF certo sem passar pela busca.
- Confirmação jornalística real (não oficial) de que Curitiba teve chuva acima da
  média em janeiro de 2022 (194,20mm vs média histórica 184,97mm) e que uma tempestade
  em **17/01/2022** causou queda de árvores e alagamentos -- uma data próxima, mas não
  idêntica, à janela já registrada no projeto (2022-01-15/16, `S17C_REF_0060`). Isso é
  consistente com um evento de chuva de vários dias (15-17/01/2022), não uma
  contradição -- mas não deve ser tratado como confirmação da data exata sem o decreto.
- `Decreto Nº 30 DE 13/01/2022` (encontrado via LegisWeb) foi verificado e é sobre
  COVID-19 (medidas restritivas, Alerta Amarelo), **não** sobre chuva/enchente --
  coincidência de datas, registrado aqui para não ser reutilizado por engano em
  sessão futura.

## Pendência concreta (atualizada após a varredura completa)

O decreto oficial específico de situação de emergência/anormalidade por chuva em
Curitiba não foi recuperado nesta sessão -- e agora sabemos que **não está na edição do
dia 17/01/2022** (393 páginas checadas, zero match real). Os próximos passos reais:

1. Repetir a mesma busca (já viável: busca por data no Legisladoc + captura do
   `window.open` real, sem precisar reimplementar `__VIEWSTATE` manualmente) para as
   edições dos dias seguintes (18-31/01/2022 e possivelmente fevereiro/2022) --
   decretos de situação de emergência costumam vir depois da avaliação de danos, não no
   dia do evento.
2. Checar nível estadual (Defesa Civil do Paraná / decreto estadual), não só municipal
   -- o padrão observado em outros estados (ex. a notícia real já encontrada sobre
   "Governo do Estado decreta situação de emergência na região de Curitiba e Litoral")
   mostra que decretos de chuva no Paraná às vezes são estaduais, não municipais.
3. Alternativa: contatar a Defesa Civil de Curitiba diretamente (mesmo canal já mapeado
   em `SUSC_18C_AQUISICAO_GEOMETRIA_OFICIAL_CURITIBA.md` para a geometria de ocorrência
   -- os dois pedidos podem ser feitos juntos).

## Diferença honesta em relação a Recife

Recife tinha um portal de diário oficial mais simples de varrer por texto (PDFs
enumerados e buscáveis em lote, 52 candidatos lidos integralmente). Curitiba usa um
sistema de busca interativo (Legisladoc) que não expõe uma rota de consulta simples --
essa é uma diferença estrutural real entre as duas fontes, não uma falha de execução.

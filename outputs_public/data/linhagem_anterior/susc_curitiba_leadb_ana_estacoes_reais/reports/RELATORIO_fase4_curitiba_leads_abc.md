# Fase 4 (Curitiba) -- Leads A/B/C reais, mesmo método do v12/SUSC-20A para Recife

**Status**: FASE4_PARCIALMENTE_CONCLUIDA_COM_ACHADOS_REAIS -- 2 de 3 leads fechados com
prova real; 1 lead com tentativa real registrada e pendência concreta explícita (mesmo
padrão de honestidade já usado nos relatórios de lead do Recife).

## Lead A -- Diário Oficial do Município

**Busca real concluída, resultado negativo.** Portal oficial (`legisladocexterno.curitiba.pr.gov.br`)
localizado; postback ASP.NET contornado via browser (captura da URL real de download por
trás do clique em "Visualizar", sem precisar reimplementar `__VIEWSTATE`). As 3 edições
completas do dia 17/01/2022 (393 páginas: edição principal + 2 suplementos) foram
baixadas e varridas por palavra-chave -- **zero decretos de enchente/emergência
climática encontrados**. O evento já documentado no projeto (`S17C_REF_0060`) não é
corroborado por esta via específica (Diário Oficial municipal do dia exato); próximo
passo real é checar edições dos dias seguintes ou nível estadual. Ver
`susc_curitiba_leada_diario_oficial_tentativa/reports/lead_a_diario_oficial_curitiba_tentativa.md`.

## Lead B -- ANA (estações fluviométricas/cotagrama reais)

**Concluído com achado real.** 104 estações na bbox metropolitana, 63 com dado real
baixado (vazão + cota, ambos os tipos testados). **9 estações cobrem o evento de
2022-01-15/16**; 2 delas (Rio Pequeno e Rio Miringuava, São José dos Pinhais) mostraram
nível acima do próprio percentil-95 histórico nessas datas -- corroboração hidrológica
real, mesmo padrão metodológico do Lead B de Recife. A única estação real dentro do
município de Curitiba que cobre a data (Rio Barigui, 65019675) não mostrou anomalia --
resultado real e reportado sem omissão, não um resultado "conveniente". Ver
`susc_curitiba_leadb_ana_estacoes_reais/reports/lead_b_ana_curitiba_estacoes_reais.md`.

## Lead C -- Global Flood Database (MODIS-validado)

**Concluído com achado real.** Do catálogo DFO (4.825 registros globais), 28 candidatos
intersectam o Paraná; **DFO_4276** (2015-07-10 a 2015-07-21, evento regional de chuva)
existe no bucket validado `gfd_v1_4`. Recortado na bbox de Curitiba: **54 pixels de
inundação nova genuína** (0 sobre água permanente), centróide geocodificado no **bairro
São Miguel, Curitiba**. Novo candidato: `LEADC_CTBA_2015_0001`. Ver
`susc_curitiba_leadc_global_flood_database/reports/lead_c_global_flood_database_curitiba.md`.

## O que isso muda para o status de Curitiba no produto

Ainda **não** promove Curitiba a "análise disponível" (status honesto continua sendo
"evidência em processamento", como o `PLANO_ACAO_produto_v1.md` já previa) -- porque:

1. A geometria oficial de ocorrência do evento de jan/2022 continua ausente (bloqueio
   já documentado em `SUSC_18C_AQUISICAO_GEOMETRIA_OFICIAL_CURITIBA.md`, não resolvido
   nesta sessão).
2. Só 2 pontos reais novos foram estabelecidos nesta rodada (corroboração ANA +
   candidato GFD) -- muito aquém dos 278 pontos que sustentam o modelo de Recife.
3. Nenhum modelo Firth próprio foi treinado para Curitiba -- isso exigiria repetir todo
   o SUSC-20B (features físico-hidrológicas por ponto) e SUSC-20C (modelagem), que não
   fazem sentido com apenas 2 pontos novos.

**O que isso muda de fato**: a lacuna deixa de ser "nada foi tentado" (situação anterior,
registrada em 18C como offline-first sem rede) e passa a ser "dados reais adquiridos,
gap remanescente é geometria de ocorrência + volume de eventos", com dois novos
candidatos concretos de evidência (ANA corroboração + GFD MODIS) prontos para entrar no
mesmo pipeline de adjudicação já usado para os pontos de Recife.

## Próximos passos reais (não prometem prazo)

1. Levar `LEADC_CTBA_2015_0001` e a corroboração ANA de 2022-01-15/16 ao mesmo processo
   de adjudicação review-only já usado para os pontos SEDEC de Recife (`SUSC-DEVRO02G`
   etc.) antes de qualquer uso como feature/label.
2. Resolver o Lead A automatizando o postback do Legisladoc ou contatando a Defesa Civil
   de Curitiba diretamente (mesmo canal já mapeado para a geometria de ocorrência em 18C
   -- os dois pedidos podem ser feitos juntos).
3. Repetir a varredura do catálogo DFO com uma bbox mais ampla (Paraná inteiro, já
   feito) e checar os candidatos restantes (ex.: 4323/2016, 4212/2014, 3790/2011) contra
   o bucket -- só 1 de 5 testados existia; os outros podem estar sob nomes/anos
   ligeiramente diferentes no bucket.

# Solicitacao tecnica de geometria de ocorrencia - eventos de alagamento em Curitiba

## Finalidade

Solicitacao de dados espaciais de ocorrencia de alagamento/inundacao/enxurrada em Curitiba, para uso
academico em analise observacional de suscetibilidade, em carater somente revisao. Os dados nao serao
usados como verdade de referencia operacional, nem para treino de modelo, e nenhum score oficial sera
alterado.

## Destinatarios sugeridos

- Defesa Civil Municipal de Curitiba
- Instituto de Pesquisa e Planejamento Urbano de Curitiba (IPPUC) / GeoCuritiba
- Portal de Dados Abertos de Curitiba

## Eventos de interesse

- Evento principal: temporal e alagamentos de 15 e 16 de janeiro de 2022 (referencia interna CUR_2022_01_15).
- Eventos adicionais: outubro de 2023 e fevereiro de 2024.

## Dados solicitados por ocorrencia

1. Data da ocorrencia (dia exato ou intervalo curto).
2. Tipo de fenomeno (alagamento, inundacao ou enxurrada), confirmado explicitamente.
3. Geometria da ocorrencia: ponto, poligono ou retangulo envolvente (bbox) da area efetivamente atingida.
4. Endereco textual de referencia, se houver (apenas complementar; nao substitui a geometria).
5. Fonte e responsavel tecnico pelo registro.
6. Precisao espacial estimada (em metros).
7. Observacao sobre o metodo de coleta (vistoria em campo, registro de chamado, sensoriamento etc.).
8. Autorizacao de uso academico dos dados.

## Requisitos tecnicos

- Sistema de referencia espacial informado (preferencialmente EPSG:4326; qualquer CRS oficial e aceito e sera convertido).
- Formatos leves: GeoJSON, shapefile, CSV com colunas de coordenada, JSON ou XLSX.
- Nao e necessario enviar imagens de satelite nem arquivos pesados.
- A geometria deve representar a ocorrencia observada, e nao area de risco generica, setor administrativo ou limite de bairro.

## Modelo de resposta

Segue planilha modelo (`modelo_planilha_resposta_curitiba.csv`) e o schema de ingestao
(`schema_resposta_esperada_curitiba.json`) para padronizar o retorno.

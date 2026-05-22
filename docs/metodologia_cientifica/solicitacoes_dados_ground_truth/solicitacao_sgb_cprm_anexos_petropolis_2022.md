# Solicitação — SGB/CPRM: Dados Georreferenciados de Campo, Petrópolis 2022

**Destinatário:** Serviço Geológico do Brasil (SGB/CPRM)  
**Referência:** Avaliação técnica pós-desastre: Petrópolis, RJ (RIGeo: https://rigeo.sgb.gov.br/handle/doc/22668)  
**Autores do relatório:** Filipe Modesto, Leandro Kuhlmann, Patrícia Jacques, Rafael Ribeiro, Thiago Santos  
**Justificativa:** Pesquisa acadêmica — construção de corpus de referência observacional para aprendizado de máquina aplicado a detecção de áreas afetadas por desastres hidrológicos/geológicos  
**Data de elaboração:** 2026-05-22  

---

## Contexto

O SGB/CPRM publicou o relatório *Avaliação técnica pós-desastre: Petrópolis, RJ* (2022) com anexos ZIP contendo 11 PDFs de avaliação de campo por bairro, produzidos entre 19/02/2022 e 02/03/2022.

Esses PDFs confirmam que a equipe técnica realizou vistorias de campo em múltiplos bairros afetados pelo evento de 15/02/2022. Para a construção de corpus de referência terrestre com uso em pesquisa, os dados digitais que embasaram esses relatórios de campo são essenciais.

---

## Dados solicitados

Solicito acesso, para fins de pesquisa acadêmica (uso não-comercial, com citação completa), aos seguintes dados digitais produzidos durante as vistorias de campo de Petrópolis 2022:

### 1. Dados vetoriais de localidades inspecionadas

- **Formato preferencial:** shapefile (.shp), geopackage (.gpkg), KMZ, KML ou GeoJSON
- **Conteúdo:** polígonos e/ou pontos de áreas/locais vistoriados, com atributos de:
  - Data da vistoria
  - Tipo de fenômeno (inundação/alagamento/enxurrada vs. deslizamento/escorregamento/corrida de massa)
  - Bairro/localidade
  - Código ou referência ao anexo PDF correspondente

### 2. Metadados obrigatórios

- Sistema de referência de coordenadas (CRS/datum)
- Escala ou precisão posicional estimada
- Método de coleta (GPS de campo, croqui georref, fotointerpretação)
- Se o dado representa ocorrência observada ou estimativa/risco

### 3. Classificação do fenômeno

- Separação explícita entre:
  - Inundação / alagamento / enxurrada / transbordamento
  - Deslizamento / escorregamento / corrida de massa
  - Ocorrência mista (ambos na mesma localidade)

### 4. Licença/termo de uso

- Confirmação de que o dado pode ser usado em pesquisa acadêmica com citação
- Eventual restrição de redistribuição

---

## Especificação técnica dos anexos cobertos

Os PDFs disponíveis no ZIP cobrem:
- Bairros Mosella e Moinho Preto (ANEXO-I e II, 19/02/2022)
- Bairro Serra Velha (ANEXO-III, 20/02/2022)
- Bairro Valparaíso (ANEXO-IV, 22/02/2022)
- Rua Teresa e imediações (ANEXO-V, 23/02/2022)
- Revisitas a Mosella e Moinho Preto (ANEXO-VI e VII, 24/02/2022)
- Estrada Velha e Vila Felipe (ANEXO-VIII, 25–26/02/2022)
- Bairro Sargento Boening (ANEXO-IX, 28/02/2022)
- Servidão Alépio Gomes da Costa (ANEXO-X, 01/03/2022)
- Bairro Quitandinha (ANEXO-XI, 02/03/2022)

Solicito especificamente dados georref que permitam distinguir, para cada bairro, se o fenômeno predominante foi inundação/alagamento ou deslizamento/escorregamento, com localização espacial explícita.

---

## Resultado esperado

Dados que permitam construir registros de ground reference auditáveis — não para treinamento direto de modelo, mas para validação de uma pipeline de análise por sensoriamento remoto de eventos de desastre urbano.

**Não é esperado:** forecast, predição de risco ou suscetibilidade. Apenas registros de ocorrência observada datada e georreferenciada.

---

*Elaborado no contexto do Protocolo C do projeto REV-P — análise de mudanças em séries temporais Sentinel por aprendizado de máquina.*

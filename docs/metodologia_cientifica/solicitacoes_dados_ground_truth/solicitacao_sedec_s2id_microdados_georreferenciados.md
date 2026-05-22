# Solicitação — Sedec/MIDR: Microdados Georreferenciados de Desastres (S2ID/Atlas)

**Destinatário:** Secretaria Nacional de Proteção e Defesa Civil (Sedec) / Ministério da Integração e do Desenvolvimento Regional (MIDR)  
**Portal S2ID:** https://s2id.mi.gov.br  
**Portal Atlas:** https://atlasdigital.mdr.gov.br  
**Justificativa:** Pesquisa acadêmica — validação de corpus de referência para eventos de inundação urbana  
**Data de elaboração:** 2026-05-22  

---

## Contexto

O S2ID (Sistema Integrado de Informações sobre Desastres) é a base oficial de registros de desastres no Brasil. O Atlas Digital de Desastres agrega dados históricos por município, tipo e impacto. Esses dados confirmam a ocorrência de eventos e permitem verificar datas, decretos e municípios afetados.

**Limitação conhecida:** Os registros do S2ID e do Atlas são em nível municipal — não fornecem geometria intraurbana (bairro, lote, patch). Para fins de pesquisa que requerem localização intraurbana, são necessários microdados georref.

---

## Dados solicitados

### 1. Registros S2ID para eventos de Petrópolis e Recife (2022)

- COBRADE do evento (tipo de desastre)
- Data de ocorrência
- Data do decreto municipal/estadual
- Número do decreto de emergência/calamidade
- Afetados, mortos, desalojados, desabrigados

### 2. Se disponíveis: dados georref de ocorrências

- Planilha ou shapefile com coordenadas de ocorrências registradas na fase de resposta
- Nível de georref disponível: município, bairro, logradouro, ponto GPS
- Se há camada espacial agregada além do limite municipal

### 3. Atlas Digital: exportação para município de Petrópolis / Recife

- Eventos COBRADE 12200 (inundação), 12300 (enxurrada), 11100 (deslizamento) entre 2015 e 2023
- Com data, afetados e qualquer geometria disponível

### 4. Metadados

- Fonte de coleta dos dados de ocorrência (Defesa Civil municipal, relatório técnico, SIGDC)
- CRS, se dado espacial disponível
- Método de localização (endereço, coordenada GPS, polígono)

### 5. Licença/termo de uso para pesquisa acadêmica

---

## Canais de acesso

- Portal S2ID: https://s2id.mi.gov.br (dados abertos parcialmente disponíveis)
- Atlas Digital: https://atlasdigital.mdr.gov.br (exportação por município)
- e-SIC / LAI: para dados não publicados diretamente nos portais

---

*Elaborado no contexto do Protocolo C do projeto REV-P.*

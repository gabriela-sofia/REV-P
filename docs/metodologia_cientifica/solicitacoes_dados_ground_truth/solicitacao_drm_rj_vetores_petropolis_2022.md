# Solicitação — DRM-RJ/NADE: Vetores Cartográficos Pós-Desastre, Petrópolis 2022

**Destinatário:** Departamento de Recursos Minerais do Rio de Janeiro (DRM-RJ) / Núcleo de Avaliação de Desastres e Emergências (NADE)  
**Portal de referência:** https://www.drm.rj.gov.br  
**Justificativa:** Pesquisa acadêmica — corpus de referência observacional para aprendizado de máquina em eventos hidrológicos/geológicos urbanos  
**Código interno REV-P:** PKG_FR_PET_001  
**Data de elaboração:** 2026-05-22  

---

## Contexto

O DRM-RJ (por meio do NADE) conduziu avaliações técnicas de campo após o evento de 15/02/2022 em Petrópolis, RJ. O relatório DRM-RJ adquirido previamente (57 páginas) contém descrição textual por bairro e localidade, confirmando avaliações de inundação, deslizamento e ocorrências mistas.

Para fins de pesquisa, a limitação atual é a ausência de geometria vetorial georreferenciada explícita — o relatório narrativo não é suficiente para construção de ground reference verificável.

---

## Dados solicitados

Solicito acesso, para uso exclusivo em pesquisa acadêmica com citação, aos seguintes dados:

### 1. Geometria vetorial por tipo de fenômeno e bairro

- **Formato:** shapefile (.shp), geopackage (.gpkg), KMZ, KML ou GeoJSON
- **Conteúdo mínimo:**
  - Polígonos ou pontos de ocorrência por localidade
  - Tipo de fenômeno por feição: inundação, alagamento, enxurrada, transbordamento, deslizamento, escorregamento, corrida de massa, misto
  - Bairro e/ou logradouro
  - Data da ocorrência ou da vistoria

### 2. Dicionário de campos

- Descrição de cada campo no shapefile/tabela de atributos
- Unidade de cada variável numérica
- Codificação de variáveis categóricas (tipo de fenômeno, grau de dano, etc.)

### 3. Metadados obrigatórios

- CRS (datum, projeção)
- Escala de levantamento
- Método de coleta (GPS, croqui, fotointerpretação, topografia)
- Se representa ocorrência observada ou área de risco/suscetibilidade

### 4. Separação de fenômeno

Solicito **explicitamente** que os dados distinguam:
- Áreas de **inundação / alagamento / enxurrada / transbordamento** (fenômeno hidrológico)
- Áreas de **deslizamento / escorregamento / corrida de massa** (fenômeno geológico/geomorfológico)
- Áreas com ocorrência simultânea de ambos (misto)

Essa separação é metodologicamente obrigatória para a pesquisa e não pode ser substituída por shapefile único sem campo discriminador.

### 5. Licença/termo de uso

- Confirmação de uso em pesquisa acadêmica com citação
- Restrição de redistribuição pública, se houver

---

## Bairros de interesse prioritário

Com base no relatório narrativo DRM-RJ já adquirido, os bairros com ocorrências mais críticas incluem:
- Alto da Serra / Chácara Flora (deslizamentos HIGH confidence)
- Caxambu (deslizamentos HIGH confidence)
- Quitandinha (transbordamento do Rio Quitandinha)
- Morin (ocorrência mista)
- Mosella, Moinho Preto, Serra Velha (ocorrências documentadas)

---

## Canal sugerido de solicitação

- Contato institucional via portal DRM-RJ
- Se necessário, via Lei de Acesso à Informação (LAI): Petrópolis/RJ ou Estado do RJ

---

*Elaborado no contexto do Protocolo C do projeto REV-P.*

# Template de Intake Manual — Fonte Observacional

## Instruções de preenchimento

Preencher manualmente após revisão direta da fonte.
- Não inserir dado pessoal (nome, CPF, endereço de pessoa física)
- Não inserir path privado do workspace local
- Não publicar dado bruto se a licença não permitir redistribuição
- Não transformar a fonte em label de treino

---

## Identificação

- **SOURCE_ID**: [SOURCE_ID]
- **OBSERVED_EVENT_ID**: [OBSERVED_EVENT_ID]
- **REGIAO**: [REGIAO]

---

## Fonte

- **FONTE**: [FONTE]
- **INSTITUICAO**: [INSTITUICAO]
- **URL**: [URL]
- **DATA_DE_ACESSO**: [DATA_DE_ACESSO]
- **TIPO_DE_DOCUMENTO**: [TIPO_DE_DOCUMENTO]

---

## Conteúdo documental

- **DATA_DO_EVENTO**: [DATA_DO_EVENTO]
- **LOCALIDADE_DESCRITA**: [LOCALIDADE_DESCRITA]
- **EVIDENCIA_ESPACIAL**: [EVIDENCIA_ESPACIAL]
  - Descrever o que a fonte apresenta: bairro, logradouro, corredor, mapa, geometria, ponto, área descritiva
  - Não inferir além do que está na fonte
- **EVIDENCIA_TEMPORAL**: [EVIDENCIA_TEMPORAL]
  - Data exata, janela de dias, série temporal — conforme documentado na fonte

---

## Licença e proveniência

- **LICENCA**: [LICENCA]
  - Preencher: público domínio, Creative Commons (especificar tipo), governo aberto, redistribuição restrita, desconhecido
- **PODE_REPUBLICAR**: [PODE_REPUBLICAR]
  - true / false / desconhecido
- **DEVE_FICAR_LOCAL_ONLY**: [DEVE_FICAR_LOCAL_ONLY]
  - true / false — dados brutos com redistribuição restrita devem permanecer em local_only/
- **ARQUIVO_LOCAL_SEGURO**: [ARQUIVO_LOCAL_SEGURO]
  - Nome do arquivo local sem path privado; apenas o nome do arquivo

---

## Relação com os gates do Protocolo C

- **GATES_QUE_PODE_FECHAR**: [GATES_QUE_PODE_FECHAR]
  - Ex: G1_EVENT_CONFIRMATION; G2_SOURCE_AVAILABILITY; G3_TEMPORAL_ALIGNMENT
- **GATES_QUE_NAO_FECHA**: [GATES_QUE_NAO_FECHA]
  - Listar explicitamente os gates que esta fonte não fecha
  - Atenção: fontes documentais não fecham G4 em nível patch-level, G7, G8 ou G9 automaticamente

---

## Lacunas identificadas

- **LACUNAS**: [LACUNAS]
  - Listar o que a fonte não fornece e que ainda será necessário para avançar

---

## Decisão metodológica

- **DECISAO_METODOLOGICA**: [DECISAO_METODOLOGICA]
  - Opções: ACCEPT_METADATA_ONLY / ACCEPT_FOR_GATE_SUPPORT / BLOCK_USE / REQUEST_MORE_INFORMATION / LICENSE_REVIEW_REQUIRED

---

## Claims

- **CLAIM_PERMITIDO**: [CLAIM_PERMITIDO]
  - Descrever apenas o que pode ser afirmado com base nesta fonte isolada
  - Ex: "há fonte rastreável que documenta o evento em tal data e localidade"
- **CLAIM_PROIBIDO**: [CLAIM_PROIBIDO]
  - Sempre incluir: ground truth operacional; flood label; training label; flood detection operacional; flood prediction; supervised training
  - Adicionar outros proibidos específicos desta fonte

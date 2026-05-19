# Log de Busca de Evento — Protocolo C

> Template para registro de busca manual futura.
> Não preencher com busca real nesta etapa (v1ho).
> Preencher quando uma busca de fonte for efetivamente executada.

---

## Identificação da busca

| Campo | Valor |
|-------|-------|
| **Log ID** | [LOG_ID] |
| **Dossier ID** | [DOSSIER_ID] |
| **Evento candidato** | [EVENTO_CANDIDATO] |
| **Região** | [REGIAO] |

---

## Execução da busca

| Campo | Valor |
|-------|-------|
| **Data da busca** | [DATA_DA_BUSCA] |
| **Revisor metodológico** | [REVISOR_METODOLOGICO] |

*O campo revisor é opcional. Não preencher com dados pessoais se não autorizado. Pode ser preenchido com papel (ex: "revisor interno", "pesquisador responsável") em vez de nome individual.*

---

## Parâmetros da busca

| Campo | Valor |
|-------|-------|
| **Fonte consultada** | [FONTE_CONSULTADA] |
| **Termo de busca** | [TERMO_DE_BUSCA] |

*Liste os termos, filtros ou consultas usados na busca.*

---

## Resultado

| Campo | Valor |
|-------|-------|
| **Resultado encontrado** | [RESULTADO_ENCONTRADO] |

*Sim / Não / Parcial. Se parcial, descreva o que foi encontrado.*

---

## Evidências encontradas

| Tipo | Descrição |
|------|-----------|
| **Evidência temporal** | [EVIDENCIA_TEMPORAL] |
| **Evidência espacial** | [EVIDENCIA_ESPACIAL] |

*Se não encontrada, preencher com "Não encontrada".*

---

## Licença e proveniência

| Campo | Valor |
|-------|-------|
| **Licença/proveniência** | [LICENCA_PROVENIENCIA] |

*Ex: CC BY 4.0, uso acadêmico permitido, termos não disponíveis, UNKNOWN.*

---

## Decisão

| Campo | Valor |
|-------|-------|
| **Decisão** | [DECISAO] |
| **Bloqueador** | [BLOQUEIO] |
| **Próxima ação** | [PROXIMA_ACAO] |

*Decisão possível: ACCEPT_METADATA_ONLY, REQUEST_MORE_INFORMATION, BLOCK_USE, SOURCE_NOT_FOUND, CONTINUE_SEARCH.*
*Bloqueador: descrever se houver (ex: licença UNKNOWN, acesso restrito). "Nenhum" se não houver.*

---

*Log de busca gerado pelo Protocolo C. Não constitui declaração de ground truth. Preencher apenas quando busca real for executada.*

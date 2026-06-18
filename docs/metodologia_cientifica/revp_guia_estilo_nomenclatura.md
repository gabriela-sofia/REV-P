# REV-P — Guia editorial e de nomenclatura

Versão: 1.0 — 2026-06-18

Este guia define o tom, a língua e a nomenclatura que o repositório REV-P deve adotar daqui em diante. O objetivo é que qualquer documento público — README, relatório de execução, doc metodológico, tabela de estágio — pareça escrito por uma equipe de pesquisa cuidadosa, não gerado automaticamente por um pipeline.

---

## 1. Tom

O repositório fala de si mesmo de forma:

- **técnica**: usa terminologia precisa da área (sensoriamento remoto, SIG, aprendizado representacional, evidência territorial);
- **direta**: diz o que fez, o que não fez e por quê — sem rodeios nem floreios;
- **humana**: prefere parágrafos curtos a listas de 15 itens; prefere frase afirmativa a disclaimer defensivo;
- **acadêmica**: cita restrições como escolhas metodológicas, não como falhas ou limitações a se desculpar;
- **específica**: menciona nomes reais (REC_00205, INMET A807, DRM-RJ, Charter 758), números reais (0.76, 768D, 59 patches), decisões reais.

O repositório **não** fala de si mesmo de forma:

- promocional ("framework robusto e abrangente");
- defensiva em excesso (não empilhar 10 frases dizendo o que não faz);
- genérica (não descrever o pipeline como "abordagem ponta-a-ponta baseada em IA");
- vaga ("metodologia avançada", "análise aprofundada").

---

## 2. Política de língua

### 2.1 Regra geral

| Contexto | Língua | Justificativa |
|---|---|---|
| Código Python (funções, classes, variáveis) | Inglês | Compatibilidade, convenção da área |
| Nomes de arquivos internos (scripts, testes, datasets) | Inglês ou misto | Legado; renomear quebraria imports |
| Documentação metodológica (`docs/`) | Português técnico | Projeto brasileiro, audiência acadêmica nacional |
| Relatórios públicos (`outputs_public/`) | Português técnico | Idem |
| README principal | Português | Entrada pública principal |
| Valores de status em campos CSV/código | Inglês aceito | São contratos técnicos, não texto narrativo |
| Títulos de seções em documentos públicos | Português | Consistência editorial |

### 2.2 Mistura permitida

Termos técnicos que não têm tradução natural consagrada podem permanecer em inglês dentro do texto em português, sem itálico forçado:

- `DINOv2` (nome do modelo)
- `Sentinel-2` (nome do satélite)
- `k-NN`, `PCA`, `UMAP` (siglas técnicas estabelecidas)
- `ground truth` (quando referido como conceito técnico)
- `patches` (quando o contexto técnico já está estabelecido)

### 2.3 Mistura proibida

Não misturar português e inglês na mesma frase descritiva sem necessidade:

❌ "O pipeline está `BLOCKED` porque não há `formal_negatives`."
✅ "O pipeline está bloqueado porque não há negativos formais (campo `formal_negatives` ausente)."

Não abrir seção de relatório com código de status em inglês sem explicação:

❌ "Status: `PROTOCOL_VALIDATED_CANDIDATE_REFERENCE` (score 0.76)"
✅ "Recife tem referência candidata validada pelo Protocolo C (pontuação: 0.76). O código interno é `PROTOCOL_VALIDATED_CANDIDATE_REFERENCE`."

---

## 3. Nomenclatura — tabela de substituições

As substituições abaixo se aplicam a texto narrativo em documentos públicos. Não se aplicam a nomes de variáveis, campos de CSV, enums ou chaves de dicionário Python.

| Termo original (inglês/jargão) | Substituição recomendada em português | Exceção |
|---|---|---|
| `scaffold` | estrutura inicial, esqueleto metodológico, contrato de código | manter como nome de arquivo se houver testes dependentes |
| `guardrail` | restrição metodológica, controle, trava | manter como enum/status em código |
| `readiness` | prontidão, estado de preparo | manter em nome de arquivo e campo CSV |
| `blocked` (status) | bloqueado por evidência insuficiente (em texto); `BLOCKED` (em tabela) | sempre explicar o motivo do bloqueio |
| `candidate` | candidato, amostra candidata, fonte candidata | usar em português quando narrativo |
| `candidate-only` | somente candidato, status candidato | manter como enum |
| `dry-run` | execução simulada, ensaio local, validação sem efeito | manter em comentário de código |
| `harness` | ambiente de ensaio, pré-execução, contrato experimental | manter como nome técnico quando estabelecido |
| `sidecar` | arquivo auxiliar, artefato auxiliar | manter quando é nome técnico de formato |
| `manifest` | manifesto | aplicar no texto; manter inglês em nome de arquivo |
| `registry` | registro | aplicar no texto; manter inglês em nome de arquivo |
| `rollup` | resumo consolidado | |
| `pipeline` (repetido) | cadeia, fluxo, etapa, processo | usar `pipeline` com parcimônia; não repetir no mesmo parágrafo |
| `end-to-end` | completo, de ponta a ponta | usar apenas quando literalmente verdadeiro |
| `comprehensive` | completo, abrangente | evitar como adjetivo vazio |
| `robust` | sólido, consistente, verificado | evitar sem evidência; nunca como marketing |
| `state-of-the-art` | referência atual na área, melhor disponível | evitar sem citação |
| `AI-powered` | baseado em representação auto-supervisionada | descrever o que realmente foi feito |
| `automated detection` | identificação, análise estrutural | o projeto não detecta |
| `prediction` | projeção, estimativa, análise (conforme contexto) | o projeto não prediz |
| `training-ready` | pronto para treinamento | manter como status técnico; nunca afirmar para este projeto |
| `ground-truth-ready` | pronto para rótulo operacional | manter como status; nunca afirmar para este projeto |

---

## 4. Frases preferidas

### Em vez de frases robóticas, usar:

| Frase robótica | Alternativa humana |
|---|---|
| "This artifact implements a conservative guardrail-based readiness scaffold." | "Este documento organiza as condições necessárias antes de avançar para a próxima etapa." |
| "The pipeline preserves blocked state across all stages." | "Os estágios seguintes estão bloqueados até que a evidência requerida esteja disponível." |
| "No operational ground truth is promoted." | "Nenhum patch é tratado como rótulo operacional nesta versão." |
| "Candidate-only status is enforced by design." | "Os itens em status candidato podem ser analisados, mas não usados como rótulo de treinamento." |
| "Operational label = 0 \| negative = 0 \| training = 0" | "Nenhum rótulo operacional, negativo formal ou amostra de treinamento foi criado." |
| "Status: BLOCKED. Reason: no formal negatives." | "Bloqueado: não há negativos formais disponíveis para esta etapa." |
| "PRESENT_2 / ABSENT / AVAILABLE_BUT_BLOCKED" | Explicar em prosa: "A cadeia tem 2 eventos com geometria presente; a sobreposição espacial está disponível tecnicamente, mas bloqueada por ausência de geometria oficial do evento." |

---

## 5. Estrutura de relatório público

Todo relatório público em `outputs_public/execution_reports/` deve ter:

1. **Título claro** em português (não apenas o código de estágio)
2. **Parágrafo de contexto** explicando o que este relatório documenta e por que existe
3. **Resultados concretos** com números reais
4. **Estado metodológico** — o que está bloqueado e por quê, em frase única
5. **Próximos passos** — apenas se relevantes e conhecidos

O que o relatório **não** deve ter:

- disclaimer repetido em cada linha de tabela
- lista de 15 restrições em vez de 1 frase de contexto
- campo `training_ready=BLOCKED` sem explicação do que seria necessário para desbloquear
- título apenas com código de versão (`revp_v2ci`)

---

## 6. Estrutura de README de projeto

O README principal do repositório deve seguir este formato:

```
# REV-P
Descrição em 4–6 linhas.

## O que o projeto faz
## O que o projeto ainda não faz
## Estrutura do repositório
## Cadeia metodológica
## Dados e artefatos locais
## Estado atual
## Como reproduzir os relatórios públicos
## Limitações
## Próximos passos
```

O README não deve ter 16 seções com listas defensivas. Deve ter 9 seções com texto claro e direto.

---

## 7. Status técnicos — política de uso

Campos de status como `BLOCKED`, `PROTOCOL_VALIDATED_CANDIDATE_REFERENCE`, `TP2_CANDIDATE_ONLY` são **contratos técnicos** e não devem ser alterados em código, CSVs ou enums.

Em documentos textuais públicos (`.md`), esses status devem:
- aparecer em `código` inline quando citados como valores técnicos;
- ter explicação em português na mesma frase ou no parágrafo seguinte;
- nunca aparecer como única informação numa seção (sempre com contexto).

---

## 8. O que nunca afirmar

O REV-P nunca deve afirmar — em nenhum documento público:

- que detecta inundação ou deslizamento operacionalmente;
- que prediz risco de enchente;
- que tem ground truth validado em nível de patch;
- que os embeddings DINOv2 representam classes físicas de risco;
- que patches candidatos são exemplos positivos de treinamento;
- que a ausência de evidência é um negativo formal;
- que o projeto está pronto para produção ou uso operacional.

---

## 9. Quando os guardrails são ditos uma vez, não precisam ser repetidos

O princípio central é: **diga a restrição uma vez, claramente, no lugar certo**. Não repita o mesmo disclaimer em cada linha de tabela, cada relatório, cada seção de README.

O lugar certo para as restrições metodológicas centrais é:
- `README.md` (seção "O que o projeto ainda não faz")
- `outputs_public/model/NO_OPERATIONAL_TRAINED_MODEL.md`
- `final_guardrails_report.md`

A partir daí, os demais documentos podem referenciar esses arquivos em vez de repetir o texto.

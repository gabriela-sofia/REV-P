# Auditoria crítica de banca — marco MV1

> Documento independente de revisão crítica (papel de banca/revisor de TCC/artigo). Não altera artefatos do marco, não cria label, não cria negativo formal, não libera treino. Lê o estado e ataca metodologicamente a defensabilidade do marco antes de stage seletivo.

## 1. Escopo da auditoria crítica

Esta auditoria atua como banca técnica adversarial sobre o marco `marco/validacao-label-free-evidencia-estrutural-mv1`. O objetivo não é confirmar o trabalho, e sim tentar derrubá-lo: localizar fragilidades metodológicas, riscos de claim indevido, buracos de evidência, problemas de reprodutibilidade e pontos que precisam de correção antes de qualquer stage/commit.

Foram lidos os relatórios de fechamento do marco, validação label-free, auditoria temporal, protocolo fail-closed, as três políticas metodológicas (ontologia de labels, evidência negativa, anti-leakage), as tabelas de topologia/vizinhos/gates/guardrails e a navegação de evidências externas. A integração final do Codex (`revp_integracao_final_marco_mv1_maturidade_revisao_humana`) **ainda não existe no repositório** no momento desta auditoria; portanto, ela não é avaliada aqui e deverá ser reauditada quando for criada.

## 2. Estado atual do marco MV1

O marco consolida três trilhas herdadas: (a) restauração forense `v2dz-v2ef` (base auditável, sem ground truth operacional), (b) auditoria temporal (Trilha A bloqueada por metadados temporais e de nuvem insuficientes) e (c) validação label-free (Trilha B executada como piloto exploratório com `n=12` embeddings DINOv2 congelados, 768D, 4/4/4 entre Curitiba/Petrópolis/Recife).

Números confirmados nos artefatos: 164 patches auditados temporalmente, 508 assets, 12 embeddings válidos, 0 patches com ≥3 datas úteis, 156/164 sem metadados temporais, 162/164 sem metadados de nuvem. Protocolo fail-closed com gates G0–G8: G0/G1 parciais, G2–G8 bloqueados, treino supervisionado bloqueado, ground truth/positivos/negativos ausentes. Todos os itens de evidência externa têm `pode_virar_label_agora=false`.

A leitura honesta é: o marco é **conservador por construção**. Ele recusa fazer claims fortes. Essa é, simultaneamente, sua maior proteção contra a banca e sua maior limitação científica.

## 3. Contribuição real defensável

O que pode ser apresentado **hoje** como contribuição, sem overclaim:

1. **Protocolo fail-closed auditável de ground truth** — ontologia de estados de label, política de evidência negativa, política anti-leakage e gates G0–G8 que mantêm treino bloqueado por padrão. Esta é a contribuição mais forte e mais original do marco: uma engenharia metodológica que impede promoção indevida de evidência.
2. **Probe estrutural label-free** com encoder congelado (DINOv2) sobre patches Sentinel, com métricas derivadas rastreáveis (distâncias de cosseno, vizinhança), explicitamente rotulado como piloto exploratório.
3. **Cadeia de proveniência e quarentena de evidência externa** review-only, com hashes SHA256 e separação explícita entre contexto territorial e evento observado.

Nada disso é um resultado preditivo. É contribuição de **metodologia e infraestrutura**, defensável como tal.

## 4. Pontos fortes

- Separação disciplinada entre observação, interpretação e não-claim.
- Guardrails explícitos e repetidos: `unknown` não vira negativo, Curitiba não vira negativo, evidência contextual não vira label, suscetibilidade não vira evento, landslide não vira flood.
- Política anti-leakage que exige independência entre fonte de label e fonte de feature (ataca circularidade na raiz).
- Fail-closed real: nenhum gate libera treino; estados `desconhecido`/`bloqueado`/`excluido` nunca treinam.
- Métricas públicas derivadas + hashes em vez de vetores brutos.
- JSONs válidos (UTF-8) e CSVs parseáveis (verificado nesta auditoria).

## 5. Fragilidades metodológicas

1. **`n=12` é estatisticamente vazio.** 6 pares intra-cidade por cidade. Nenhum intervalo de confiança, teste de permutação ou bootstrap. Qualquer leitura além de "ilustração" é frágil.
2. **Separação topológica dentro do ruído.** Distância média intra-cidade `0.287` vs inter-cidades `0.310` — diferença de ~0.02. Pior: a variância intra é maior que o gap (Curitiba intra `0.177`, Recife intra `0.352`). A consistência de vizinho na mesma cidade é `0.417` (acaso ≈ 0.27 para 4/12). Há sinal fraco, não estrutura robusta.
3. **Confound temporal/nuvem não controlado.** Os 12 embeddings não têm data de aquisição nem fração de nuvem confirmadas por patch. A "topologia entre cidades" pode refletir condição de aquisição (data, sensor, nuvem) e não a cidade. Este é o ataque de banca mais perigoso ao único resultado numérico do marco.
4. **Gap de domínio DINOv2→Sentinel sem checagem.** DINOv2 foi pré-treinado em imagens naturais RGB; Sentinel é multiespectral. Não há sanity check de domínio nem documentação de como Sentinel foi convertido para a entrada do encoder (bandas, normalização). Os embeddings podem capturar artefato espectral em vez de semântica de cena.
5. **Pacote público não reprodutível isoladamente.** Os 12 embeddings e a matriz de similaridade derivam de inputs `local_runs/dino_embeddings/v1ge`, `v1gv` (git-ignored). Um revisor externo não reproduz o resultado só com os artefatos públicos.
6. **Fragmentação de proveniência amostral.** Convivem três universos — 59 (corpus original), 164 (auditoria temporal), 12 (embeddings) — sem uma reconciliação explícita de quais patches dos 164/59 são os 12.

## 6. Riscos fatais

Estes são riscos **fatais por natureza** (invalidariam a ciência se materializados). No estado atual estão **controlados por guardrail ou declarados como abertos sem reivindicação** — nenhum está violado. Eles só matam o trabalho se um claim for elevado:

- Promover o piloto `n=12` a evidência estatística final.
- Tratar a topologia DINOv2 como detecção de inundação.
- Converter `unknown`/ausência de evento/Curitiba em negativo (classe 0).
- Usar suscetibilidade (carta SGB) como evento observado.
- Cruzar coorte de movimento de massa (Petrópolis) com coorte de inundação.
- Usar evidência contextual ou a própria feature como fonte de label (circularidade).
- Liberar treino sem fechar os gates.

A banca vai testar exatamente esses pontos. A defesa atual é **textual** (guardrails em prosa/CSV), não programática.

## 7. Riscos corrigíveis

- Ausência de teste de significância/IC nas distâncias topológicas → adicionar permutação/bootstrap quando expandir.
- Confound temporal/nuvem → registrar data e fração de nuvem por patch dos 12 e controlar.
- Gap de domínio → executar checagem de sanidade DINOv2-Sentinel (controle nulo, composição de bandas, augmentações).
- Não-reprodutibilidade → publicar manifesto de reprodução com hashes e instruções; rotular `local_only` como quarentena.
- Downloads externos incompletos (CEMADEN, APAC, Copernicus, ANA por estação) → solicitação formal/LAI e consulta por estação.
- Excesso de linguagem defensiva → consolidar guardrails em uma seção e dar espaço à contribuição.
- Cabeçalho de branch desatualizado no relatório temporal (`analise/auditoria-...`).

## 8. Limitações aceitáveis

Estas são limitações que uma banca aceita **desde que declaradas** e que não invalidam a contribuição de metodologia:

- Ground truth operacional ausente (declarado, não simulado).
- Positivos e negativos formais ausentes (política exige evidência que ainda não existe).
- Trilha A bloqueada por metadados temporais (estado real do corpus).
- Nenhuma geometria de evento observado adquirida — apenas contexto territorial.
- Vetores brutos permanecem locais.

## 9. Claims permitidos

- "infraestrutura auditável" / "marco review-only"
- "validação label-free" / "piloto exploratório"
- "topologia dos embeddings" / "representação visual congelada"
- "evidência contextual como probe externo"
- "fila de revisão humana" / "priorização de revisão humana"
- "ground truth operacional ausente" / "treino supervisionado bloqueado"
- "limitação metodológica" / "risco corrigível"

## 10. Claims proibidos

- "modelo detecta inundação" / "acurácia de detecção"
- "modelo prediz suscetibilidade"
- "Curitiba é negativo" / "classe 0"
- "positivo confirmado" / "negativo confirmado"
- "ground truth fechado" / "treino liberado" / "validação operacional"

## 11. Ataques prováveis da banca

1. "Com `n=12` você não tem nada estatístico." (o ataque mais óbvio)
2. "Sua separação intra/inter é ~0.02, dentro da variância intra-cidade — onde está a estrutura?"
3. "Como você sabe que o embedding não está só agrupando por data/nuvem/sensor?"
4. "DINOv2 é treinado em fotos; por que vale em Sentinel multiespectral? Mostre o controle."
5. "Posso reproduzir? Os embeddings estão em pasta local ignorada pelo git."
6. "Suscetibilidade não é evento. Por que aparece como evidência?"
7. "Petrópolis é deslizamento; por que está na mesma análise de inundação?"
8. "Se Curitiba não tem evento, ela não é o seu negativo?" (armadilha)
9. "Qual é a contribuição, então, se não há detecção, predição nem ground truth?"

## 12. Respostas metodológicas recomendadas

1. "É um piloto exploratório explicitamente rotulado; não reivindicamos inferência estatística — a contribuição é metodológica."
2. "Reportamos a sobreposição e a variância intra; o sinal é fraco e declarado como tal, não como separação de classes."
3. "Reconhecemos o confound; o próximo passo registra data/nuvem por patch e adiciona controle — por isso não promovemos o resultado."
4. "Usamos o encoder congelado como probe; a validade de domínio é uma checagem pendente que bloqueia qualquer claim mais forte."
5. "O pacote público traz métricas derivadas e hashes; publicaremos manifesto de reprodução. Vetores brutos ficam em quarentena local por política do projeto."
6. "Suscetibilidade entra apenas como contexto, com overlay bloqueado; nunca como evento — está no guardrail e no protocolo."
7. "Coortes de movimento de massa e inundação são separadas na ontologia; não há cruzamento."
8. "Curitiba é contraste estrutural label-free; negativo formal exige evidência explícita de não-inundação, que não temos."
9. "A contribuição é um protocolo fail-closed auditável + um probe label-free; é metodologia e infraestrutura, não um classificador."

## 13. Pontos que precisam de ajuste textual

- Quantificar explicitamente a fragilidade topológica (sobreposição intra/inter, variância intra) no relatório de validação.
- Consolidar a repetição de guardrails (hoje 9–10 linhas em cada artefato) em uma seção única referenciável.
- Reconciliar em texto os três universos amostrais (59 / 164 / 12) e dizer quais patches são os 12.
- Corrigir o campo de branch no relatório de auditoria temporal.
- Rotular caminhos `local_only` como quarentena não distribuída sempre que aparecerem em artefato público.

## 14. Pontos que precisam de nova evidência

- Data de aquisição e fração de nuvem por patch (mínimo para os 12).
- Expansão de embeddings rastreáveis para ≥30 e idealmente 59 patches.
- Geometria de evento observado (footprint), não contexto territorial.
- Série hidrológica ANA por estação do Capibaribe; resposta formal CEMADEN; rapid mapping Copernicus.
- Checagem de sanidade de domínio DINOv2-Sentinel com controle nulo.

## 15. Pontos que precisam de revisão humana

- Submeter geometrias candidatas (malhas IBGE, bacias GeoCuritiba, carta SGB) à fila de revisão humana — todas hoje contexto, não evento.
- Adjudicação de qualquer candidato positivo futuro (G5 bloqueado).
- Revisar a separação de coortes landslide/flood antes de qualquer uso de Petrópolis.
- Marcar REC_00019 (amostra estrutural e candidato histórico forte) para separação explícita feature/label, evitando circularidade futura.

## 16. Pontos que bloqueiam treino

Todos os gates G2–G8 bloqueados; G0/G1 apenas parciais. Treino permanece corretamente bloqueado por: ausência de janela temporal fechada, ausência de geometria patch-evento, ausência de fonte de label independente, ausência de revisão humana, ausência de negativos formais e anti-leakage não aprovado por amostra. Nenhuma ação desta auditoria desbloqueia treino.

## 17. Pontos que bloqueiam ground truth operacional

Ground truth operacional ausente porque não há: evento observado com geometria vetorial e CRS fechado, janela temporal compatível por amostra, nem fonte de label independente adjudicada. A evidência externa adquirida é contexto territorial/suscetibilidade — nenhuma sustenta footprint de evento observado. Enquanto isso persistir, ground truth operacional permanece ausente (e deve ser declarado como tal, nunca simulado).

## 18. Recomendações antes de stage seletivo

1. **Pode-se fazer stage seletivo agora, com revisão manual** — não há claim operacional, label indevido, negativo formal indevido ou treino liberado no estado atual.
2. Stagear apenas os artefatos intencionais do marco; **não** usar `git add -A` (o working tree tem arquivos não relacionados e materiais `v2dz-v2ef`/restauração que devem ser avaliados à parte).
3. Antes de qualquer elevação de claim em artigo/slides: resolver confound temporal/nuvem, checagem de domínio DINOv2-Sentinel e reprodutibilidade.
4. Converter os guardrails críticos (unknown≠negativo, Curitiba≠negativo, contextual≠label, suscetibilidade≠evento) em **checagens programáticas** que falhem o teste se violadas — hoje a defesa é só textual.
5. Aplicar os ajustes textuais da seção 13 (baratos e de alto impacto na percepção da banca).

## 19. Conclusão crítica

O marco MV1 é **defensável como marco review-only de metodologia e infraestrutura**, não como resultado científico preditivo. Sua força é a disciplina fail-closed; sua fraqueza é a fragilidade do único resultado numérico (`n=12`, separação dentro do ruído, confounds não controlados) e a não-reprodutibilidade isolada do pacote público.

Veredito de banca: **aprovado com ressalvas para stage seletivo + revisão manual**, com a condição explícita de que nenhum claim seja elevado acima de "piloto exploratório / infraestrutura auditável". O maior risco metodológico é o confound temporal/nuvem somado ao gap de domínio DINOv2-Sentinel, que tornam a topologia interpretável apenas como ilustração. O maior ajuste textual necessário é quantificar essa fragilidade em vez de descrevê-la apenas qualitativamente. A contribuição real, hoje, é o **protocolo fail-closed auditável** — e é nele que o trabalho deve se apoiar na defesa.
</content>
</invoke>

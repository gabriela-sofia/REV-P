# Reconciliação: rascunho do Overleaf (`planejamento02_2.pdf`) × main.tex v12 (REV-P)

**Data**: 2026-08-21
**Escopo**: fundir suas edições novas no Overleaf (autoria em equipe, abertura
reescrita, três notas suas entre colchetes) com o que o `main.tex` v12 do
REV-P já tinha fechado e o rascunho ainda descrevia como plano (E3, E4, E5,
E6). Segue o mesmo formato de `NOTA_versoes.md` — nada foi mudado em
silêncio.

---

## 1. Antes de qualquer coisa: dois pontos que preciso que você decida, eu não posso

**A autoria mudou para quatro pessoas, e o texto inteiro ainda descreve trabalho individual.** O rascunho do Overleaf lista você, Bárbara, Nicole e Rafaella, Turma 8A, Equipe 8 — mas o parágrafo de infraestrutura diz "o trabalho é individual... sem divisão de carga entre pessoas, o paralelismo vem da sobreposição de etapas", e o cronograma se justifica do mesmo jeito. Todo o histórico de commits do REV-P, sem exceção, é assinado só por você. Não sei se a equipe é administrativa da disciplina (você segue sendo a autora técnica) ou se o trabalho de fato vai se dividir daqui pra frente — e não posso adivinhar isso. **Mantive "trabalho individual"** porque é o que os commits sustentam até hoje, mas se as outras três vão de fato contribuir tecnicamente, essa frase (e a lógica do cronograma) precisa mudar antes de entregar — um professor vai notar a contradição entre a autoria e o texto.

**"Equipe 8 N"**: tratei o "N" depois do 8 como resíduo do placeholder antigo (`\textcolor{red}{N}`) e removi, deixando só "Equipe 8". Se "8N" era o código real da equipe, é só devolver o N.

---

## 2. O que estava desatualizado no rascunho do Overleaf, e por quê

O rascunho divergiu do REV-P a partir de algum ponto antes da v10 (20/08) — a abertura da Seção I já estava mais à frente que a v12 (sua reescrita é a que ficou), mas a Seção III (Etapas) e um trecho da Seção II ainda descreviam E3, E4, E5 e E6 como planejadas, com números que já tinham sido corrigidos no REV-P.

| Onde | O rascunho do Overleaf dizia | O real (commits de 20/08) | O que a versão entregue faz |
|---|---|---|---|
| §II, Material externo | "Falta a validação prospectiva e a camada de explicação" | E4 rodou (`MOD-PROSP-02`); a camada de explicação já existe como gerador por regras (E6) | Frase trocada pela sentença correta, já usada na v12 |
| §II, caracterização dos dados | "contraste de HAND... de cerca de 3\,m em planície a quase 28\,m em serra" | Na base harmonizada: 2,5\,m e 34,5\,m (`a1109ef`) | Números corrigidos |
| §II, Serviço | "Curitiba tem modelo próprio, mas colapso... Petrópolis segue com zero pontos rotulados... Estender às demais é o objetivo declarado de E5" (E5 no futuro) | E5 já rodou: grade nas três regiões, Petrópolis responde `transferência_sem_referência_local`, não mais `region_not_supported` | Parágrafo reescrito com o comportamento real de cada região e os cinco *gates* nomeados |
| §III, E3 | Sem "(concluída)", entregável em branco | `MOD-SERRA-03`: serra AUC 0,7916, planície 0,7245, transferência 0,7957; correção do orçamento de EPV do estrato íngreme (1 variável, não 2 nem 4) | Parágrafo completo com coeficientes e IC |
| §III, E4 | "Teste ainda não realizado... cerca de 158 datas em 25 anos" | `MOD-PROSP-02`: 110 datas, 201 eventos, 8 cortes, AUC médio 0,7992; Curitiba não sustenta o teste | Parágrafo reescrito com os números reais |
| §III, E5 | Entregável e evidência em branco (etapa futura) | Grade a 120\,m: 56.666/65.275/172.015 células; distância de domínio medida | Parágrafo completo |
| §III, E6 | "Expor o modelo pelo contrato e testar a explicação" (futuro) | Contrato roda como função pura, cinco *gates*, 29 testes; só falta o transporte HTTP | Marcado "contrato executável", com o que falta nomeado |
| §III, Riscos, E4 | "se o desempenho temporal não superar o acaso, o resultado é publicado como está" (hipotético) | A regra já foi cumprida — Curitiba não sustentou o teste e isso está relatado | Frase passa ao passado |
| Fig. 1, caixa da coluna 1 | `PE3D \| GLO-30 \| CHIRPS` | Chuva é fonte única ERA5-Land desde 16/08; CHIRPS foi o produto que a auditoria de confundimento retirou | Caixa corrigida para `ERA5-LAND` (a v12 já tinha essa correção; o PDF do Overleaf ainda não) |
| Fig. 1, caixa do contrato | *gates*: geometria, CRS, região, modelo, variáveis (5, sem domínio) | O texto (§II) já nomeia cinco *gates*: geometria, região, modelo, variáveis **e domínio** — o mais novo, motivado pela distância de elevação de Curitiba | Caixa da figura alinhada ao texto (essa mesma inconsistência já existia dentro da própria v12 — a figura nunca tinha sido atualizada) |
| Tabela I, linha 2 | "CHIRPS e ERA5-Land" como se as duas estivessem em uso | Fonte única ERA5-Land; CHIRPS citado só como o produto removido | Célula reescrita |
| Tabela I, linha 2, chuva | "Chuva recuperada para 100%" | 99,99\% (não 100%, tem uma fração residual) | Número ajustado; mesma correção aplicada no corpo do texto |

---

## 3. As suas três notas entre colchetes, resolvidas

**"(tem que organizar melhor como que vai falar do peso do dino e seus embeddings no projeto)"** — §I: adicionei uma frase que nomeia o DINOv2 como codificador congelado, nunca ajustado aos dados, usado só pra medir similaridade e alimentar a fila de revisão, com a rota de preditor encerrada depois de três tentativas nulas.

**"(EU QUERO QUE AQUI ELABORE SOBRE A PRESENÇA DO DINO)"** — §II, Serviço: adicionei uma frase que liga o DINOv2 diretamente ao contrato — ele entra só como evidência anexada à explicação (ilustra a resposta pra quem revisa), nenhum *gate* o consulta, e ele nunca desloca o escore, remetendo aos dois limites já desenhados na Fig. 1.

**"[elaborar o q sao as unidades ou usar pontos q nem recife se for]"** — os 1.471 de Curitiba são **grupos de validação**, não pontos individuais (o conjunto bruto tem mais pontos que isso; eles se agrupam por evento, o mesmo princípio de agrupamento que a Seção II já declara para todo o projeto). Mantive o número 1.471 — não troquei por um novo, porque a fonte que o define (`ext_modelo_fluvial_multirregiao_v1.md`, 12/08) é anterior à harmonização final de 20/08, e eu não tenho um recontagem mais recente pra confirmar se mudou. Se você tiver o número atualizado da tabela única, vale conferir antes de entregar.

**Bônus, não pedido, mas decisão de E2/§II**: acrescentei uma frase curta explicando "ajuste fluvial" (o ajuste que estima suscetibilidade a enchente, não outro mecanismo como movimento de massa) e "cadeia de terreno global" (variável de um produto genérico, não rederivada pela cadeia própria do projeto) — era exatamente o ponto que você marcou pra resolver "na caracterização dos dados".

**Também reintegrado, sem que você tivesse marcado**: o parágrafo corrido de "O que a caracterização dos dados impõe" ainda não tinha o achado de que a chuva não discrimina na escala do modelo (~11\,km vs. comparação por evento) — isso já estava na Tabela I da própria v12, mas não tinha sido replicado no texto corrido. Corrigido para os dois ficarem consistentes.

---

## 4. O que continua genuinamente em aberto — isto sim é seu, não meu

- **Paginação.** Este documento cresceu desde a v9 (E3/E4/E5/E6 saindo do papel) e agora cresceu mais um pouco com as três elaborações que você pediu. Ninguém confirmou 3 páginas no Overleaf ainda. Se estourar, a ordem de corte já definida (`NOTA_versoes.md` §15) continua valendo: (1) lista de limitações de Recife no E6, (2) frase do portão de domínio em §II, (3) ICs dos coeficientes no E3, (4) frase de célula vazia no E5 — e, se ainda faltar espaço, a explicação de "ajuste fluvial"/"cadeia de terreno global" que acabei de adicionar é a mais cortável das três elaborações novas, porque é a única puramente definicional.
- **E6 sem transporte HTTP** — contrato pronto, falta expor como serviço.
- **Decisão sobre a chuva** (commit `9d24d61`, 20/08) — a chuva não discrimina na escala do modelo; qual rota tomar a partir disso é decisão sua, ainda não tomada.
- **Autoria e "trabalho individual"** — Seção 1 acima.

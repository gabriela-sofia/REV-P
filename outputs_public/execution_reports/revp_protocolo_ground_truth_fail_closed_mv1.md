# Protocolo ground truth fail-closed MV1

## 1. Escopo
Este pacote formaliza uma camada pública, auditável e fail-closed para definir quando um patch poderá, no futuro, virar amostra de treino. Ele não treina modelo, não cria labels reais, não cria positivos formais reais, não cria negativos formais reais e não promove ground truth operacional.

## 2. Por que este protocolo é necessário
O marco MV1 demonstrou uma infraestrutura auditável e um piloto label-free, mas também preservou bloqueios centrais. Sem um protocolo explícito, há risco de transformar evidência contextual, topologia DINOv2, ausência de evento ou contraste regional em alvo supervisionado.

## 3. Relação com o marco label-free MV1
A validação label-free MV1 permanece exploratória. Os embeddings DINOv2 congelados servem para topologia estrutural, vizinhança e priorização de revisão humana. Eles não provam inundação e não criam label.

## 4. Relação com a curadoria externa em andamento
A curadoria externa pode produzir insumos futuros, mas este pacote não depende de downloads externos, não altera artefatos de navegação e não altera quarentena externa. Qualquer evidência externa futura deve passar pelos gates antes de uso supervisionado.

## 5. Ontologia de estados de label
A ontologia pública define `positivo_ouro`, `positivo_prata`, `negativo_pareado`, `negativo_dificil`, `desconhecido`, `excluido` e `bloqueado`. Apenas estados futuros com evidência formal e todos os gates satisfeitos podem ser considerados para treino; no estado atual, nenhum treino está liberado.

## 6. Política de evidência negativa
Negativo formal exige evidência explícita de não inundação. Ausência de evidência, ausência de evento, unknown e Curitiba por default não satisfazem essa política.

## 7. Separação entre fonte de label e fonte de feature
Fonte de label deve ser independente de asset, embedding, feature, score contextual e métrica usada pelo modelo. Evidência contextual pode orientar revisão humana, mas não preencher alvo supervisionado.

## 8. Gates de readiness para treino
Os gates G0 a G8 controlam identidade de patch, identidade de asset, janela temporal, geometria, fonte de label independente, revisão humana, política de negativos, anti-leakage e liberação final.

## 9. Estado atual dos gates
Gates bloqueados ou não satisfeitos: G0_patch_id_valido, G1_asset_id_valido, G2_janela_temporal_fechada, G3_geometria_espacial_fechada, G4_fonte_label_independente, G5_revisao_humana_completa, G6_politica_negativo_satisfeita, G7_anti_leakage_aprovado, G8_liberado_para_treino_true. Nenhum gate final libera treino supervisionado.

## 10. Por que treino ainda está bloqueado
Treino supervisionado permanece bloqueado porque ground truth operacional está ausente, positivos formais estão ausentes, negativos formais estão ausentes, revisão humana não está completa, geometria e janela temporal não estão fechadas e anti-leakage ainda não foi aprovado por amostra.

## 11. Como evidências externas poderão ser usadas futuramente
Evidências externas poderão ser usadas como fonte independente ou como contexto de revisão apenas se sua origem, data, geometria, licença, hash/proveniência e independência em relação às features forem auditáveis.

## 12. Como evitar circularidade
A matriz label-feature deve registrar, para cada amostra, se a fonte de label aparece como feature. Se houver sobreposição, a amostra deve ficar bloqueada.

## 13. Como evitar `unknown = negative`
`unknown` deve permanecer desconhecido. Lacuna documental, ausência de registro e ausência de evento não criam negativo formal.

## 14. Como evitar Curitiba como negativo por default
Curitiba pode ser usada como contraste estrutural label-free, mas não vira negativo formal por localização. Qualquer negativo futuro exige evidência explícita de não inundação.

## 15. Como tratar landslide vs flood
Eventos de movimento de massa e inundação devem permanecer separados na ontologia. Um evento landslide não deve ser usado como positivo de flood, nem como negativo automático de flood, sem política explícita e revisão humana.

## 16. Próximos passos para ground truth real
Expandir evidências independentes, fechar geometria e janela temporal, formalizar revisão humana/adjudicação, aplicar política de negativos, executar auditoria anti-leakage por amostra e só depois avaliar feature table multimodal.

## 17. Guardrails preservados
- unknown nunca vira negativo
- ausência de evento nunca vira negativo formal
- Curitiba nunca vira negativo formal por default
- evidência contextual nunca vira label
- DINOv2 não prova inundação
- restauração v2dz-v2ef não vira ground truth operacional
- auditoria temporal não libera Trilha A
- piloto label-free com n=12 não vira evidência estatística final
- nenhum treino supervisionado é liberado
- fonte de label e fonte de feature devem ser independentes

## Status consolidado
- Branch: `marco/validacao-label-free-evidencia-estrutural-mv1`
- Status do protocolo: `PROTOCOLO_FAIL_CLOSED_FORMALIZADO_SEM_LIBERAR_TREINO`
- Treino supervisionado: `bloqueado`
- Ground truth operacional: `ausente`
- Positivos formais: `ausente`
- Negativos formais: `ausente`

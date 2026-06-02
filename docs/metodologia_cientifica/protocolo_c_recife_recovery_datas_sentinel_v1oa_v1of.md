# Protocolo C - Recife recovery datas Sentinel v1oa-v1of

O bloco v1oa-v1of tenta recuperar scene_date REC a partir de manifests, filenames, sidecars e metadata local sem processar pixels.

Somente scene_date confirmada por fonte aceitavel entra no rematch temporal. Datas ausentes, conflitantes ou de baixa confianca mantem bloqueio.

C4 permanece fechado com formal_negative_count=0; DINO permanece REVIEW_ONLY_REPRESENTATION e nao cria label.

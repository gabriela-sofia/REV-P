# v2az - Como chegar ao TP4

1. Preencha boundary real do patch.
2. Preencha poligono observado real do evento.
3. Rode `python scripts/run_v2az_turning_point_replay_orchestrator.py --mode dry_run`.
4. Se feeds validos aparecerem, rode com `--mode replay`.
5. Revise o overlay v2au.
6. Aceite no maximo `C4_CANDIDATE_REQUIRES_HUMAN_REVIEW`.
7. Nunca treine modelo nesta etapa.

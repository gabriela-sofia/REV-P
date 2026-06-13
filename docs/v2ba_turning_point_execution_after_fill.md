# Execution after filling

1. Place the real external file and provenance.
2. Run `python scripts/run_v2ba_minimal_real_geometry_acquisition_workbench.py --mode validate`.
3. If feeds exist, run `python scripts/run_v2az_turning_point_replay_orchestrator.py --mode dry_run`.
4. Run v2az `replay` only after reviewing inputs.
5. Review v2au output.
6. Stop at `C4_CANDIDATE_REQUIRES_HUMAN_REVIEW`; create no label.

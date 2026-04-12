# Repo Keep Delete Plan

Keep in repo:

Core model code
- mlb_k_model/
- run_daily_board.py
- train_system.py
- run_quant_audit.py
- run_model_vs_market_eval.py
- run_model_vs_market_mainline_eval.py
- fit_mainline_pmf_postcal.py
- fit_residual_mainline_calibrator.py

Reproducible pipeline scripts
- run_pregame_pipeline.sh
- build_actionable_card.py
- build_production_card.py
- build_conservative_production_card.py

Repo docs
- README.md
- REPO_KEEP_DELETE_PLAN.md

Required local artifacts
- data_rebuild/pitcher_k_system.joblib
- data_rebuild/games.parquet
- data_rebuild/players.parquet
- data_rebuild/plate_appearances.parquet
- data_rebuild/external/

Archive, do not immediately delete
- any file matching .bak_before_
- proof scripts and proof outputs created during debugging
- old dated board outputs under outputs/board/
- one-off scratch scripts used only during patching
- temporary proof files used to test API flows

Delete candidates after verification
- stale proof files in outputs/
- empty or obsolete board output folders
- duplicate patch scripts fully replaced by run_pregame_pipeline.sh
- superseded backup files

Do not put in public GitHub
- raw API keys
- secrets files
- large private data snapshots unless intentionally published
- unnecessary local caches
- personal filesystem metadata files

Future recommended structure

mlb_pitcher_k_model_elite/
  mlb_k_model/
  scripts/
    run_pregame_pipeline.sh
    retrain_and_recalibrate.sh
  outputs/
  data_rebuild/
  README.md
  REPO_KEEP_DELETE_PLAN.md

Cleanup rule:
If a file is not needed for model training, auditing, board generation, production-card generation, or reproducibility, archive it or remove it.

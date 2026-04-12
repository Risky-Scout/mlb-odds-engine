# MLB Pitcher Strikeout Model

A transparent, reproducible MLB pitcher strikeout modeling pipeline.

Main purpose:
- estimate pitcher strikeout distributions
- convert distributions into market-line probabilities
- compare model pricing to sportsbook pricing
- filter bets through a conservative, quote-confirmed production card

Main output:
outputs/board/YYYY-MM-DD/pregame_conservative_production_card.csv

Supporting outputs:
- pregame_conservative_production_pool.csv
- pregame_conservative_production_card_report.md
- pregame_summary.csv
- pregame_candidate_bets.csv
- pregame_line_grid.csv
- pregame_exact_pmf.csv

Daily run:
./run_pregame_pipeline.sh 2026-04-11

Pipeline steps:
1. pull current MLB pitcher strikeout quotes
2. build a quote-universe pregame board
3. score pitcher strikeout PMFs and threshold probabilities
4. apply the strict EV gate
5. create a conservative production card with:
   - near-main-line bets only
   - cross-book confirmation
   - one best vendor per exact bet
   - one best bet per pitcher

Expected local inputs:
- data_rebuild/games.parquet
- data_rebuild/players.parquet
- data_rebuild/plate_appearances.parquet
- data_rebuild/pitcher_k_system.joblib
- data_rebuild/external/

Environment:
- ODDS_API_KEY
- BDL_API_KEY if required by the board workflow

Near-term roadmap:
- add a one-command retrain and recalibration pipeline
- add a live-board workflow
- add a sizing and risk layer
- tighten packaging for GitHub portfolio presentation

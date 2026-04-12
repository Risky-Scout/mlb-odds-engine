# MLB Odds Engine

Transparent, reproducible MLB probability and pricing models for pregame and live betting markets.

## What this repository does

This repository is built around a sports-quant workflow:

- estimate true outcome distributions
- convert those distributions into market-line probabilities
- calibrate market-facing probabilities
- identify positive expected value bets
- produce conservative, reproducible betting cards

## Current status

The current locked baseline is a v1 pregame MLB pitcher strikeout engine.

Current production artifact:
- outputs/board/YYYY-MM-DD/pregame_conservative_production_card.csv

## Core workflows

### Daily pregame pipeline
./run_pregame_pipeline.sh 2026-04-11

### Full retrain and recalibration pipeline
./retrain_and_recalibrate.sh 2026-04-11

## Core project files

- mlb_k_model/
- run_daily_board.py
- run_pregame_pipeline.sh
- retrain_and_recalibrate.sh
- train_system.py
- run_quant_audit.py
- run_model_vs_market_mainline_eval.py
- fit_mainline_pmf_postcal.py
- fit_residual_mainline_calibrator.py

## Validation philosophy

The workflow is evaluated with:

- PA model calibration and discrimination
- PMF quality metrics
- strict main-line model-versus-market evaluation
- EV-gate validation
- conservative production-card filtering

## Repository structure

- mlb_k_model/ : core modeling code
- artifacts/v1_baseline/ : preserved benchmark outputs
- README.md : repo overview
- METHODOLOGY.md : methodology summary
- REPO_KEEP_DELETE_PLAN.md : cleanup and organization plan

## Current development priority

Build the live and in-play MLB pitcher strikeout workflow on the live-inplay-pitcher-strikeouts branch without destabilizing the locked pregame baseline.

## Long-term goal

Extend the same architecture across additional MLB betting markets while preserving transparency, reproducibility, and calibration discipline.

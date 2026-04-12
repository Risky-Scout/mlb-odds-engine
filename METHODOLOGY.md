# Methodology

## Objective

Build MLB betting models that estimate true probabilities and distributions better than the market, remain well-calibrated, and identify positive expected value bets in pregame and live settings.

## Current scope

This repository currently contains a working v1 pregame MLB pitcher strikeout odds engine. The pipeline is designed to be transparent, reproducible, and extensible to additional MLB prop and game markets.

## Modeling framework

The current pregame pitcher strikeout workflow has four layers.

### 1. True-outcome model

The base model estimates the pitcher strikeout distribution for a game or game state. It produces a full probability mass function over strikeout outcomes rather than only a point estimate.

### 2. Market-pricing layer

The strikeout distribution is converted into threshold probabilities for sportsbook strikeout lines. These probabilities are compared directly against sportsbook prices and no-vig market probabilities.

### 3. Calibration layer

A calibration stack is applied to improve threshold pricing quality while preserving the underlying distribution model. The workflow uses:
- PA-level calibration assessment
- PMF post-calibration
- residual main-line calibration

### 4. Bet-selection layer

Candidate bets are filtered through a strict EV gate and then a conservative production-card layer. The current conservative card requires:
- strict-gate approval
- sufficient quote depth
- near-main-line pricing
- cross-book confirmation
- one best vendor per exact bet
- one best bet per pitcher

## Daily pregame workflow

The pregame workflow is designed to run from one command:
- fetch current quote universe
- build quote-universe board
- score distributions and threshold probabilities
- apply calibration-aware pricing
- generate conservative production card

Main daily script:
- run_pregame_pipeline.sh

Main final output:
- outputs/board/YYYY-MM-DD/pregame_conservative_production_card.csv

## Retraining and recalibration workflow

The full lifecycle is intended to be reproducible:
- retrain base model
- run quant audit
- run market-vs-model evaluation
- fit PMF post-calibration
- fit residual main-line calibration
- rerun final audit
- rebuild conservative production card

Main lifecycle script:
- retrain_and_recalibrate.sh

## Validation philosophy

The system is evaluated with:
- PA model metrics
- PMF metrics
- strict main-line model-vs-market evaluation
- holdout gate validation
- conservative production-card filtering

## Current status

The current committed baseline is a v1 pregame MLB pitcher strikeout model that:
- produces full strikeout distributions
- prices strikeout thresholds
- outperforms the market on the matched strict main-line evaluation used in this workflow
- generates a conservative final pregame card

## Next development priority

The next major build is the live or in-play MLB pitcher strikeout model.

## Repo philosophy

This repository is intended to demonstrate:
- transparent sports-quant modeling
- reproducible research-to-production workflow
- disciplined probability and market-pricing methodology
- extensibility across multiple MLB betting markets

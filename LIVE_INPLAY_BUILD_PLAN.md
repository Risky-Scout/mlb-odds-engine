# Live In-Play Build Plan

## Objective

Build a live and in-play MLB pitcher strikeout model that estimates true probabilities and distributions better than the market, remains well-calibrated, and identifies positive expected value bets during games.

## Design principle

Do not destabilize the locked pregame baseline.

The live build should extend the current architecture:
- true-outcome model
- market-pricing layer
- calibration layer
- bet-selection layer

## Phase 1

1. Build live game-state representation
2. Join live quote universe
3. Update PMF from current state instead of pregame state
4. Price live threshold markets
5. Emit live candidate rows

## Phase 2

1. Build live model-versus-market evaluator
2. Add live calibration diagnostics
3. Add conservative live production-card logic

## Phase 3

1. Create one-command live pipeline
2. Add later sizing and risk layer

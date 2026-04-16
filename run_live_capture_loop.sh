#!/usr/bin/env bash
set -euo pipefail

DATE_ARG="${1:-$(TZ=America/New_York date +%F)}"
INTERVAL_MIN="${2:-10}"
RUNS="${3:-6}"

cd "/Users/josephshackelford/woo_models/CyberDuck/MLB Pages/Pitcher Strike Outs Model/Version 1/mlb_pitcher_k_model_elite"
source .venv/bin/activate

if [[ -z "${ODDS_API_KEY:-}" ]]; then
  echo "ODDS_API_KEY is not set"
  exit 1
fi

echo "DATE=$DATE_ARG"
echo "INTERVAL_MIN=$INTERVAL_MIN"
echo "RUNS=$RUNS"

for ((i=1; i<=RUNS; i++)); do
  echo
  echo "===== live capture run $i / $RUNS ====="
  date -u

  python run_live_inplay_pipeline.py \
    --date "$DATE_ARG" \
    --data-dir data_rebuild \
    --system-path data_rebuild/pitcher_k_system.joblib \
    --out-dir outputs/board \
    --predictions-dir Predictions \
    --refresh-board

  if [[ "$i" -lt "$RUNS" ]]; then
    echo "Sleeping ${INTERVAL_MIN} minutes..."
    sleep "$((INTERVAL_MIN * 60))"
  fi
done

echo
echo "===== build live eval dataset ====="
python build_live_eval_dataset.py \
  --date "$DATE_ARG" \
  --data-dir data_rebuild \
  --out-dir outputs/live_eval

echo
echo "===== summarize live eval + calibration prep ====="
python summarize_live_eval_and_prep_calibration.py \
  --date "$DATE_ARG" \
  --eval-dir outputs/live_eval

echo
echo "===== final summary ====="
sed -n '1,120p' outputs/live_eval/"$DATE_ARG"/live_eval_metrics_summary.json

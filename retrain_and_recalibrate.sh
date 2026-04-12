#!/usr/bin/env bash
set -euo pipefail

BOARD_DATE="${1:-$(date +%F)}"
DATA_DIR="${DATA_DIR:-data_rebuild}"
SYSTEM_PATH="${SYSTEM_PATH:-${DATA_DIR}/pitcher_k_system.joblib}"
OUT_BASE="${OUT_BASE:-outputs}"
STAMP="$(date +%Y%m%d_%H%M%S)"

BASE_AUDIT_DIR="${OUT_BASE}/audit_retrain_base_${STAMP}"
BASE_MAINLINE_DIR="${OUT_BASE}/model_vs_market_mainline_retrain_base_${STAMP}"
PMF_POSTCAL_DIR="${OUT_BASE}/pmf_postcal_retrain_${STAMP}"
POST_PMF_MAINLINE_DIR="${OUT_BASE}/model_vs_market_mainline_after_pmf_postcal_${STAMP}"
RESIDUAL_CAL_DIR="${OUT_BASE}/residual_mainline_calibration_${STAMP}"
FINAL_MAINLINE_DIR="${OUT_BASE}/model_vs_market_mainline_after_residual_cal_${STAMP}"
FINAL_AUDIT_DIR="${OUT_BASE}/audit_retrain_final_${STAMP}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

echo "==> Retrain and recalibrate pipeline"
echo "    BOARD_DATE=${BOARD_DATE}"
echo "    DATA_DIR=${DATA_DIR}"
echo "    SYSTEM_PATH=${SYSTEM_PATH}"
echo "    OUT_BASE=${OUT_BASE}"
echo "    STAMP=${STAMP}"

echo
echo "==> Step 1: Retrain base system"
python train_system.py \
  --data-dir "${DATA_DIR}" \
  --out-path "${SYSTEM_PATH}"

echo
echo "==> Step 2: Base quant audit"
python run_quant_audit.py \
  --data-dir "${DATA_DIR}" \
  --system-path "${SYSTEM_PATH}" \
  --out-dir "${BASE_AUDIT_DIR}"

echo
echo "==> Step 3: Base strict main-line evaluation"
python run_model_vs_market_mainline_eval.py \
  --data-dir "${DATA_DIR}" \
  --system-path "${SYSTEM_PATH}" \
  --out-dir "${BASE_MAINLINE_DIR}"

BASE_ROWS="${BASE_MAINLINE_DIR}/model_vs_market_mainline_rows.parquet"

echo
echo "==> Step 4: Fit PMF post-calibration"
python fit_mainline_pmf_postcal.py \
  --data-dir "${DATA_DIR}" \
  --rows-path "${BASE_ROWS}" \
  --system-path "${SYSTEM_PATH}" \
  --out-dir "${PMF_POSTCAL_DIR}"

echo
echo "==> Step 5: Strict main-line evaluation after PMF post-calibration"
python run_model_vs_market_mainline_eval.py \
  --data-dir "${DATA_DIR}" \
  --system-path "${SYSTEM_PATH}" \
  --out-dir "${POST_PMF_MAINLINE_DIR}"

POST_PMF_ROWS="${POST_PMF_MAINLINE_DIR}/model_vs_market_mainline_rows.parquet"

echo
echo "==> Step 6: Fit residual main-line calibrator"
python fit_residual_mainline_calibrator.py \
  --rows-path "${POST_PMF_ROWS}" \
  --system-path "${SYSTEM_PATH}" \
  --out-dir "${RESIDUAL_CAL_DIR}"

echo
echo "==> Step 7: Final strict main-line evaluation after residual calibration"
python run_model_vs_market_mainline_eval.py \
  --data-dir "${DATA_DIR}" \
  --system-path "${SYSTEM_PATH}" \
  --out-dir "${FINAL_MAINLINE_DIR}"

echo
echo "==> Step 8: Final quant audit"
python run_quant_audit.py \
  --data-dir "${DATA_DIR}" \
  --system-path "${SYSTEM_PATH}" \
  --out-dir "${FINAL_AUDIT_DIR}"

echo
echo "==> Step 9: Build fresh conservative pregame card"
./run_pregame_pipeline.sh "${BOARD_DATE}"

echo
echo "==> Retrain and recalibration pipeline complete"
echo "    Base audit:           ${BASE_AUDIT_DIR}"
echo "    Base main-line eval:  ${BASE_MAINLINE_DIR}"
echo "    PMF postcal:          ${PMF_POSTCAL_DIR}"
echo "    Post-PMF eval:        ${POST_PMF_MAINLINE_DIR}"
echo "    Residual cal:         ${RESIDUAL_CAL_DIR}"
echo "    Final main-line eval: ${FINAL_MAINLINE_DIR}"
echo "    Final audit:          ${FINAL_AUDIT_DIR}"
echo "    Final card:           outputs/board/${BOARD_DATE}/pregame_conservative_production_card.csv"

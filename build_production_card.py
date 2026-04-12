from pathlib import Path
import numpy as np
import pandas as pd

BOARD_DATE = "2026-04-11"
MAX_BETS = 12
MIN_VENDOR_COUNT = 4
MIN_EDGE = 0.20
MIN_CONF_DIST = 0.10
MAX_MAINLINE_DISTANCE = 1.0
MIN_EXACT_BOOKS = 2

base = Path("outputs/board") / BOARD_DATE
cand_path = base / "pregame_candidate_bets.csv"
summary_path = base / "pregame_summary.csv"

if not cand_path.exists():
    raise SystemExit(f"Missing candidate file: {cand_path}")
if not summary_path.exists():
    raise SystemExit(f"Missing summary file: {summary_path}")

cand = pd.read_csv(cand_path)
summary = pd.read_csv(summary_path)

if cand.empty:
    raise SystemExit("pregame_candidate_bets.csv is empty")

required = [
    "game_id", "pitcher_id", "pitcher_name", "vendor", "line_value", "best_side",
    "best_ev", "gate_conf_dist",
    "model_over_prob", "market_over_prob", "model_under_prob", "market_under_prob",
    "over_odds", "under_odds", "passes_strict_ev_gate", "action_tier",
]
missing = [c for c in required if c not in cand.columns]
if missing:
    raise SystemExit(f"Missing required columns in candidate bets: {missing}")

for c in [
    "game_id", "pitcher_id", "line_value", "best_ev", "gate_conf_dist",
    "model_over_prob", "market_over_prob", "model_under_prob", "market_under_prob",
    "over_odds", "under_odds"
]:
    cand[c] = pd.to_numeric(cand[c], errors="coerce")

cand["passes_strict_ev_gate"] = cand["passes_strict_ev_gate"].astype(str).str.lower().isin(["true", "1", "yes"])
cand["action_tier"] = cand["action_tier"].astype(str)
cand["best_side"] = cand["best_side"].astype(str).str.lower().str.strip()
cand["vendor"] = cand["vendor"].astype(str).str.lower().str.strip()

summary_cols = [c for c in ["game_id", "pitcher_id", "vendor_count", "top_quote_count", "market_mean"] if c in summary.columns]
summary = summary[summary_cols].copy()
for c in ["game_id", "pitcher_id", "vendor_count", "top_quote_count", "market_mean"]:
    if c in summary.columns:
        summary[c] = pd.to_numeric(summary[c], errors="coerce")

df = cand.merge(summary, on=["game_id", "pitcher_id"], how="left")

# Strict gate first
df = df.loc[
    df["passes_strict_ev_gate"]
    & df["best_ev"].ge(MIN_EDGE)
    & df["gate_conf_dist"].ge(MIN_CONF_DIST)
    & df["vendor_count"].fillna(0).ge(MIN_VENDOR_COUNT)
].copy()

if df.empty:
    raise SystemExit("No rows survived strict gate + vendor depth filter")

# Main-line / near-main-line filter
df["mainline_distance"] = (df["line_value"] - df["market_mean"]).abs()
df = df.loc[df["mainline_distance"].fillna(999).le(MAX_MAINLINE_DISTANCE)].copy()

if df.empty:
    raise SystemExit("No rows survived near-main-line filter")

# Exact cross-book confirmation:
# same pitcher/game/line/side appears at >= 2 books
grp_cols = ["game_id", "pitcher_id", "line_value", "best_side"]
confirm = (
    df.groupby(grp_cols, as_index=False)
      .agg(
          exact_book_count=("vendor", "nunique"),
          mean_best_ev=("best_ev", "mean"),
          median_best_ev=("best_ev", "median"),
      )
)

df = df.merge(confirm, on=grp_cols, how="left")
df = df.loc[df["exact_book_count"].fillna(0).ge(MIN_EXACT_BOOKS)].copy()

if df.empty:
    raise SystemExit("No rows survived cross-book confirmation filter")

# Build selected-side market probability
df["selected_market_prob"] = np.where(
    df["best_side"].eq("over"),
    df["market_over_prob"],
    df["market_under_prob"],
)

# Rank vendors for the same confirmed bet
# Higher EV is better, higher confidence is better, smaller implied market probability
# on the selected side generally means a better bettor price.
df["decision_score"] = df["best_ev"] + 0.25 * df["gate_conf_dist"]
df["production_score"] = (
    df["decision_score"]
    + 0.10 * df["exact_book_count"].fillna(0)
    + 0.05 * df["vendor_count"].fillna(0)
    - 0.10 * df["mainline_distance"].fillna(0)
)

df = df.sort_values(
    ["production_score", "best_ev", "gate_conf_dist", "selected_market_prob"],
    ascending=[False, False, False, True],
    kind="stable",
).reset_index(drop=True)

# Best vendor per exact confirmed bet
dedup_bets = (
    df.drop_duplicates(subset=["game_id", "pitcher_id", "line_value", "best_side"], keep="first")
      .sort_values(["production_score", "best_ev", "gate_conf_dist"], ascending=[False, False, False], kind="stable")
      .reset_index(drop=True)
)

# Safer production card: one best bet per pitcher, capped
prod = (
    dedup_bets.drop_duplicates(subset=["game_id", "pitcher_id"], keep="first")
             .sort_values(["production_score", "best_ev", "gate_conf_dist"], ascending=[False, False, False], kind="stable")
             .head(MAX_BETS)
             .reset_index(drop=True)
)

show_cols = [
    "game_id", "pitcher_id", "pitcher_name", "vendor",
    "line_value", "market_mean", "mainline_distance",
    "best_side", "best_ev", "decision_score", "production_score",
    "gate_conf_dist", "vendor_count", "top_quote_count", "exact_book_count",
    "model_over_prob", "market_over_prob", "model_under_prob", "market_under_prob",
    "selected_market_prob", "over_odds", "under_odds",
    "passes_strict_ev_gate", "action_tier",
]
show_cols = [c for c in show_cols if c in prod.columns]

prod = prod[show_cols].copy()
dedup_bets = dedup_bets[[c for c in show_cols if c in dedup_bets.columns]].copy()

prod_path = base / "pregame_production_card.csv"
pool_path = base / "pregame_production_pool.csv"
report_path = base / "pregame_production_card_report.md"

prod.to_csv(prod_path, index=False)
dedup_bets.to_csv(pool_path, index=False)

lines = []
lines.append(f"# Pregame Production Card - {BOARD_DATE}")
lines.append("")
lines.append(f"- Raw candidate rows: {len(cand)}")
lines.append(f"- After strict gate + vendor depth: {len(cand.merge(summary, on=['game_id','pitcher_id'], how='left').loc[(cand['passes_strict_ev_gate'].astype(str).str.lower().isin(['true','1','yes']))])}")
lines.append(f"- After near-main-line filter: {int((df['mainline_distance'] <= MAX_MAINLINE_DISTANCE).sum()) if len(df) else 0}")
lines.append(f"- After exact cross-book confirmation: {len(df)}")
lines.append(f"- Deduped production pool: {len(dedup_bets)}")
lines.append(f"- Final production card: {len(prod)}")
lines.append("")
lines.append(f"Rules: mainline_distance <= {MAX_MAINLINE_DISTANCE}, exact_book_count >= {MIN_EXACT_BOOKS}, one bet per pitcher, cap = {MAX_BETS}")
lines.append("")
lines.append("## Final production card")
lines.append("")
if len(prod):
    lines.append(prod.to_markdown(index=False))
else:
    lines.append("No production bets.")
report_path.write_text("\n".join(lines))

print("WROTE:")
print(prod_path)
print(pool_path)
print(report_path)
print("")
print("COUNTS:")
print("raw candidate rows:", len(cand))
print("after strict gate + vendor depth + near-main-line + cross-book confirm:", len(df))
print("dedup production pool:", len(dedup_bets))
print("final production card:", len(prod))
print("")
print("FINAL CARD:")
print(prod.to_string(index=False))

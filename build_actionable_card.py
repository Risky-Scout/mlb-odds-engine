from pathlib import Path
import pandas as pd

BOARD_DATE = "2026-04-11"
base = Path("outputs/board") / BOARD_DATE
cand_path = base / "pregame_candidate_bets.csv"

if not cand_path.exists():
    raise SystemExit(f"Missing candidate file: {cand_path}")

df = pd.read_csv(cand_path)
if df.empty:
    raise SystemExit("pregame_candidate_bets.csv is empty")

required = [
    "game_id", "pitcher_id", "pitcher_name", "vendor", "line_value",
    "best_side", "best_ev", "gate_conf_dist",
    "passes_strict_ev_gate", "action_tier",
    "model_over_prob", "market_over_prob",
    "model_under_prob", "market_under_prob",
    "over_odds", "under_odds",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

# Normalize types
for c in [
    "game_id", "pitcher_id", "line_value", "best_ev", "gate_conf_dist",
    "model_over_prob", "market_over_prob", "model_under_prob", "market_under_prob",
    "over_odds", "under_odds"
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["passes_strict_ev_gate"] = df["passes_strict_ev_gate"].astype(str).str.lower().isin(["true", "1", "yes"])
df["action_tier"] = df["action_tier"].astype(str)

# Keep only validated gate-passing rows
gated = df.loc[df["passes_strict_ev_gate"]].copy()
if gated.empty:
    raise SystemExit("No rows passed the strict EV gate")

# A stable decision score for ranking actionable bets
# Higher EV matters most; confidence distance breaks ties modestly.
gated["decision_score"] = gated["best_ev"] + 0.25 * gated["gate_conf_dist"]

# Deduplicate identical betting opportunities across vendors:
# keep the single best vendor per distinct (game, pitcher, side, line)
gated = gated.sort_values(
    ["decision_score", "best_ev", "gate_conf_dist"],
    ascending=[False, False, False],
    kind="stable",
).reset_index(drop=True)

dedup = (
    gated.drop_duplicates(
        subset=["game_id", "pitcher_id", "line_value", "best_side"],
        keep="first",
    )
    .sort_values(["decision_score", "best_ev", "gate_conf_dist"], ascending=[False, False, False], kind="stable")
    .reset_index(drop=True)
)

# Safer deployment card: at most one bet per pitcher
top1 = (
    dedup.drop_duplicates(subset=["game_id", "pitcher_id"], keep="first")
    .sort_values(["decision_score", "best_ev", "gate_conf_dist"], ascending=[False, False, False], kind="stable")
    .reset_index(drop=True)
)

# Friendly column order
show_cols = [
    "game_id", "pitcher_id", "pitcher_name", "vendor", "line_value", "best_side",
    "best_ev", "decision_score", "gate_conf_dist",
    "model_over_prob", "market_over_prob", "model_under_prob", "market_under_prob",
    "over_odds", "under_odds", "passes_strict_ev_gate", "action_tier",
]
show_cols = [c for c in show_cols if c in dedup.columns]

dedup = dedup[show_cols].copy()
top1 = top1[show_cols].copy()

dedup_path = base / "pregame_actionable_bets.csv"
top1_path = base / "pregame_actionable_top1_per_pitcher.csv"
report_path = base / "pregame_actionable_report.md"

dedup.to_csv(dedup_path, index=False)
top1.to_csv(top1_path, index=False)

lines = []
lines.append(f"# Actionable Pregame Card - {BOARD_DATE}")
lines.append("")
lines.append(f"- Raw candidate rows: {len(df)}")
lines.append(f"- Gate-passing rows: {len(gated)}")
lines.append(f"- Deduped actionable bets: {len(dedup)}")
lines.append(f"- Top-1-per-pitcher actionable bets: {len(top1)}")
lines.append("")
lines.append("## Top actionable bets")
lines.append("")
if len(top1):
    lines.append(top1.head(25).to_markdown(index=False))
else:
    lines.append("No actionable bets.")

report_path.write_text("\n".join(lines))

print("WROTE:")
print(dedup_path)
print(top1_path)
print(report_path)
print("")
print("COUNTS:")
print("raw candidate rows:", len(df))
print("gate-passing rows:", len(gated))
print("dedup actionable rows:", len(dedup))
print("top1 per pitcher rows:", len(top1))
print("")
print("TOP 20 ACTIONABLE:")
print(top1.head(20).to_string(index=False))

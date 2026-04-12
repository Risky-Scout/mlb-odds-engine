from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def clip_prob(x):
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)


def find_col(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(f"Could not find any of required columns: {candidates}")
    return None


def normalize_side(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"o", "over", "ov"}:
        return "over"
    if s in {"u", "under", "un"}:
        return "under"
    return s


def prepare_rows(df: pd.DataFrame) -> pd.DataFrame:
    best_ev_col = find_col(df, ["best_ev", "ev_best", "selected_ev", "best_expected_value"])
    model_over_col = find_col(df, ["model_over_prob", "raw_model_over_prob"])
    market_over_col = find_col(df, ["market_over_prob"])
    actual_over_col = find_col(df, ["actual_over"])

    vendor_col = find_col(df, ["vendor", "book", "bookmaker"], required=False)
    line_col = find_col(df, ["line_value"], required=False)

    side_col = find_col(
        df,
        ["best_side", "selected_side", "best_bet_side", "bet_side", "edge_side"],
        required=False,
    )

    out = df.copy()

    if side_col is None:
        over_ev_col = find_col(df, ["over_ev", "ev_over", "model_over_ev"], required=False)
        under_ev_col = find_col(df, ["under_ev", "ev_under", "model_under_ev"], required=False)
        if over_ev_col is not None and under_ev_col is not None:
            out["_side"] = np.where(
                pd.to_numeric(out[over_ev_col], errors="coerce").fillna(-np.inf)
                >= pd.to_numeric(out[under_ev_col], errors="coerce").fillna(-np.inf),
                "over",
                "under",
            )
        else:
            raise SystemExit(
                "Could not determine selected side. Need one of best_side/selected_side/etc or over_ev+under_ev columns."
            )
    else:
        out["_side"] = out[side_col].map(normalize_side)

    out["_best_ev"] = pd.to_numeric(out[best_ev_col], errors="coerce")
    out["_model_over"] = pd.to_numeric(out[model_over_col], errors="coerce")
    out["_market_over"] = pd.to_numeric(out[market_over_col], errors="coerce")
    out["_actual_over"] = pd.to_numeric(out[actual_over_col], errors="coerce")

    out["_vendor"] = out[vendor_col].astype(str) if vendor_col is not None else "ALL"
    out["_line"] = pd.to_numeric(out[line_col], errors="coerce") if line_col is not None else np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["_best_ev", "_model_over", "_market_over", "_actual_over", "_side"]).copy()

    out["_model_sel_prob"] = np.where(out["_side"].eq("over"), out["_model_over"], 1.0 - out["_model_over"])
    out["_market_sel_prob"] = np.where(out["_side"].eq("over"), out["_market_over"], 1.0 - out["_market_over"])
    out["_actual_sel"] = np.where(out["_side"].eq("over"), out["_actual_over"], 1.0 - out["_actual_over"])
    out["_conf_dist"] = np.abs(out["_model_over"] - 0.5)

    out["_model_sel_prob"] = clip_prob(out["_model_sel_prob"])
    out["_market_sel_prob"] = clip_prob(out["_market_sel_prob"])
    out["_actual_sel"] = pd.to_numeric(out["_actual_sel"], errors="coerce")

    out = out.dropna(subset=["_model_sel_prob", "_market_sel_prob", "_actual_sel"]).copy()
    return out


def gate_metrics(g: pd.DataFrame) -> dict:
    y = g["_actual_sel"].astype(int).to_numpy()
    pm = clip_prob(g["_model_sel_prob"].to_numpy())
    pk = clip_prob(g["_market_sel_prob"].to_numpy())

    model_ll = float(log_loss(y, pm))
    market_ll = float(log_loss(y, pk))
    model_br = float(brier_score_loss(y, pm))
    market_br = float(brier_score_loss(y, pk))

    return {
        "n_selected": int(len(g)),
        "coverage": float(len(g) / max(len(df_all), 1)),
        "selected_hit_rate": float(np.mean(y)),
        "selected_mean_best_ev": float(g["_best_ev"].mean()),
        "selected_model_log_loss": model_ll,
        "selected_market_log_loss": market_ll,
        "delta_log_loss_market_minus_model": float(market_ll - model_ll),
        "selected_model_brier": model_br,
        "selected_market_brier": market_br,
        "delta_brier_market_minus_model": float(market_br - model_br),
        "selected_mean_model_sel_prob": float(np.mean(pm)),
        "selected_mean_market_sel_prob": float(np.mean(pk)),
        "selected_mean_conf_dist": float(g["_conf_dist"].mean()),
    }


def better(a: dict, b: dict) -> bool:
    ll_gap_tol = 1e-4
    br_gap_tol = 1e-4
    ev_tol = 1e-6

    if a["delta_log_loss_market_minus_model"] > b["delta_log_loss_market_minus_model"] + ll_gap_tol:
        return True
    if abs(a["delta_log_loss_market_minus_model"] - b["delta_log_loss_market_minus_model"]) <= ll_gap_tol:
        if a["delta_brier_market_minus_model"] > b["delta_brier_market_minus_model"] + br_gap_tol:
            return True
        if abs(a["delta_brier_market_minus_model"] - b["delta_brier_market_minus_model"]) <= br_gap_tol:
            if a["selected_mean_best_ev"] > b["selected_mean_best_ev"] + ev_tol:
                return True
            if abs(a["selected_mean_best_ev"] - b["selected_mean_best_ev"]) <= ev_tol:
                if a["selected_hit_rate"] > b["selected_hit_rate"] + ev_tol:
                    return True
                if abs(a["selected_hit_rate"] - b["selected_hit_rate"]) <= ev_tol:
                    if a["n_selected"] > b["n_selected"]:
                        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-selected", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(args.rows_path)
    global df_all
    df_all = prepare_rows(raw)

    edge_grid = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    conf_grid = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

    vendor_counts = df_all["_vendor"].value_counts()
    vendor_opts = ["ALL"] + sorted(vendor_counts[vendor_counts >= args.min_selected].index.tolist())

    line_counts = df_all["_line"].value_counts(dropna=True)
    line_opts = ["ALL"] + sorted(line_counts[line_counts >= args.min_selected].index.tolist())

    rows_out = []
    best_row = None
    best_mask = None

    for edge in edge_grid:
        for conf in conf_grid:
            for vendor in vendor_opts:
                for line in line_opts:
                    mask = (df_all["_best_ev"] >= edge) & (df_all["_conf_dist"] >= conf)

                    if vendor != "ALL":
                        mask &= df_all["_vendor"].eq(vendor)
                    if line != "ALL":
                        mask &= df_all["_line"].eq(float(line))

                    g = df_all.loc[mask].copy()
                    if len(g) < args.min_selected:
                        continue

                    m = gate_metrics(g)
                    m["min_edge"] = float(edge)
                    m["min_conf_dist"] = float(conf)
                    m["vendor_filter"] = vendor
                    m["line_filter"] = line

                    # strict requirement: keep only gates where model beats market
                    m["model_beats_market"] = bool(
                        m["delta_log_loss_market_minus_model"] > 0
                        and m["delta_brier_market_minus_model"] > 0
                    )

                    rows_out.append(m)

                    if not m["model_beats_market"]:
                        continue

                    if best_row is None or better(m, best_row):
                        best_row = m
                        best_mask = mask.copy()

    if not rows_out:
        raise SystemExit("No valid gate rows were generated.")

    sweep = pd.DataFrame(rows_out).sort_values(
        [
            "model_beats_market",
            "delta_log_loss_market_minus_model",
            "delta_brier_market_minus_model",
            "selected_mean_best_ev",
            "selected_hit_rate",
            "n_selected",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    sweep.to_csv(out_dir / "strict_ev_gate_sweep.csv", index=False)

    if best_row is None:
        decision = {
            "accepted": False,
            "reason": "No gate beat market on both selected-row log loss and Brier while meeting min-selected constraint.",
            "n_rows_total": int(len(df_all)),
            "min_selected": int(args.min_selected),
        }
        with open(out_dir / "strict_ev_gate_decision.json", "w") as f:
            json.dump(decision, f, indent=2)
        print(json.dumps(decision, indent=2))
        return

    selected = df_all.loc[best_mask].copy()
    selected.to_parquet(out_dir / "strict_ev_gate_selected_rows.parquet", index=False)

    decision = {
        "accepted": True,
        "n_rows_total": int(len(df_all)),
        "min_selected": int(args.min_selected),
        "chosen_gate": {
            "min_edge": best_row["min_edge"],
            "min_conf_dist": best_row["min_conf_dist"],
            "vendor_filter": best_row["vendor_filter"],
            "line_filter": best_row["line_filter"],
            "n_selected": best_row["n_selected"],
            "coverage": best_row["coverage"],
            "selected_hit_rate": best_row["selected_hit_rate"],
            "selected_mean_best_ev": best_row["selected_mean_best_ev"],
            "selected_model_log_loss": best_row["selected_model_log_loss"],
            "selected_market_log_loss": best_row["selected_market_log_loss"],
            "delta_log_loss_market_minus_model": best_row["delta_log_loss_market_minus_model"],
            "selected_model_brier": best_row["selected_model_brier"],
            "selected_market_brier": best_row["selected_market_brier"],
            "delta_brier_market_minus_model": best_row["delta_brier_market_minus_model"],
            "selected_mean_model_sel_prob": best_row["selected_mean_model_sel_prob"],
            "selected_mean_market_sel_prob": best_row["selected_mean_market_sel_prob"],
            "selected_mean_conf_dist": best_row["selected_mean_conf_dist"],
        },
    }

    with open(out_dir / "strict_ev_gate_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    print("\nBEST GATE")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def _first_present_value(obj, candidates):
    for c in candidates:
        if c in obj.index:
            val = obj[c]
            if pd.notna(val):
                return val
    return None


def _starter_team_id(starter):
    return _first_present_value(starter, [
        "team.id",
        "team_id",
        "teamId",
        "team",
    ])


def _game_team_id(game, side: str):
    if side == "home":
        return _first_present_value(game, [
            "home_team.id",
            "home_team_id",
            "homeTeam.id",
            "homeTeamId",
            "home_id",
            "home.id",
            "home_team",
        ])
    return _first_present_value(game, [
        "away_team.id",
        "away_team_id",
        "awayTeam.id",
        "awayTeamId",
        "away_id",
        "away.id",
        "away_team",
    ])


def _is_home_pitcher_from_rows(starter, game) -> int:
    st = _starter_team_id(starter)
    hm = _game_team_id(game, "home")
    aw = _game_team_id(game, "away")

    try:
        if st is not None and hm is not None and str(st) == str(hm):
            return 1
        if st is not None and aw is not None and str(st) == str(aw):
            return 0
    except Exception:
        pass

    return 0


def _first_present_value(obj, candidates):
    for c in candidates:
        if c in obj.index:
            val = obj[c]
            if pd.notna(val):
                return val
    return None


def _starter_team_id(starter):
    return _first_present_value(starter, [
        "team.id",
        "team_id",
        "teamId",
        "team",
    ])


def _game_team_id(game, side: str):
    if side == "home":
        return _first_present_value(game, [
            "home_team.id",
            "home_team_id",
            "homeTeam.id",
            "homeTeamId",
            "home_id",
            "home.id",
            "home_team",
        ])
    return _first_present_value(game, [
        "away_team.id",
        "away_team_id",
        "awayTeam.id",
        "awayTeamId",
        "away_id",
        "away.id",
        "away_team",
    ])


def _is_home_pitcher_from_rows(starter, game) -> int:
    st = _starter_team_id(starter)
    hm = _game_team_id(game, "home")
    aw = _game_team_id(game, "away")

    try:
        if st is not None and hm is not None and str(st) == str(hm):
            return 1
        if st is not None and aw is not None and str(st) == str(aw):
            return 0
    except Exception:
        pass

    return 0


from .api_clients import BallDontLieClient, ESPNClient, MarketQuote, OddsAPIClient
from .data_pipeline import FeatureBuilder
from .external_data import map_odds_api_quotes_to_bdl, market_quotes_from_df, normalize_odds_api_event_odds
from .market import line_event_probs_from_pmf
from .system import StrikeoutBettingSystem


@dataclass
class PriceBoardResult:
    summary: pd.DataFrame
    exact_pmf: pd.DataFrame
    line_grid: pd.DataFrame
    candidate_bets: pd.DataFrame


def _yyyymmdd(date_str: str) -> str:
    return pd.Timestamp(date_str).strftime("%Y%m%d")


def _safe_name_from_lineup(lineup_df: pd.DataFrame, player_id: int) -> str:
    row = lineup_df[lineup_df["player.id"].eq(player_id)]
    if row.empty:
        return str(player_id)
    val = row.iloc[0].get("player.full_name")
    return str(val) if pd.notna(val) else str(player_id)


def _quote_frame(quotes: List[MarketQuote]) -> pd.DataFrame:
    if not quotes:
        return pd.DataFrame()
    return pd.DataFrame([q.__dict__ for q in quotes])


class DailyPriceBoard:
    def __init__(
        self,
        system: StrikeoutBettingSystem,
        bdl: BallDontLieClient,
        espn: Optional[ESPNClient] = None,
        odds_api: Optional[OddsAPIClient] = None,
        players_df: Optional[pd.DataFrame] = None,
    ):
        self.system = system
        self.bdl = bdl
        self.espn = espn or ESPNClient()
        self.odds_api = odds_api
        self.players_df = players_df if players_df is not None else pd.DataFrame()


    def _odds_api_quote_frame(self, games: pd.DataFrame) -> pd.DataFrame:
        if self.odds_api is None or self.players_df.empty or games.empty:
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        try:
            events = self.odds_api.list_events(sport="baseball_mlb")
        except Exception:
            return pd.DataFrame()

        for event in events:
            try:
                payload = self.odds_api.get_event_odds(
                    str(event["id"]),
                    sport="baseball_mlb",
                    regions="us",
                    markets="pitcher_strikeouts,pitcher_strikeouts_alternate",
                    odds_format="american",
                )
            except Exception:
                continue

            norm = normalize_odds_api_event_odds(payload)
            if not norm.empty:
                frames.append(norm)

        if not frames:
            return pd.DataFrame()

        odds_df = pd.concat(frames, ignore_index=True)
        return map_odds_api_quotes_to_bdl(odds_df, games, self.players_df)

    def _combine_quotes(
        self,
        *,
        game_id: int,
        pitcher_id: int,
        bdl_quotes: List[MarketQuote],
        odds_api_quotes_df: pd.DataFrame,
    ) -> List[MarketQuote]:
        out = list(bdl_quotes)
        if not odds_api_quotes_df.empty:
            out.extend(market_quotes_from_df(odds_api_quotes_df, game_id=game_id, player_id=pitcher_id, prop_type="pitcher_strikeouts"))
            out.extend(market_quotes_from_df(odds_api_quotes_df, game_id=game_id, player_id=pitcher_id, prop_type="pitcher_strikeouts_alternate"))
        return out

    def _candidate_rows(
        self,
        *,
        game_id: int,
        pitcher_id: int,
        pitcher_name: str,
        mode: str,
        final_pmf: np.ndarray,
        market_quotes: List[MarketQuote],
        market_pmf: Optional[np.ndarray],
        uncertainty_base: float,
    ) -> pd.DataFrame:
        qf = _quote_frame(market_quotes)
        if qf.empty:
            return pd.DataFrame()

        rows: List[Dict] = []
        consensus = self.system.market_model.consensus_thresholds(market_quotes)
        consensus_map = {float(r.line_value): r for r in consensus.itertuples(index=False)}

        for row in qf.itertuples(index=False):
            model_over, model_under, _ = line_event_probs_from_pmf(final_pmf, float(row.line_value))
            market_over = np.nan
            market_under = np.nan
            vendor_count = 1
            vendor_std = 0.0
            if float(row.line_value) in consensus_map:
                c = consensus_map[float(row.line_value)]
                market_over = float(c.fair_over_prob)
                market_under = float(c.fair_under_prob)
                vendor_count = int(c.vendor_count)
                vendor_std = float(c.vendor_std)

            for side, prob, book_odds, market_prob in [
                ("over", model_over, row.over_odds, market_over),
                ("under", model_under, row.under_odds, market_under),
            ]:
                if book_odds is None or pd.isna(book_odds):
                    continue

                uncertainty = float(uncertainty_base + vendor_std + (abs(prob - market_prob) if pd.notna(market_prob) else 0.02))
                edge = float(prob - (market_prob if pd.notna(market_prob) else 0.0))
                from .portfolio import bet_ev
                ev = bet_ev(prob, int(book_odds))

                rows.append(
                    {
                        "mode": mode,
                        "game_id": int(game_id),
                        "pitcher_id": int(pitcher_id),
                        "pitcher_name": pitcher_name,
                        "vendor": row.vendor,
                        "line_value": float(row.line_value),
                        "bet_side": side,
                        "bet_prob": float(prob),
                        "book_odds": int(book_odds),
                        "market_prob": float(market_prob) if pd.notna(market_prob) else np.nan,
                        "edge": edge,
                        "ev": ev,
                        "uncertainty": uncertainty,
                        "vendor_count": vendor_count,
                        "abs_model_market_gap": abs(prob - market_prob) if pd.notna(market_prob) else 0.0,
                        "is_live": int(mode == "live"),
                    }
                )

        cand = pd.DataFrame(rows)
        if cand.empty:
            return cand

        cand = self.system.bet_selector.score(cand)
        cand = self.system.risk_sizer.size_portfolio(cand)
        return cand.sort_values(["selected", "ev", "edge"], ascending=[False, False, False]).reset_index(drop=True)

    def _one_pitcher_board(
        self,
        *,
        game_id: int,
        pitcher_id: int,
        pitcher_name: str,
        mode: str,
        state: Dict,
        lineup_df: pd.DataFrame,
        prop_quotes: List[MarketQuote],
    ) -> tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pmf_model = self.system.true_model.predict_pmf(state=state, future_lineup=lineup_df)
        market_view = self.system.market_model.build_view(prop_quotes, pmf_model)
        final_pmf = pmf_model.copy()

        # If a market-aware blender exists, use it line-by-line in candidate space. The PMF itself stays
        # as the calibrated true-outcome PMF; the market PMF is shown separately via market probabilities.
        exact = self.system.true_model.exact_count_odds(final_pmf).assign(
            mode=mode,
            game_id=int(game_id),
            pitcher_id=int(pitcher_id),
            pitcher_name=pitcher_name,
        )

        line_values = sorted(set([float(q.line_value) for q in prop_quotes] + [x + 0.5 for x in range(0, 13)]))
        grid = self.system.true_model.line_grid(final_pmf, line_values).assign(
            mode=mode,
            game_id=int(game_id),
            pitcher_id=int(pitcher_id),
            pitcher_name=pitcher_name,
        )

        uncertainty_base = 0.01 + float(market_view.efficiency.get("mean_vendor_std", 0.0))
        candidates = self._candidate_rows(
            game_id=game_id,
            pitcher_id=pitcher_id,
            pitcher_name=pitcher_name,
            mode=mode,
            final_pmf=final_pmf,
            market_quotes=prop_quotes,
            market_pmf=market_view.market_pmf,
            uncertainty_base=uncertainty_base,
        )

        summary = {
            "mode": mode,
            "game_id": int(game_id),
            "pitcher_id": int(pitcher_id),
            "pitcher_name": pitcher_name,
            "k_mean": float((np.arange(len(final_pmf)) * final_pmf).sum()),
            "k_median": int(np.searchsorted(np.cumsum(final_pmf), 0.5)),
            "k_p80": int(np.searchsorted(np.cumsum(final_pmf), 0.8)),
            "k_p90": int(np.searchsorted(np.cumsum(final_pmf), 0.9)),
            "top_quote_count": len(prop_quotes),
            "market_mean": float((np.arange(len(market_view.market_pmf)) * market_view.market_pmf).sum()) if market_view.market_pmf is not None else np.nan,
            "market_mean_shift": float(market_view.efficiency.get("mean_shift", np.nan)) if market_view.efficiency else np.nan,
            "vendor_count": float(market_view.efficiency.get("vendor_count", np.nan)) if market_view.efficiency else np.nan,
        }
        return summary, exact, grid, candidates

    def build_pregame_board(self, *, date: str, history_pa_df: pd.DataFrame) -> PriceBoardResult:
        games = pd.DataFrame(self.bdl.list_games(dates=[date], season_type="regular"))
        if games.empty:
            return PriceBoardResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        lineup_df = pd.json_normalize(self.bdl.get_lineups(games["id"].astype(int).tolist()))
        odds_api_quotes_df = self._odds_api_quote_frame(games)
        if lineup_df.empty:
            return PriceBoardResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        summaries: List[Dict] = []
        exacts: List[pd.DataFrame] = []
        grids: List[pd.DataFrame] = []
        cands: List[pd.DataFrame] = []

        for _, game in games.iterrows():
            game_id = int(game["id"])
            g_lineup = lineup_df[lineup_df["game_id"].eq(game_id)].copy()
            if g_lineup.empty:
                continue

            probable = g_lineup[g_lineup["is_probable_pitcher"].eq(True)].copy()
            batters = g_lineup[g_lineup["batting_order"].notna()].copy()

            for _, starter in probable.iterrows():
                pitcher_id = int(starter["player.id"])
                pitcher_name = str(starter["player.full_name"])
                opp = batters[batters["team.id"] != _starter_team_id(starter)].copy()
                if opp.empty:
                    continue

                future_lineup = FeatureBuilder.build_future_lineup_frame(
                    opp,
                    history_pa_df,
                    pitcher_id=pitcher_id,
                    as_of_date=date,
                )
                if future_lineup.empty:
                    continue

                latest = (
                    history_pa_df[history_pa_df["pitcher_id"].eq(pitcher_id)]
                    .sort_values(["game_date", "game_id", "pa_number"])
                    .tail(1)
                )
                if latest.empty:
                    continue

                base = latest.iloc[0]
                state = {
                    "game_id": game_id,
                    "pitcher_id": pitcher_id,
                    "pitcher_hand": base.get("pitcher_hand"),
                    "is_home_pitcher": _is_home_pitcher_from_rows(starter, game),
                    "inning": 1,
                    "outs_start": 0,
                    "bf_so_far": 0,
                    "k_so_far": 0,
                    "walk_so_far": 0,
                    "hit_so_far": 0,
                    "outs_so_far": 0,
                    "pitch_count_so_far": 0,
                    "avg_pitch_count_so_far": base.get("pitcher_pa_pitch_ct_60"),
                    "times_through_order": 0,
                    "runners_on": 0,
                    "is_scoring_position": 0,
                    "pitcher_workload_flag": 0,
                    "days_rest": base.get("days_rest"),
                    "pitcher_k_rate_15": base.get("pitcher_k_rate_15"),
                    "pitcher_k_rate_30": base.get("pitcher_k_rate_30"),
                    "pitcher_k_rate_60": base.get("pitcher_k_rate_60"),
                    "pitcher_walk_rate_60": base.get("pitcher_walk_rate_60"),
                    "pitcher_hit_rate_60": base.get("pitcher_hit_rate_60"),
                    "pitcher_out_rate_60": base.get("pitcher_out_rate_60"),
                    "pitcher_pa_pitch_ct_60": base.get("pitcher_pa_pitch_ct_60"),
                    "pitcher_velo_60": base.get("pitcher_velo_60"),
                    "pitcher_spin_60": base.get("pitcher_spin_60"),
                    "pitcher_extension_60": base.get("pitcher_extension_60"),
                    "pitcher_xba_contact_60": base.get("pitcher_xba_contact_60"),
                    "pitcher_barrel_rate_60": base.get("pitcher_barrel_rate_60"),
                    "lineup_mean_batter_k_rate": float(future_lineup["batter_k_rate_60"].mean()),
                    "lineup_mean_batter_walk_rate": float(future_lineup["batter_walk_rate_60"].mean()),
                    "lineup_mean_batter_hit_rate": float(future_lineup["batter_hit_rate_60"].mean()),
                    "lineup_mean_batter_barrel_rate": float(future_lineup["batter_barrel_rate_60"].mean()),
                    "lineup_mean_batter_xba": float(future_lineup["batter_xba_contact_60"].mean()),
                    "lineup_same_side_share": float((future_lineup["batter_side"] == base.get("pitcher_hand")).mean()),
                }

                prop_quotes = self._combine_quotes(
                    game_id=game_id,
                    pitcher_id=pitcher_id,
                    bdl_quotes=self.bdl.get_player_props(game_id, player_id=pitcher_id, prop_type="pitcher_strikeouts"),
                    odds_api_quotes_df=odds_api_quotes_df,
                )
                summary, exact, grid, cand = self._one_pitcher_board(
                    game_id=game_id,
                    pitcher_id=pitcher_id,
                    pitcher_name=pitcher_name,
                    mode="pregame",
                    state=state,
                    lineup_df=future_lineup,
                    prop_quotes=prop_quotes,
                )
                summaries.append(summary)
                exacts.append(exact)
                grids.append(grid)
                cands.append(cand)

        return PriceBoardResult(
            summary=pd.DataFrame(summaries),
            exact_pmf=pd.concat(exacts, ignore_index=True) if exacts else pd.DataFrame(),
            line_grid=pd.concat(grids, ignore_index=True) if grids else pd.DataFrame(),
            candidate_bets=pd.concat(cands, ignore_index=True) if cands else pd.DataFrame(),
        )

    def build_live_board(self, *, date: str, history_pa_df: pd.DataFrame) -> PriceBoardResult:
        games = pd.DataFrame(self.bdl.list_games(dates=[date], season_type="regular"))
        if games.empty:
            return PriceBoardResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        lineup_df = pd.json_normalize(self.bdl.get_lineups(games["id"].astype(int).tolist()))
        odds_api_quotes_df = self._odds_api_quote_frame(games)
        summaries: List[Dict] = []
        exacts: List[pd.DataFrame] = []
        grids: List[pd.DataFrame] = []
        cands: List[pd.DataFrame] = []

        for _, game in games.iterrows():
            game_id = int(game["id"])
            if str(game.get("status", "")).upper() in {"STATUS_FINAL", "STATUS_POSTPONED", "STATUS_CANCELED"}:
                continue

            pas = self.bdl.get_plate_appearances(game_id)
            if not pas:
                continue

            live_pa = FeatureBuilder.flatten_plate_appearances(pas, game_id)
            live_pa["is_home_pitcher"] = (live_pa["half_inning"].astype(str).str.lower().eq("top")).astype(int)
            g_lineup = lineup_df[lineup_df["game_id"].eq(game_id)].copy()
            batters = g_lineup[g_lineup["batting_order"].notna()].copy()

            active_pitchers = live_pa.groupby("is_home_pitcher", as_index=False).tail(1)[["pitcher_id", "is_home_pitcher"]].drop_duplicates()
            for _, row in active_pitchers.iterrows():
                pitcher_id = int(row["pitcher_id"])
                is_home_pitcher = int(row["is_home_pitcher"])
                opp_team_id = _safe_int(_game_team_id(game, "away") if is_home_pitcher else _game_team_id(game, "home"))
                opp = batters[batters["team.id"].eq(opp_team_id)].copy()
                if opp.empty:
                    continue

                future_lineup = FeatureBuilder.build_future_lineup_frame(
                    opp,
                    history_pa_df,
                    pitcher_id=pitcher_id,
                    as_of_date=date,
                )
                if future_lineup.empty:
                    continue

                state = FeatureBuilder.build_live_state_from_pas(
                    live_pa,
                    pitcher_id=pitcher_id,
                    history_pa_df=history_pa_df,
                    current_game_id=game_id,
                    is_home_pitcher=is_home_pitcher,
                )
                if state is None:
                    continue

                # refresh lineup aggregates with the current opponent
                state["lineup_mean_batter_k_rate"] = float(future_lineup["batter_k_rate_60"].mean())
                state["lineup_mean_batter_walk_rate"] = float(future_lineup["batter_walk_rate_60"].mean())
                state["lineup_mean_batter_hit_rate"] = float(future_lineup["batter_hit_rate_60"].mean())
                state["lineup_mean_batter_barrel_rate"] = float(future_lineup["batter_barrel_rate_60"].mean())
                state["lineup_mean_batter_xba"] = float(future_lineup["batter_xba_contact_60"].mean())
                state["lineup_same_side_share"] = float((future_lineup["batter_side"] == state.get("pitcher_hand")).mean())
                pitcher_name = _safe_name_from_lineup(g_lineup, pitcher_id)

                prop_quotes = self._combine_quotes(
                    game_id=game_id,
                    pitcher_id=pitcher_id,
                    bdl_quotes=self.bdl.get_player_props(game_id, player_id=pitcher_id, prop_type="pitcher_strikeouts"),
                    odds_api_quotes_df=odds_api_quotes_df,
                )
                summary, exact, grid, cand = self._one_pitcher_board(
                    game_id=game_id,
                    pitcher_id=pitcher_id,
                    pitcher_name=pitcher_name,
                    mode="live",
                    state=state,
                    lineup_df=future_lineup,
                    prop_quotes=prop_quotes,
                )
                summaries.append(summary)
                exacts.append(exact)
                grids.append(grid)
                cands.append(cand)

        return PriceBoardResult(
            summary=pd.DataFrame(summaries),
            exact_pmf=pd.concat(exacts, ignore_index=True) if exacts else pd.DataFrame(),
            line_grid=pd.concat(grids, ignore_index=True) if grids else pd.DataFrame(),
            candidate_bets=pd.concat(cands, ignore_index=True) if cands else pd.DataFrame(),
        )


def write_daily_board(*, date: str, pregame: PriceBoardResult, live: PriceBoardResult, out_dir: str | Path) -> None:
    out_dir = Path(out_dir) / date
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = {"pregame": pregame, "live": live}
    for name, result in sections.items():
        result.summary.to_csv(out_dir / f"{name}_summary.csv", index=False)
        result.exact_pmf.to_csv(out_dir / f"{name}_exact_pmf.csv", index=False)
        result.line_grid.to_csv(out_dir / f"{name}_line_grid.csv", index=False)
        result.candidate_bets.to_csv(out_dir / f"{name}_candidate_bets.csv", index=False)

    lines = [f"# Pitcher Strikeout Board - {date}", "", "## Pre-game", ""]
    lines.append(pregame.summary.head(50).to_markdown(index=False) if not pregame.summary.empty else "_No pre-game rows._")
    lines.extend(["", "## Live Inplay", ""])
    lines.append(live.summary.head(50).to_markdown(index=False) if not live.summary.empty else "_No live rows._")
    (out_dir / "board_report.md").write_text("\n".join(lines))

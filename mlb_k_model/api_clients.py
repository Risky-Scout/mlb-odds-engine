
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests


@dataclass
class MarketQuote:
    game_id: int
    player_id: int
    vendor: str
    prop_type: str
    line_value: float
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    milestone_odds: Optional[int] = None
    updated_at: Optional[str] = None


class BallDontLieClient:
    """
    Thin wrapper around the BALLDONTLIE MLB API.

    Relevant MLB GOAT endpoints:
    - games
    - players
    - injuries
    - stats
    - season_stats
    - team season stats
    - player splits
    - plays
    - plate appearances
    - odds
    - player props
    - lineups
    """
    base_url = "https://api.balldontlie.io/mlb/v1"

    def __init__(self, api_key: str, timeout: int = 20):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get_all_pages(self, path: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = dict(params or {})
        out: List[Dict[str, Any]] = []
        cursor = None

        while True:
            if cursor is not None:
                params["cursor"] = cursor
            payload = self._get(path, params=params)
            out.extend(payload.get("data", []))
            meta = payload.get("meta") or {}
            cursor = meta.get("next_cursor")
            if cursor is None:
                break

        return out

    def list_games(
        self,
        *,
        dates: Optional[Iterable[str]] = None,
        seasons: Optional[Iterable[int]] = None,
        game_ids: Optional[Iterable[int]] = None,
        team_ids: Optional[Iterable[int]] = None,
        season_type: Optional[str] = None,
        postseason: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if dates:
            params["dates[]"] = list(dates)
        if seasons:
            params["seasons[]"] = list(seasons)
        if team_ids:
            params["team_ids[]"] = list(team_ids)
        if season_type:
            params["season_type"] = season_type
        if postseason is not None:
            params["postseason"] = str(postseason).lower()

        if game_ids:
            return [self.get_game(int(game_id)) for game_id in game_ids]

        return self._get_all_pages("games", params=params)

    def get_game(self, game_id: int) -> Dict[str, Any]:
        return self._get(f"games/{game_id}").get("data", {})

    def list_players(
        self,
        *,
        cursor: Optional[int] = None,
        per_page: int = 100,
        search: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        team_ids: Optional[Iterable[int]] = None,
        player_ids: Optional[Iterable[int]] = None,
        active_only: bool = False,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"per_page": min(int(per_page), 100)}
        if cursor is not None:
            params["cursor"] = int(cursor)
        if search:
            params["search"] = search
        if first_name:
            params["first_name"] = first_name
        if last_name:
            params["last_name"] = last_name
        if team_ids:
            params["team_ids[]"] = list(team_ids)
        if player_ids:
            params["player_ids[]"] = list(player_ids)
        path = "players/active" if active_only else "players"
        return self._get_all_pages(path, params=params)

    def get_injuries(
        self,
        *,
        team_ids: Optional[Iterable[int]] = None,
        player_ids: Optional[Iterable[int]] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if team_ids:
            params["team_ids[]"] = list(team_ids)
        if player_ids:
            params["player_ids[]"] = list(player_ids)
        return self._get_all_pages("player_injuries", params=params)

    def get_stats(
        self,
        *,
        game_ids: Optional[Iterable[int]] = None,
        player_ids: Optional[Iterable[int]] = None,
        seasons: Optional[Iterable[int]] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if game_ids:
            params["game_ids[]"] = list(game_ids)
        if player_ids:
            params["player_ids[]"] = list(player_ids)
        if seasons:
            params["seasons[]"] = list(seasons)
        return self._get_all_pages("stats", params=params)

    def get_season_stats(
        self,
        *,
        season: int,
        player_ids: Optional[Iterable[int]] = None,
        team_id: Optional[int] = None,
        season_type: str = "regular",
        postseason: Optional[bool] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"season": int(season), "season_type": season_type}
        if player_ids:
            params["player_ids[]"] = list(player_ids)
        if team_id is not None:
            params["team_id"] = int(team_id)
        if postseason is not None:
            params["postseason"] = str(postseason).lower()
        if sort_by:
            params["sort_by"] = sort_by
        if sort_order:
            params["sort_order"] = sort_order
        return self._get_all_pages("season_stats", params=params)

    def get_team_season_stats(
        self,
        *,
        season: int,
        team_id: Optional[int] = None,
        season_type: str = "regular",
        postseason: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"season": int(season), "season_type": season_type}
        if team_id is not None:
            params["team_id"] = int(team_id)
        if postseason is not None:
            params["postseason"] = str(postseason).lower()
        return self._get_all_pages("teams/season_stats", params=params)

    def get_player_splits(self, *, player_id: int, season: int) -> Dict[str, Any]:
        return self._get("players/splits", params={"player_id": int(player_id), "season": int(season)}).get("data", {})

    def get_lineups(self, game_ids: Iterable[int]) -> List[Dict[str, Any]]:
        params = {"game_ids[]": list(game_ids), "per_page": 100}
        return self._get_all_pages("lineups", params=params)

    def get_plays(self, game_id: int) -> List[Dict[str, Any]]:
        return self._get_all_pages("plays", params={"game_id": int(game_id), "per_page": 100})

    def get_plate_appearances(self, game_id: int) -> List[Dict[str, Any]]:
        return self._get("plate_appearances", params={"game_id": int(game_id)}).get("data", [])

    def get_game_odds(
        self,
        *,
        dates: Optional[Iterable[str]] = None,
        game_ids: Optional[Iterable[int]] = None,
        vendors: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if dates:
            params["dates[]"] = list(dates)
        if game_ids:
            params["game_ids[]"] = list(game_ids)
        if vendors:
            params["vendors[]"] = list(vendors)
        return self._get_all_pages("odds", params=params)

    def get_player_props(
        self,
        game_id: int,
        *,
        player_id: Optional[int] = None,
        prop_type: str = "pitcher_strikeouts",
        vendors: Optional[Iterable[str]] = None,
    ) -> List[MarketQuote]:
        params: Dict[str, Any] = {"game_id": int(game_id), "prop_type": prop_type}
        if player_id is not None:
            params["player_id"] = int(player_id)
        if vendors:
            params["vendors[]"] = list(vendors)

        rows = self._get("odds/player_props", params=params).get("data", [])
        quotes: List[MarketQuote] = []

        for row in rows:
            market = row.get("market") or {}
            mtype = market.get("type")
            quotes.append(
                MarketQuote(
                    game_id=int(row["game_id"]),
                    player_id=int(row["player_id"]),
                    vendor=str(row.get("vendor", "")),
                    prop_type=str(row.get("prop_type", prop_type)),
                    line_value=float(row.get("line_value", 0.0)),
                    over_odds=int(market["over_odds"]) if market.get("over_odds") is not None else None,
                    under_odds=int(market["under_odds"]) if market.get("under_odds") is not None else None,
                    milestone_odds=int(market["odds"]) if mtype == "milestone" and market.get("odds") is not None else None,
                    updated_at=row.get("updated_at"),
                )
            )

        return quotes


class OddsAPIClient:
    """
    Thin wrapper around The Odds API v4.

    For MLB pitcher strikeouts, the important calls are:
    - /sports/baseball_mlb/events
    - /sports/baseball_mlb/events/{eventId}/markets
    - /sports/baseball_mlb/events/{eventId}/odds
    - /historical/sports/baseball_mlb/events
    - /historical/sports/baseball_mlb/events/{eventId}/odds
    """
    base_url = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str, timeout: int = 20):
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any] | List[Dict[str, Any]]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        params = dict(params or {})
        params["apiKey"] = self.api_key
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def list_sports(self, *, all_sports: bool = False) -> List[Dict[str, Any]]:
        return self._get("sports", params={"all": str(bool(all_sports)).lower()})  # type: ignore[return-value]

    def list_events(
        self,
        *,
        sport: str = "baseball_mlb",
        date_format: str = "iso",
        event_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"dateFormat": date_format}
        if event_ids:
            params["eventIds"] = ",".join(str(x) for x in event_ids)
        return self._get(f"sports/{sport}/events", params=params)  # type: ignore[return-value]

    def get_odds(
        self,
        *,
        sport: str = "baseball_mlb",
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        date_format: str = "iso",
        event_ids: Optional[Iterable[str]] = None,
        bookmakers: Optional[Iterable[str]] = None,
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        if event_ids:
            params["eventIds"] = ",".join(str(x) for x in event_ids)
        if bookmakers:
            params["bookmakers"] = ",".join(str(x) for x in bookmakers)
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        return self._get(f"sports/{sport}/odds", params=params)  # type: ignore[return-value]

    def get_event_markets(
        self,
        event_id: str,
        *,
        sport: str = "baseball_mlb",
        regions: str = "us",
        date_format: str = "iso",
    ) -> Dict[str, Any]:
        params = {"regions": regions, "dateFormat": date_format}
        return self._get(f"sports/{sport}/events/{event_id}/markets", params=params)  # type: ignore[return-value]

    def get_event_odds(
        self,
        event_id: str,
        *,
        sport: str = "baseball_mlb",
        regions: str = "us",
        markets: str = "pitcher_strikeouts,pitcher_strikeouts_alternate",
        odds_format: str = "american",
        date_format: str = "iso",
        bookmakers: Optional[Iterable[str]] = None,
        include_multipliers: bool = False,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
            "includeMultipliers": str(bool(include_multipliers)).lower(),
        }
        if bookmakers:
            params["bookmakers"] = ",".join(str(x) for x in bookmakers)
        return self._get(f"sports/{sport}/events/{event_id}/odds", params=params)  # type: ignore[return-value]

    def get_historical_events(
        self,
        *,
        date: str,
        sport: str = "baseball_mlb",
        date_format: str = "iso",
        event_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"date": date, "dateFormat": date_format}
        if event_ids:
            params["eventIds"] = ",".join(str(x) for x in event_ids)
        return self._get(f"historical/sports/{sport}/events", params=params)  # type: ignore[return-value]

    def get_historical_odds(
        self,
        *,
        date: str,
        sport: str = "baseball_mlb",
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        date_format: str = "iso",
        bookmakers: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "date": date,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        if bookmakers:
            params["bookmakers"] = ",".join(str(x) for x in bookmakers)
        return self._get(f"historical/sports/{sport}/odds", params=params)  # type: ignore[return-value]

    def get_historical_event_odds(
        self,
        event_id: str,
        *,
        date: str,
        sport: str = "baseball_mlb",
        regions: str = "us",
        markets: str = "pitcher_strikeouts,pitcher_strikeouts_alternate",
        odds_format: str = "american",
        date_format: str = "iso",
        bookmakers: Optional[Iterable[str]] = None,
        include_multipliers: bool = False,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "date": date,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
            "includeMultipliers": str(bool(include_multipliers)).lower(),
        }
        if bookmakers:
            params["bookmakers"] = ",".join(str(x) for x in bookmakers)
        return self._get(f"historical/sports/{sport}/events/{event_id}/odds", params=params)  # type: ignore[return-value]


class PyBaseballClient:
    """
    Lazy pybaseball wrapper.

    The package exposes MLBAM id lookup plus Statcast / Savant pulls.
    This client only imports pybaseball when one of these methods is called.
    """

    def __init__(self, cache: bool = True):
        self.cache = cache
        self._pybaseball = None

    def _lib(self):
        if self._pybaseball is None:
            try:
                import pybaseball as pyb
            except ImportError as exc:  # pragma: no cover
                raise ImportError("pybaseball is not installed. Run: pip install pybaseball") from exc
            if self.cache:
                try:
                    pyb.cache.enable()
                except Exception:
                    pass
            self._pybaseball = pyb
        return self._pybaseball

    def playerid_lookup(self, last: str, first: Optional[str] = None, fuzzy: bool = False) -> pd.DataFrame:
        pyb = self._lib()
        return pyb.playerid_lookup(last, first, fuzzy=fuzzy)

    def statcast(self, start_dt: str, end_dt: Optional[str] = None) -> pd.DataFrame:
        pyb = self._lib()
        return pyb.statcast(start_dt=start_dt, end_dt=end_dt)

    def statcast_pitcher(self, start_dt: str, end_dt: Optional[str], player_id: int) -> pd.DataFrame:
        pyb = self._lib()
        return pyb.statcast_pitcher(start_dt=start_dt, end_dt=end_dt, player_id=int(player_id))

    def statcast_batter(self, start_dt: str, end_dt: Optional[str], player_id: int) -> pd.DataFrame:
        pyb = self._lib()
        return pyb.statcast_batter(start_dt=start_dt, end_dt=end_dt, player_id=int(player_id))

    def statcast_pitcher_expected_stats(self, year: int, min_pa: Optional[int] = None) -> pd.DataFrame:
        pyb = self._lib()
        return pyb.statcast_pitcher_expected_stats(year, min_pa) if min_pa is not None else pyb.statcast_pitcher_expected_stats(year)

    def statcast_pitcher_exitvelo_barrels(self, year: int, min_bbe: Optional[int] = None) -> pd.DataFrame:
        pyb = self._lib()
        return pyb.statcast_pitcher_exitvelo_barrels(year, min_bbe) if min_bbe is not None else pyb.statcast_pitcher_exitvelo_barrels(year)

    def statcast_pitcher_pitch_arsenal(self, year: int, min_p: Optional[int] = None, arsenal_type: str = "average_speed") -> pd.DataFrame:
        pyb = self._lib()
        kwargs = {"arsenal_type": arsenal_type}
        if min_p is not None:
            kwargs["minP"] = min_p
        return pyb.statcast_pitcher_pitch_arsenal(year, **kwargs)

    def statcast_pitcher_arsenal_stats(self, year: int, min_pa: int = 25) -> pd.DataFrame:
        pyb = self._lib()
        return pyb.statcast_pitcher_arsenal_stats(year, minPA=min_pa)

    def statcast_pitcher_pitch_movement(self, year: int, min_p: Optional[int] = None, pitch_type: str = "ALL") -> pd.DataFrame:
        pyb = self._lib()
        kwargs = {"pitch_type": pitch_type}
        if min_p is not None:
            kwargs["minP"] = min_p
        return pyb.statcast_pitcher_pitch_movement(year, **kwargs)

    def statcast_pitcher_percentile_ranks(self, year: int) -> pd.DataFrame:
        pyb = self._lib()
        return pyb.statcast_pitcher_percentile_ranks(year)

    def statcast_batter_expected_stats(self, year: int, min_pa: Optional[int] = None) -> pd.DataFrame:
        pyb = self._lib()
        return pyb.statcast_batter_expected_stats(year, min_pa) if min_pa is not None else pyb.statcast_batter_expected_stats(year)

    def statcast_batter_exitvelo_barrels(self, year: int, min_bbe: Optional[int] = None) -> pd.DataFrame:
        pyb = self._lib()
        return pyb.statcast_batter_exitvelo_barrels(year, min_bbe) if min_bbe is not None else pyb.statcast_batter_exitvelo_barrels(year)


class ESPNClient:
    """
    Lightweight ESPN public API wrapper.

    Useful as a second live feed. This repo uses it as a fallback, not the main data backbone.
    """
    site_base = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
    core_base = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
    cdn_base = "https://cdn.espn.com/core/mlb"

    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, url: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resp = self.session.get(url, params=params or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def scoreboard(self, date_yyyymmdd: Optional[str] = None) -> Dict[str, Any]:
        params = {"dates": date_yyyymmdd} if date_yyyymmdd else None
        return self._get(f"{self.site_base}/scoreboard", params=params)

    def summary(self, event_id: int) -> Dict[str, Any]:
        return self._get(f"{self.site_base}/summary", params={"event": int(event_id)})

    def core_game(self, event_id: int) -> Dict[str, Any]:
        return self._get(f"{self.cdn_base}/game", params={"xhr": 1, "gameId": int(event_id)})

    def core_boxscore(self, event_id: int) -> Dict[str, Any]:
        return self._get(f"{self.cdn_base}/boxscore", params={"xhr": 1, "gameId": int(event_id)})

    def competition(self, event_id: int, competition_id: Optional[int] = None) -> Dict[str, Any]:
        if competition_id is None:
            competition_id = event_id
        url = f"{self.core_base}/events/{int(event_id)}/competitions/{int(competition_id)}"
        return self._get(url)

    def competition_odds(self, event_id: int, competition_id: Optional[int] = None) -> Dict[str, Any]:
        if competition_id is None:
            competition_id = event_id
        url = f"{self.core_base}/events/{int(event_id)}/competitions/{int(competition_id)}/odds"
        return self._get(url)

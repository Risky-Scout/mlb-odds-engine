from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class LiveInplayState:
    game_id: int
    pitcher_id: int
    snapshot_ts: str | None = None
    strikeouts_so_far: int | None = None
    batters_faced_so_far: int | None = None
    pitches_thrown_so_far: int | None = None
    innings_completed: float | None = None


def build_live_inplay_state(*args: Any, **kwargs: Any) -> pd.DataFrame:
    raise NotImplementedError("Live in-play state builder not implemented yet.")


def predict_live_inplay_pmf(*args: Any, **kwargs: Any):
    raise NotImplementedError("Live in-play PMF path not implemented yet.")


def build_live_inplay_candidate_rows(*args: Any, **kwargs: Any) -> pd.DataFrame:
    raise NotImplementedError("Live in-play candidate row builder not implemented yet.")

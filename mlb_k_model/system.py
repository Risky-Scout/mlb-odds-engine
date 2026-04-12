
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from .calibration import LineProbabilityBlender
from .market import MarketModel
from .portfolio import BetSelector, RiskSizer
from .strikeout_model import StrikeoutDistributionModel


@dataclass
class StrikeoutBettingSystem:
    true_model: StrikeoutDistributionModel
    market_model: MarketModel = field(default_factory=MarketModel)
    blender: LineProbabilityBlender = field(default_factory=LineProbabilityBlender)
    bet_selector: BetSelector = field(default_factory=BetSelector)
    risk_sizer: RiskSizer = field(default_factory=RiskSizer)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "StrikeoutBettingSystem":
        return joblib.load(path)

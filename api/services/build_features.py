"""
Feature builder for prediction requests.

Joins a prediction request (brand, channel, weekstart, spend) to the matching
row from data/train/promo_dummies.csv, producing a single-row DataFrame of
features suitable for the trained pipeline.
"""

import os
from pathlib import Path
from datetime import date
from typing import Optional

import pandas as pd

# Default to <repo>/api/data/variables/promo_dummies.csv, resolved relative
# to this file so paths work regardless of CWD. Override with PROMO_PATH env var.
_DEFAULT_PROMO_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "variables" / "promo_dummies.csv"
)


class FeatureBuilder:
    """Loads promo dummies and assembles per-request feature rows."""

    def __init__(self, promo_path: Optional[str] = None):
        self.promo_path = Path(
            promo_path or os.environ.get("PROMO_PATH") or _DEFAULT_PROMO_PATH
        )
        self._promo_df: Optional[pd.DataFrame] = None

    def load(self) -> int:
        target_path = self.promo_path
        if not target_path.exists():
            fallback = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "train"
                / "promo_dummies.csv"
            )
            if fallback.exists():
                target_path = fallback
            else:
                print(f"Warning: promo_dummies.csv not found at {target_path} or {fallback}.")
                self._promo_df = pd.DataFrame()
                return 0

        df = pd.read_csv(target_path, parse_dates=["weekstart"])
        df["weekstart"] = df["weekstart"].dt.date
        self._promo_df = df
        return len(df)

    @property
    def is_loaded(self) -> bool:
        return self._promo_df is not None and not self._promo_df.empty

    @property
    def promo_columns(self) -> list[str]:
        if self._promo_df is None:
            return []
        return [c for c in self._promo_df.columns if c not in ("weekstart", "brand")]

    def build_features(
        self,
        brand: str,
        model: str,
        weekstart: date,
        spend: float,
    ) -> pd.DataFrame:
        """Return a 1-row DataFrame with request fields + promo/discount dummies."""
        if self._promo_df is None:
            self.load()
        if self._promo_df is None:
            raise RuntimeError("Promo dummies failed to load.")

        match = self._promo_df[
            (self._promo_df["brand"] == brand)
            & (self._promo_df["weekstart"] == weekstart)
        ]

        if match.empty:
            raise ValueError(
                f"No promo_dummies row for brand='{brand}', weekstart='{weekstart}'. "
                f"Check data/train/promo_dummies.csv coverage."
            )

        row = match.iloc[[0]].drop(columns=["brand", "weekstart"]).reset_index(drop=True)
        row.insert(0, "spend", spend)
        return row


_feature_builder: Optional[FeatureBuilder] = None


def get_feature_builder() -> FeatureBuilder:
    """Return the global feature builder singleton (lazily loaded)."""
    global _feature_builder
    if _feature_builder is None:
        _feature_builder = FeatureBuilder()
        _feature_builder.load()
    return _feature_builder

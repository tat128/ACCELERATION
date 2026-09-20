"""In-session defect registry backed by a pandas DataFrame."""
from __future__ import annotations

from datetime import datetime
import pandas as pd

_SCHEMA = ['unit_id', 'image_name', 'defect_type', 'confidence', 'status', 'timestamp']


class DefectRegistry:
    """Accumulates per-image inference results within a Streamlit session."""

    def __init__(self):
        self._records: list[dict] = []

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(self, image_name: str, result: dict) -> None:
        self._records.append({
            'unit_id':     len(self._records) + 1,
            'image_name':  image_name,
            'defect_type': result.get('defect_type', 'unknown'),
            'confidence':  round(result.get('confidence', 0.0), 4),
            'status':      result.get('status', 'UNKNOWN'),
            'timestamp':   datetime.now().strftime('%H:%M:%S'),
        })

    def clear(self) -> None:
        self._records.clear()

    # ── Access ────────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=_SCHEMA)
        return pd.DataFrame(self._records)

    def __len__(self) -> int:
        return len(self._records)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        df = self.to_dataframe()
        if df.empty:
            return {k: 0 for k in ('total', 'pass', 'defect', 'uncertain', 'novel', 'defect_rate')}

        total     = len(df)
        n_pass    = int((df['status'] == 'PASS').sum())
        n_defect  = int((df['status'] == 'DEFECT').sum())
        n_uncert  = int((df['status'] == 'UNCERTAIN').sum())
        n_novel   = int((df['status'] == 'NOVEL').sum())

        return {
            'total':       total,
            'pass':        n_pass,
            'defect':      n_defect,
            'uncertain':   n_uncert,
            'novel':       n_novel,
            'defect_rate': round(n_defect / total * 100, 2),
            'by_type':     df[df['status'] == 'DEFECT']['defect_type'].value_counts().to_dict(),
        }

    # ── Drift analysis data ───────────────────────────────────────────────────

    def drift_dataframe(self, window: int = 3) -> pd.DataFrame:
        """Return dataframe augmented with rolling defect rate."""
        df = self.to_dataframe()
        if df.empty:
            return df
        df = df.copy()
        df['is_defect'] = (df['status'] == 'DEFECT').astype(int)
        df['rolling_defect_rate'] = (
            df['is_defect'].rolling(window, min_periods=1).mean() * 100
        )
        return df

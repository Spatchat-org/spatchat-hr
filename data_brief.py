# data_brief.py
from typing import Optional
import pandas as pd

from schema_detect import ID_COL, TS_COL

def build_dataset_brief(df: Optional[pd.DataFrame], max_ids: int = 20) -> str:
    """
    Produce a compact dataset summary the LLM can read.
    Keep it short to stay within token budgets.
    """
    if df is None or df.empty:
        return "No dataset is loaded."

    lines = []
    lines.append("DATASET SUMMARY")
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Columns ({len(df.columns)}): {', '.join(map(str, df.columns.tolist()))}")

    # IDs
    if ID_COL in df.columns:
        ids = df[ID_COL].astype(str)
        nunq = ids.nunique(dropna=True)
        preview = ", ".join(ids.unique().astype(str)[:max_ids])
        extra = "" if nunq <= max_ids else f" … (+{nunq - max_ids} more)"
        lines.append(f"- Unique animals: {nunq}")
        lines.append(f"- Example animal IDs: {preview}{extra}")
    else:
        lines.append("- Unique animals: (no ID column detected)")

    # Time range
    if TS_COL in df.columns and df[TS_COL].notna().any():
        ts = df[TS_COL].dropna()
        lines.append(f"- Time range (UTC): {ts.min().isoformat()} → {ts.max().isoformat()}")
    else:
        lines.append("- Time range: (no timestamp detected)")

    # Geo hints
    if "latitude" in df.columns and "longitude" in df.columns:
        lat_ok = df["latitude"].between(-90, 90).mean()
        lon_ok = df["longitude"].between(-180, 180).mean()
        lines.append(f"- Lat/Lon validity: {lat_ok:.0%} / {lon_ok:.0%} rows within bounds")

    return "\n".join(lines)

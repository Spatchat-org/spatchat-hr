# dataset_context.py
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from schema_detect import ID_COL, TS_COL


# ---- JSON-safe conversion helpers ------------------------------------------
def _to_json_safe_scalar(x: Any) -> Any:
    """Convert numpy/pandas scalars & datetimes into plain JSON-safe Python types."""
    # numpy numeric / bool
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)

    # pandas/py datetime-like
    if isinstance(x, (pd.Timestamp,)):
        # keep timezone if present; isoformat() handles both tz-aware and naive
        return x.isoformat()
    # Python datetime/date (in case a series already parsed to these)
    try:
        from datetime import datetime, date
        if isinstance(x, (datetime, date)):
            return x.isoformat()
    except Exception:
        pass

    # numpy arrays → lists
    if isinstance(x, np.ndarray):
        return [_to_json_safe_scalar(v) for v in x.tolist()]

    # pass through base types (str, int, float, bool, None)
    return x


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert dicts/lists/tuples/sets to JSON-safe structures."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(v) for v in obj]
    return _to_json_safe_scalar(obj)


# ---- small sample for the model --------------------------------------------
def _safe_head(df: pd.DataFrame, n: int = 3) -> List[Dict[str, Any]]:
    """
    Return first n rows as a list of dicts with JSON-safe scalars
    (converts timestamps, numpy scalars, etc.).
    """
    recs = df.head(n).to_dict(orient="records")
    return _to_json_safe(recs)  # ensure sample is JSON-safe


# ---- public API -------------------------------------------------------------
def build_dataset_context(df: pd.DataFrame, topk_ids: int = 10) -> Dict[str, Any]:
    """
    Return a compact, strictly factual summary of the loaded dataset.
    Keep this small to stay within token limits. All values are JSON-safe.
    """
    if df is None or df.empty:
        return {"empty": True}

    cols = list(df.columns)

    brief: Dict[str, Any] = {
        "empty": False,
        "num_rows": int(len(df)),
        "num_columns": int(len(cols)),
        "columns": cols,
        "sample_rows": _safe_head(df, 3),
    }

    # Lat/lon ranges if present
    if "latitude" in df.columns and "longitude" in df.columns:
        try:
            lat_series = pd.to_numeric(df["latitude"], errors="coerce")
            lon_series = pd.to_numeric(df["longitude"], errors="coerce")
            if lat_series.notna().any():
                brief["latitude_range"] = [
                    float(lat_series.min()), float(lat_series.max())
                ]
            if lon_series.notna().any():
                brief["longitude_range"] = [
                    float(lon_series.min()), float(lon_series.max())
                ]
        except Exception:
            pass

    # Animal IDs
    if ID_COL in df.columns:
        ids = df[ID_COL].astype(str)
        brief["num_animals"] = int(ids.nunique(dropna=True))
        vc = ids.value_counts(dropna=True).head(topk_ids)
        brief["top_animals"] = [
            {"id": str(k), "n_rows": int(v)} for k, v in vc.items()
        ]
    else:
        brief["num_animals"] = None
        brief["top_animals"] = []

    # Timestamp coverage
    if TS_COL in df.columns:
        ts = pd.to_datetime(df[TS_COL], errors="coerce", utc=True)
        if ts.notna().any():
            brief["time_min_utc"] = ts.min().isoformat()
            brief["time_max_utc"] = ts.max().isoformat()
            brief["num_rows_with_time"] = int(ts.notna().sum())
        else:
            brief["time_min_utc"] = None
            brief["time_max_utc"] = None
            brief["num_rows_with_time"] = 0
    else:
        brief["time_min_utc"] = None
        brief["time_max_utc"] = None
        brief["num_rows_with_time"] = 0

    # Final pass to guarantee JSON-safe (defensive)
    return _to_json_safe(brief)

# dataset_context.py
import pandas as pd
from typing import Dict, Any
from schema_detect import ID_COL, TS_COL

def _safe_head(df: pd.DataFrame, n: int = 3) -> list[dict[str, Any]]:
    # tiny sample for the model (as plain dicts)
    return df.head(n).to_dict(orient="records")

def build_dataset_context(df: pd.DataFrame, topk_ids: int = 10) -> Dict[str, Any]:
    """
    Return a compact, strictly factual summary of the loaded dataset.
    Keep this small to stay within token limits.
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
    if "latitude" in df and "longitude" in df:
        try:
            brief["latitude_range"]  = [float(pd.to_numeric(df["latitude"], errors="coerce").min()),
                                        float(pd.to_numeric(df["latitude"], errors="coerce").max())]
            brief["longitude_range"] = [float(pd.to_numeric(df["longitude"], errors="coerce").min()),
                                        float(pd.to_numeric(df["longitude"], errors="coerce").max())]
        except Exception:
            pass

    # Animal IDs
    if ID_COL in df:
        ids = df[ID_COL].astype(str)
        brief["num_animals"] = int(ids.nunique(dropna=True))
        # Top-k frequency list
        vc = ids.value_counts(dropna=True).head(topk_ids)
        brief["top_animals"] = [{"id": str(k), "n_rows": int(v)} for k, v in vc.items()]
    else:
        brief["num_animals"] = None
        brief["top_animals"] = []

    # Timestamp coverage
    if TS_COL in df:
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

    return brief

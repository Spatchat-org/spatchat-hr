# schema_detect.py
import re
import pandas as pd
from typing import Tuple, List, Optional

# ---- Canonical output names used across the app ----
ID_COL = "animal_id"
TS_COL = "timestamp"

# ---- Candidate name lists (case-insensitive) ----
ID_CANDIDATES = [
    "animal_id", "individual_id", "individual", "id", "track_id",
    "subject_id", "tag_id", "collar_id", "name", "animal", "bird_id"
]
TS_CANDIDATES = [
    "timestamp", "datetime", "date_time", "time", "date",
    "fix_time", "acquisition_time"
]

def _best_match_by_name(columns: List[str], candidates: List[str]) -> Optional[str]:
    low = [c.lower().strip() for c in columns]
    for c in candidates:
        if c in low:
            return columns[low.index(c)]
    # fuzzy-ish fallback: substring contains
    for c in candidates:
        for i, col in enumerate(low):
            if c in col:
                return columns[i]
    return None

def _is_timestamp_series(s: pd.Series, min_ok: float = 0.8) -> bool:
    try:
        parsed = pd.to_datetime(s, errors="coerce", utc=True)
        frac = parsed.notna().mean()
        return frac >= min_ok
    except Exception:
        return False

def _id_like_series(s: pd.Series, n_rows: int) -> bool:
    # Prefer categorical-like or string/object columns with repeats
    # Many tracking datasets reuse IDs across many rows
    try:
        nunq = s.nunique(dropna=True)
        if s.dtype == "object":
            return 1 < nunq < n_rows  # not unique for every row, not all same
        # integers/strings that repeat
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s):
            return 1 < nunq < n_rows
    except Exception:
        pass
    return False

def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    # 1) by name
    name = _best_match_by_name(cols, ID_CANDIDATES)
    if name:
        return name
    # 2) by pattern/behavior
    n = len(df)
    for c in cols:
        if _id_like_series(df[c], n):
            return c
    return None

def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    # 1) by name
    name = _best_match_by_name(cols, TS_CANDIDATES)
    if name and _is_timestamp_series(df[name]):
        return name
    # 2) scan for parseable datetime columns
    best = None
    best_score = 0.0
    for c in cols:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
            score = parsed.notna().mean()
            if score > best_score and score >= 0.8:
                best = c
                best_score = score
        except Exception:
            continue
    return best

def detect_and_standardize(df: pd.DataFrame):
    """
    Returns (df2, messages)
    - df2: with standardized columns animal_id (string) and/or timestamp (datetime, UTC) if confidently detected.
    - messages: chat guidance if we couldn't detect either/both.
    """
    msgs = []
    out = df.copy()

    id_col = detect_id_column(out)
    ts_col = detect_timestamp_column(out)

    # Standardize ID column if found
    if id_col:
        if id_col != ID_COL:
            out[ID_COL] = out[id_col]
        out[ID_COL] = out[ID_COL].astype(str)
    else:
        msgs.append(
            "I couldn't detect an individual ID column. If your data has one, say: "
            "**“ID column is tag_id”** (replace `tag_id` with your column name). "
            "Or say **“no id”** if there isn’t one."
        )

    # Standardize timestamp if found
    if ts_col:
        if ts_col != TS_COL:
            out[TS_COL] = out[ts_col]
        out[TS_COL] = pd.to_datetime(out[TS_COL], errors="coerce", utc=True)
        # if too many NaT, consider unknown
        if out[TS_COL].notna().mean() < 0.8:
            out.drop(columns=[TS_COL], errors="ignore", inplace=True)
            ts_col = None

    if not ts_col:
        msgs.append(
            "I couldn't detect a timestamp column. If your data has one, say: "
            "**“Timestamp column is datetime”** (replace `datetime` with your column name). "
            "Or say **“no timestamp”** if you don't have times."
        )

    return out, msgs

# ---------- User command parsing for chat ----------
# Supports:
#  - "id column is <name>"
#  - "animal id is <name>"
#  - "timestamp column is <name>"
#  - "time column is <name>"
#  - "no id"
#  - "no timestamp"

ID_RE = re.compile(r'(?:^|\b)(?:id|animal\s*id)\s*column\s*is\s*([A-Za-z0-9_:-]+)', re.I)
ID_RE_SHORT = re.compile(r'(?:^|\b)(?:id|animal\s*id)\s*is\s*([A-Za-z0-9_:-]+)', re.I)
TS_RE = re.compile(r'(?:^|\b)(?:timestamp|time|datetime)\s*column\s*is\s*([A-Za-z0-9_:-]+)', re.I)
TS_RE_SHORT = re.compile(r'(?:^|\b)(?:timestamp|time|datetime)\s*is\s*([A-Za-z0-9_:-]+)', re.I)
NO_ID_RE = re.compile(r'(?:^|\b)no\s+id\b', re.I)
NO_TS_RE = re.compile(r'(?:^|\b)no\s+(?:ts|timestamp|time|datetime)\b', re.I)

def parse_metadata_command(text: str):
    """
    Returns dict like {"id_col": "tag_id"} or {"timestamp_col": "datetime"}
    or {"no_id": True} / {"no_timestamp": True}, or None.
    """
    if NO_ID_RE.search(text):
        return {"no_id": True}
    if NO_TS_RE.search(text):
        return {"no_timestamp": True}

    m = ID_RE.search(text) or ID_RE_SHORT.search(text)
    if m:
        return {"id_col": m.group(1)}

    m = TS_RE.search(text) or TS_RE_SHORT.search(text)
    if m:
        return {"timestamp_col": m.group(1)}

    return None

def try_apply_user_mapping(df: pd.DataFrame, cmd: dict):
    """
    Applies user-provided mapping onto df and standardizes to animal_id / timestamp.
    Returns (df2, message)
    """
    out = df.copy()
    if cmd.get("no_id"):
        if ID_COL in out:
            out.drop(columns=[ID_COL], inplace=True, errors="ignore")
        msg = "Got it — proceeding without an individual ID column."
        return out, msg

    if cmd.get("no_timestamp"):
        if TS_COL in out:
            out.drop(columns=[TS_COL], inplace=True, errors="ignore")
        msg = "Got it — proceeding without a timestamp column."
        return out, msg

    if "id_col" in cmd:
        col = cmd["id_col"]
        if col not in out.columns:
            return out, f"I couldn't find a column named `{col}`."
        out[ID_COL] = out[col].astype(str)
        return out, f"Using `{col}` as **{ID_COL}**."

    if "timestamp_col" in cmd:
        col = cmd["timestamp_col"]
        if col not in out.columns:
            return out, f"I couldn't find a column named `{col}`."
        out[TS_COL] = pd.to_datetime(out[col], errors="coerce", utc=True)
        if out[TS_COL].notna().mean() < 0.5:
            out.drop(columns=[TS_COL], inplace=True, errors="ignore")
            return out, f"`{col}` didn’t parse as timestamps for most rows."
        return out, f"Using `{col}` as **{TS_COL}**."

    return out, "No changes applied."

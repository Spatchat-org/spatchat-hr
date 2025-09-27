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
    "fix_time", "acquisition_time", "gmt_datetime", "utc_time", "gps_time"
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
    try:
        nunq = s.nunique(dropna=True)
        if s.dtype == "object":
            return 1 < nunq < n_rows
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
    """
    Detect a timestamp-like column by (1) name and (2) value profile.
    Rules:
      - Never pick latitude/longitude columns.
      - For numeric columns, only accept if they look like real timestamps:
          * Unix seconds (>=1e8),
          * Unix milliseconds (>=1e11),
          * Excel serial day numbers (~ [20000, 60000]) -> years ~1954–2064.
      - For strings, require that >=80% rows parse to datetime.
    """
    cols = list(df.columns)
    low = [c.lower().strip() for c in cols]

    # 0) Known columns to ignore for timestamp detection
    COORD_LIKE = {"lat", "latitude", "lon", "longitude", "x", "y", "easting", "northing"}

    # 1) By name (exact or substring), but never pick coord-like names
    name = _best_match_by_name(cols, TS_CANDIDATES)
    if name and name.lower().strip() not in COORD_LIKE:
        # sanity check: must parse for at least 50% rows
        try:
            parsed = pd.to_datetime(df[name], errors="coerce", utc=True)
            if parsed.notna().mean() >= 0.5:
                return name
        except Exception:
            pass  # fall through to scanning

    # 2) Scan columns by value profile with guards
    best = None
    best_score = 0.0

    for c in cols:
        lc = c.lower().strip()
        if lc in COORD_LIKE:
            continue

        s = df[c]

        # Reject obvious non-candidates
        if s.dtype.kind in ("b",):  # booleans
            continue

        is_numeric = pd.api.types.is_numeric_dtype(s)
        name_looks_timey = any(tok in lc for tok in ["time", "date", "stamp", "datetime"])

        if is_numeric:
            # Only consider numeric if name suggests time OR values look like real timestamps
            sn = pd.to_numeric(s, errors="coerce")
            if sn.notna().mean() < 0.8:
                continue

            median_abs = float(sn.abs().median())

            looks_unix_seconds = median_abs >= 1e8           # ~1973+
            looks_unix_millis  = median_abs >= 1e11          # ms since epoch
            looks_excel_days   = 20000 <= median_abs <= 60000 # ~1954–2064

            if not (name_looks_timey or looks_unix_seconds or looks_unix_millis or looks_excel_days):
                # e.g., latitude ~ 40, altitude ~ 1700, etc. -> reject
                continue

        # Finally, require that parsing succeeds for >=80% rows
        try:
            parsed = pd.to_datetime(s, errors="coerce", utc=True)
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
            "**“animal id is tag_id”** (replace `tag_id` with your column name). "
            "Or say **“no id”** if there isn’t one."
        )

    # Standardize timestamp if found
    if ts_col:
        if ts_col != TS_COL:
            out[TS_COL] = out[ts_col]
        out[TS_COL] = pd.to_datetime(out[TS_COL], errors="coerce", utc=True)
        if out[TS_COL].notna().mean() < 0.8:
            out.drop(columns=[TS_COL], errors="ignore", inplace=True)
            ts_col = None

    if not ts_col:
        msgs.append(
            "I couldn't detect a timestamp column. If your data has one, say: "
            "**“timestamp is GMT_DateTime”** (replace `GMT_DateTime` with your column name). "
            "Or say **“no timestamp”** if you don't have times."
        )

    return out, msgs

# --------------------------------------------------------------------------------------
# New: detect_metadata(df) the app expects
# Returns (id_col_or_None, ts_col_or_None, notes_str)
# --------------------------------------------------------------------------------------
def detect_metadata(df: pd.DataFrame):
    id_col = detect_id_column(df)
    ts_col = detect_timestamp_column(df)
    parts = []
    parts.append(f"Detected individual ID column: **{id_col}**." if id_col else "Couldn't detect an individual ID column.")
    parts.append(f"Detected timestamp column: **{ts_col}**." if ts_col else "Couldn't detect a timestamp column.")
    return id_col, ts_col, " ".join(parts)

# ---------- Chat command parsing ----------
# Support both styles:
#   "GMT_DateTime is timestamp" / "timestamp is GMT_DateTime"
#   "animal id is tag_id" / "id is name"
#   "timestamp column is foo" / "id column is bar"
#   "no timestamp" / "no id"

# Loose patterns:
_ID_WORD = r"(?:id|animal\s*id|individual(?:\s*id)?)"
_TS_WORD = r"(?:timestamp|time|datetime|date\s*time)"

ID_COL_IS = re.compile(rf"^\s*({_ID_WORD})\s+column\s+is\s+([A-Za-z0-9_:-]+)\s*$", re.I)
ID_IS_COL = re.compile(rf"^\s*({_ID_WORD})\s+is\s+([A-Za-z0-9_:-]+)\s*$", re.I)
COL_IS_ID = re.compile(rf"^\s*([A-Za-z0-9_:-]+)\s+is\s+({_ID_WORD})\s*$", re.I)

TS_COL_IS = re.compile(rf"^\s*({_TS_WORD})\s+column\s+is\s+([A-Za-z0-9_:-]+)\s*$", re.I)
TS_IS_COL = re.compile(rf"^\s*({_TS_WORD})\s+is\s+([A-Za-z0-9_:-]+)\s*$", re.I)
COL_IS_TS = re.compile(rf"^\s*([A-Za-z0-9_:-]+)\s+is\s+({_TS_WORD})\s*$", re.I)

NO_ID_RE = re.compile(r"(?:^|\b)no\s+id\b", re.I)
NO_TS_RE = re.compile(r"(?:^|\b)no\s+(?:ts|timestamp|time|datetime)\b", re.I)

def parse_metadata_command(text: str):
    """
    Normalized command dictionary for app.py:
      - {'animal_id': <col>} or {'animal_id': None}
      - {'timestamp': <col>} or {'timestamp': None}
    Back-compat aliases also returned:
      - {'id_col': <col>} and/or {'timestamp_col': <col>}
      - {'no_id': True} / {'no_timestamp': True}
    """
    t = (text or "").strip()
    if not t:
        return {}

    # no-... forms
    if NO_ID_RE.search(t):
        return {"animal_id": None, "no_id": True}
    if NO_TS_RE.search(t):
        return {"timestamp": None, "no_timestamp": True}

    # "X column is Y" / "X is Y" / "Y is X" for ID
    m = ID_COL_IS.search(t) or ID_IS_COL.search(t)
    if m:
        col = m.group(2)
        return {"animal_id": col, "id_col": col}
    m = COL_IS_ID.search(t)
    if m:
        col = m.group(1)
        return {"animal_id": col, "id_col": col}

    # same for timestamp
    m = TS_COL_IS.search(t) or TS_IS_COL.search(t)
    if m:
        col = m.group(2)
        return {"timestamp": col, "timestamp_col": col}
    m = COL_IS_TS.search(t)
    if m:
        col = m.group(1)
        return {"timestamp": col, "timestamp_col": col}

    return {}

def try_apply_user_mapping(df: pd.DataFrame, cmd: dict):
    """
    Applies {'animal_id': <col or None>, 'timestamp': <col or None>} (and/or legacy keys)
    to a copy of df, standardizing to 'animal_id' and 'timestamp'.
    Returns (df2, message_str).
    """
    out = df.copy()
    msgs = []

    # normalize keys
    if "id_col" in cmd and "animal_id" not in cmd:
        cmd["animal_id"] = cmd["id_col"]
    if "timestamp_col" in cmd and "timestamp" not in cmd:
        cmd["timestamp"] = cmd["timestamp_col"]

    # handle "no id"/"no timestamp"
    if cmd.get("no_id") or (cmd.get("animal_id", ... ) is None):
        if ID_COL in out.columns:
            out.drop(columns=[ID_COL], inplace=True, errors="ignore")
        msgs.append("Okay, I'll ignore any individual ID (treat all points as one animal).")
    elif "animal_id" in cmd:
        col = cmd["animal_id"]
        if col not in out.columns:
            return out, f"I couldn't find a column named **{col}**."
        out[ID_COL] = out[col].astype(str)
        msgs.append(f"Set **{ID_COL}** from **{col}**.")

    if cmd.get("no_timestamp") or (cmd.get("timestamp", ... ) is None):
        if TS_COL in out.columns:
            out.drop(columns=[TS_COL], inplace=True, errors="ignore")
        msgs.append("Okay, I'll ignore timestamps (no tracks will be drawn).")
    elif "timestamp" in cmd:
        col = cmd["timestamp"]
        if col not in out.columns:
            return out, f"I couldn't find a column named **{col}**."
        ts = pd.to_datetime(out[col], errors="coerce", utc=True)
        out[TS_COL] = ts
        if out[TS_COL].notna().mean() < 0.5:
            out.drop(columns=[TS_COL], inplace=True, errors="ignore")
            return out, f"`{col}` didn’t parse as timestamps for most rows."
        msgs.append(f"Set **{TS_COL}** from **{col}**.")

    if not msgs:
        msgs.append("Not sure what to change. Try: **timestamp is GMT_DateTime** or **animal id is tag_id**.")

    return out, " ".join(msgs)

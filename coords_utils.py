# coords_utils.py
import re
import pandas as pd

def looks_like_latlon(df: pd.DataFrame, x_col: str, y_col: str):
    """Return 'lonlat' or 'latlon' if columns look like degrees; else None."""
    try:
        x_vals = df[x_col].astype(float)
        y_vals = df[y_col].astype(float)
        if x_vals.between(-180, 180).all() and y_vals.between(-90, 90).all():
            return "lonlat"
        if x_vals.between(-90, 90).all() and y_vals.between(-180, 180).all():
            return "latlon"
    except Exception:
        return None
    return None

def looks_invalid_latlon(df: pd.DataFrame, lat_col: str, lon_col: str) -> bool:
    """True if labeled lat/lon columns are out of bounds."""
    try:
        lat = df[lat_col].astype(float)
        lon = df[lon_col].astype(float)
        return not (lat.between(-90, 90).all() and lon.between(-180, 180).all())
    except Exception:
        return True

def parse_levels_from_text(text: str):
    """Extract % levels (clamp 100→99) with sane defaults."""
    levels = [int(val) for val in re.findall(r'\b([1-9][0-9]?|100)\b', text)]
    levels = [min(l, 99) for l in levels]
    return sorted(set(levels or [95]))

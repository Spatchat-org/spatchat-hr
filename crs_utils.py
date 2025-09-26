import re

def _parse_epsg_literal(s: str):
    s = str(s).strip()
    m = re.search(r'(?i)\bepsg\s*:\s*(\d{4,6})\b', s)
    if m: return int(m.group(1))
    m = re.match(r'^\s*(\d{4,6})\s*$', s)
    if m: return int(m.group(1))
    return None

def _parse_utm_any(s: str):
    txt = str(s).strip()
    patterns = [
        r'(?i)\butm\b[^0-9]*?(\d{1,2})\s*([A-Za-z])?',
        r'(?i)\bzone\s*(\d{1,2})\s*([A-Za-z])?',
        r'\b(\d{1,2})\s*([C-HJ-NP-Xc-hj-np-x])\b',
        r'\b(\d{1,2})\s*([NnSs])\b',
    ]
    m = None
    for p in patterns:
        m = re.search(p, txt)
        if m: break
    if not m: return None
    zone = int(m.group(1))
    band = (m.group(2) or '').upper()
    if band in ('N','S'):
        hemi = 'N' if band == 'N' else 'S'
    elif band:
        hemi = 'N' if band >= 'N' else 'S'
    else:
        hemi = 'N'
    return (32600 if hemi == 'N' else 32700) + zone

def resolve_crs(user_text: str) -> int:
    if not user_text:
        raise ValueError("Empty CRS.")
    code = _parse_epsg_literal(user_text)
    if code: return code
    code = _parse_utm_any(user_text)
    if code: return code
    raise ValueError("Invalid CRS. Try forms like 'EPSG:32610', '32610', 'UTM 10T', or 'zone 10N'.")

def parse_crs_input(crs_input):
    return resolve_crs(crs_input)

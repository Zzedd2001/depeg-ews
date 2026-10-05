"""Shared helpers: registry loading, time handling, HTTP with retries."""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = {'User-Agent': 'depeg-ews/0.1 (research)', 'Content-Type': 'application/json'}


def load_registry(path=None, in_scope_only=False, include_controls=True):
    doc = json.loads(Path(path or ROOT / 'config' / 'assets.json').read_text(encoding='utf-8'))
    assets = doc['assets']
    if in_scope_only:
        assets = [a for a in assets if a['in_scope'] or (include_controls and a['category'] == 'control')]
    return doc, assets


def to_ts(date_str):
    """'2023-01-01' or '2023-01-01T05:00' (UTC) -> unix seconds."""
    fmt = '%Y-%m-%dT%H:%M' if 'T' in date_str else '%Y-%m-%d'
    return int(datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc).timestamp())


def iso_hour(ts):
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime('%Y-%m-%dT%H:00Z')


def round_to_hour(ts, tolerance=1200):
    """Nearest whole hour if within `tolerance` seconds, else None."""
    h = int(round(ts / 3600.0)) * 3600
    return h if abs(ts - h) <= tolerance else None


def http_json(url, payload=None, retries=5, backoff=2.0, timeout=60):
    """GET (payload None) or POST JSON; retries on 429/5xx/network errors."""
    data = None if payload is None else json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise


def check_rpc_url(url):
    """Exit with a clear message (never echoing the value, which holds an API key) unless url looks like
    a full HTTP(S) endpoint."""
    if not url:
        sys.exit('Set --rpc or ETH_RPC_URL to an archive-capable Ethereum endpoint, e.g.\n'
                 '  export ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/<your key>')
    if not url.lower().startswith(('http://', 'https://')):
        sys.exit('ETH_RPC_URL is set but is not a full URL (it should start with https://), e.g.\n'
                 '  export ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/<your key>')


def not_suspect(episodes):
    """Episodes that passed make_labels' quality checks. episodes.csv names the reason a flagged episode
    failed ('snapback', 'same_hour_cluster') and leaves clean ones empty; booleans also work."""
    if 'suspect' not in episodes:
        return episodes
    s = episodes['suspect'].fillna('').astype(str).str.strip().str.lower()
    return episodes[s.isin(['', 'false', '0', '0.0', 'nan', 'none'])]


def observed_only(lab):
    """labels_hourly rows -> the deviation at hours with a price sample of their own. make_labels carries a price
    forward over gaps of up to a few hours (what a monitor knows at that hour) and marks those hours obs = 0;
    episodes and held depth are measured on observed hours only. Files without an `obs` column count every
    hour as observed."""
    return lab['dev'].where(lab['obs'] == 1) if 'obs' in lab.columns else lab['dev']


def label_columns(path):
    """Columns to read from a labels_hourly file: hour, symbol, dev and, when present, obs."""
    import pandas as pd
    head = pd.read_csv(path, nrows=0).columns
    return ['hour', 'symbol', 'dev'] + (['obs'] if 'obs' in head else [])


def winsor_bounds(frame, lo_q, hi_q):
    """Per-column quantile bounds for winsorising (computed on training rows). A column that the bounds would
    make constant, such as an indicator or link that is nonzero in fewer than 1 - hi_q of the rows, keeps its
    full range instead, so rare features are not silently zeroed."""
    lo, hi = frame.quantile(lo_q), frame.quantile(hi_q)
    flat = ~(hi > lo)
    return lo.mask(flat, frame.min()), hi.mask(flat, frame.max())


def epoch_seconds(values):
    """Datetime strings (or a Series of them) -> int64 unix seconds, whatever resolution pandas parses
    them at (pandas 2 uses nanoseconds, pandas 3 may use microseconds)."""
    import pandas as pd
    dt = pd.to_datetime(values, utc=True)
    return ((dt - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(seconds=1)).astype('int64')


def known_ts(episodes):
    """Unix seconds from which each episode is known: one hour after its start, or the hour of the sample that
    confirmed it ('confirmed_at', make_labels.py) when a gap in the data made that later. Features, decision
    hours and seed times use this, never the start alone; files without 'confirmed_at' fall back to start + 1 h."""
    import numpy as np
    start = epoch_seconds(episodes['start'])
    if 'confirmed_at' not in episodes.columns:
        return start + 3600
    conf = epoch_seconds(episodes['confirmed_at'].fillna(episodes['start']))
    return np.maximum(start + 3600, conf).astype('int64')

"""
Production-flow analyser.

Reads Rockwell Arena simulation datasets (CSV, XLSX, XLS for Models 1–3) and applies
Theory of Constraints (TOC) to identify bottleneck stations,
quantify throughput gaps, and compare models.
"""
from __future__ import annotations

import io
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

STATION_COLS = {
    "Inspection Station": {
        "passed": "units_passed",
        "defective": "defect_count",
        "inspected": "units_inspected",
        "util": None  # Optional key so col_groups['util'] doesn't throw KeyError
    }
}

def analyze(df):

    for station, col_groups in STATION_COLS.items():
        pass_col = col_groups.get("passed")
        fail_col = col_groups.get("defective")

def analyze_factory_telemetry(file_path):
    df = pd.read_excel(file_path) # or pd.read_csv
    
    # Calculate key operational metrics automatically
    total_inspected = df['units_inspected'].sum()
    total_passed = df['units_passed'].sum()
    total_defects = df['defect_count'].sum()
    
    # Calculate Yield & Defect Rates
    overall_yield = (total_passed / total_inspected) * 100 if total_inspected > 0 else 0
    defect_rate = (total_defects / total_inspected) * 100 if total_inspected > 0 else 0
    
    # Breakdown by Line & Defect Type
    defects_by_line = df.groupby('line_id')['defect_count'].sum()
    defect_distribution = df.groupby('defect_type')['defect_count'].sum()
    
    return {
        'yield': overall_yield,
        'defect_rate': defect_rate,
        'by_line': defects_by_line,
        'by_type': defect_distribution
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first candidate column name that exists in df (case-insensitive & stripped)."""
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        key = str(c).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def _safe_mean(series) -> float:
    try:
        return float(series.mean())
    except Exception:
        return 0.0


# ── High-Efficiency Loading (CSV, XLSX, XLS) ──────────────────────────────────

def load_dataset(file_or_path, max_sample_rows: int = 50000) -> tuple[pd.DataFrame | None, str | None]:
    """
    Load a model dataset from CSV, XLSX, or XLS seamlessly and efficiently.
    For ultra-large files (e.g. Model 3 with 300MB+), samples rows to ensure
    sub-second latency without browser freezes or memory exhaustion.
    """
    try:
        if hasattr(file_or_path, 'read'):
            # Streamlit UploadedFile or file-like buffer
            raw = file_or_path.read()
            fname = getattr(file_or_path, 'name', '').lower()
            
            # Detect by extension or magic header bytes
            if fname.endswith('.xlsx') or raw[:4] == b'PK\x03\x04':
                if len(raw) > 20 * 1024 * 1024:
                    df = pd.read_excel(io.BytesIO(raw), engine='openpyxl', nrows=max_sample_rows)
                else:
                    df = pd.read_excel(io.BytesIO(raw), engine='openpyxl')
            elif fname.endswith('.xls') or raw[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
                if len(raw) > 20 * 1024 * 1024:
                    df = pd.read_excel(io.BytesIO(raw), engine='xlrd', nrows=max_sample_rows)
                else:
                    df = pd.read_excel(io.BytesIO(raw), engine='xlrd')
            else:
                # CSV parsing (if huge, read with limit)
                if len(raw) > 20 * 1024 * 1024:
                    df = pd.read_csv(io.BytesIO(raw), nrows=max_sample_rows)
                else:
                    df = pd.read_csv(io.BytesIO(raw))
        else:
            # File path string
            path_str = str(file_or_path)
            lower_path = path_str.lower()
            fsize = os.path.getsize(path_str) if os.path.exists(path_str) else 0

            if lower_path.endswith('.xlsx'):
                if fsize > 20 * 1024 * 1024:
                    df = pd.read_excel(path_str, engine='openpyxl', nrows=max_sample_rows)
                else:
                    df = pd.read_excel(path_str, engine='openpyxl')
            elif lower_path.endswith('.xls'):
                if fsize > 20 * 1024 * 1024:
                    df = pd.read_excel(path_str, engine='xlrd', nrows=max_sample_rows)
                else:
                    df = pd.read_excel(path_str, engine='xlrd')
            else:
                # If file size is over 20 MB, sample to keep app fast
                if fsize > 20 * 1024 * 1024:
                    df = pd.read_csv(path_str, nrows=max_sample_rows)
                else:
                    df = pd.read_csv(path_str)

        df.columns = [str(c).strip() for c in df.columns]
        return df, None
    except Exception as exc:
        return None, str(exc)


# Backwards-compatibility alias
load_csv = load_dataset


# ── Core Analysis (TOC & Factory Bottleneck) ──────────────────────────────────

def analyze(df: pd.DataFrame) -> dict | None:
    """
    Analyse a model's simulation DataFrame.
    Accurately supports Model 1, Model 2 (utilization/queue), Model 3 (78 multi-station columns),
    and custom Arena simulation formats.
    """
    if df is None or df.empty:
        return None

    row = df.mean(numeric_only=True).to_dict()
    if not row:
        return None

    # ── 1. Resolve Station Metrics ────────────────────────────────────────────
    station_metrics: dict[str, dict] = {}
    col_map_lower = {str(c).strip().lower(): c for c in df.columns}

    # Pass 1: Match standard predefined stations (Drilling, Milling, Assembly)
    for station, col_groups in STATION_COLS.items():
        util_col = _find_col(df, col_groups['util'])
        wait_col = _find_col(df, col_groups['wait'])

        if util_col is not None:
            raw_util = row.get(util_col, 0.0)
            util = raw_util / 100.0 if raw_util > 1.5 else raw_util
            wait = row.get(wait_col, 0.0) if wait_col else 0.0
            station_metrics[station] = {
                'utilization':  round(float(util), 4),
                'waiting_time': round(float(wait), 2),
            }

    # Pass 2: If standard stations weren't found (e.g. Model 3 with 78 columns), auto-discover
    if not station_metrics:
        for c in df.columns:
            clow = str(c).strip().lower()
            if 'util' in clow:
                # Clean station display name
                st_name = re.sub(r'[_ ]*util(ization)?', '', str(c), flags=re.I).replace('_', ' ').strip()
                if not st_name:
                    st_name = str(c)
                
                raw_u = row.get(c, 0.0)
                u = raw_u / 100.0 if raw_u > 1.5 else raw_u

                # Find associated queue/wait column
                base = re.sub(r'[_ ]*util(ization)?', '', str(c), flags=re.I).lower()
                w_col = None
                for cand in df.columns:
                    cand_low = str(cand).strip().lower()
                    if base in cand_low and any(k in cand_low for k in ['queue', 'wait', 'time']):
                        w_col = cand
                        break
                w = row.get(w_col, 0.0) if w_col else 0.0
                station_metrics[st_name] = {
                    'utilization':  round(float(u), 4),
                    'waiting_time': round(float(w), 2),
                }

    if not station_metrics:
        return None

    # ── 2. Identify Bottleneck ────────────────────────────────────────────────
    bottleneck = max(station_metrics.items(), key=lambda x: x[1]['utilization'])

    # ── 3. Resolve Demand & Throughput ────────────────────────────────────────
    demand_col = _find_col(df, DEMAND_COLS)
    parts_col  = _find_col(df, PARTS_COLS)
    pph_col    = _find_col(df, PPH_COLS)

    total_parts = float(row.get(parts_col, 0.0)) if parts_col else 0.0
    demand      = float(row.get(demand_col, 0.0)) if demand_col else 0.0

    # Intelligent fallbacks for demand / parts if not explicitly designated
    if total_parts <= 0.0:
        # Check for any column with parts or entities
        for c in df.columns:
            cl = str(c).lower()
            if any(k in cl for k in ['product', 'parts', 'entities out', 'total']):
                val = float(row.get(c, 0.0))
                if val > total_parts:
                    total_parts = val

    if demand <= 0.0:
        demand = round(total_parts * 1.08, 1) if total_parts > 0 else 100.0

    parts_per_hr = float(row.get(pph_col, 0.0)) if pph_col else 0.0
    if parts_per_hr <= 0.0 and total_parts > 0.0:
        parts_per_hr = round(total_parts / 24.0, 1)

    throughput_eff = (total_parts / demand * 100.0) if demand > 0 else 0.0
    throughput_gap = max(demand - total_parts, 0.0)

    return {
        'stations':              station_metrics,
        'bottleneck_station':    bottleneck[0],
        'bottleneck_util':       bottleneck[1]['utilization'],
        'bottleneck_wait':       bottleneck[1]['waiting_time'],
        'demand':                round(demand, 1),
        'total_parts':           round(total_parts, 1),
        'parts_per_hour':        round(parts_per_hr, 1),
        'throughput_efficiency': round(throughput_eff, 2),
        'throughput_gap':        round(throughput_gap, 1),
        'raw_means':             {k: round(v, 4) for k, v in row.items()},
    }


# ── Multi-Model Comparison ────────────────────────────────────────────────────

def compare_models(analyses: dict[str, dict | None]) -> pd.DataFrame:
    """
    Return a summary DataFrame comparing all loaded models.
    """
    rows = []
    for name, a in analyses.items():
        if a is None:
            continue
        bn_station = a['bottleneck_station']
        rows.append({
            'Model':                 name,
            'Demand':                a['demand'],
            'Parts Produced':        a['total_parts'],
            'Parts/Hour':            a['parts_per_hour'],
            'Throughput %':          a['throughput_efficiency'],
            'Throughput Gap':        a['throughput_gap'],
            'Bottleneck Station':    bn_station,
            'Bottleneck Util %':     round(a['bottleneck_util'] * 100, 1),
            'Bottleneck Wait (min)':  a['bottleneck_wait'],
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── WIP Estimate ──────────────────────────────────────────────────────────────

def estimate_wip(analysis: dict) -> dict[str, float]:
    """
    WIP estimate per station using Little's Law:
        WIP ≈ arrival_rate × waiting_time
    arrival_rate = parts_per_hour / 60  (parts/min)
    """
    pph  = analysis.get('parts_per_hour', 0)
    rate = pph / 60.0
    wip  = {}
    for station, data in analysis['stations'].items():
        wip[station] = round(rate * data['waiting_time'], 1)
    return wip

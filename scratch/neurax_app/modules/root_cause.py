"""
Root-cause correlation engine.

Matches observed process conditions (from production_analyzer output)
against defect patterns (from defect_registry summary) using a
dynamic evidence framework supporting arbitrary station topologies (Model 1, 2, 3, etc.).
"""
from __future__ import annotations

import pandas as pd
from utils.constants import CORRELATION_RULES


# ── Station failure mode mapping helper ────────────────────────────────────────

def _infer_defect_modes(station_name: str) -> tuple[list[str], str, str]:
    """
    Infers likely defect manifestations, physical failure mechanisms,
    and corrective action recommendations from station name semantics.
    """
    s = station_name.lower()
    if 'drill' in s:
        return (
            ['crack', 'scratch'],
            f"{station_name} station tool vibration and thermal cutting friction",
            f"Inspect drill tooling for micro-flaking, maintain coolant pressure, and verify spindle feeds.",
        )
    elif 'mill' in s or 'cnc' in s:
        return (
            ['crack', 'scratch'],
            f"{station_name} spindle vibration and thermal cutting stress",
            f"Check spindle bearing runout, increase cutting fluid lubricity, and verify tool offsets.",
        )
    elif 'press' in s or 'blank' in s or 'stamp' in s:
        return (
            ['crack', 'hole'],
            f"{station_name} dynamic mechanical stamping impact and die fatigue",
            f"Inspect die clearance, polish punch radii, and recalibrate hydraulic tonnage setpoints.",
        )
    elif 'cell' in s:
        return (
            ['crack', 'scratch'],
            f"{station_name} pacing stress and multi-axis fixture misalignment",
            f"Rebalance robotic cell cycle times, inspect part clamps, and clean locating pins.",
        )
    elif 'paint' in s or 'coat' in s:
        return (
            ['rust', 'scratch'],
            f"{station_name} surface coverage inconsistency and environmental exposure",
            f"Recalibrate spray nozzles, check curing oven temperature profile, and monitor humidity.",
        )
    elif 'asm' in s or 'assembly' in s:
        return (
            ['scratch', 'hole'],
            f"{station_name} handling abrasion and component fitment friction",
            f"Add protective rubberized guides, smooth bin transfer edges, and verify operator torque tools.",
        )
    elif 'quality' in s or 'inspect' in s:
        return (
            ['rust', 'scratch'],
            f"{station_name} queue dwell latency and repeated handling",
            f"Streamline quality gating, minimize buffer accumulation time, and use soft-touch grips.",
        )
    else:
        return (
            ['crack', 'scratch'],
            f"{station_name} flow constraint and pacing bottleneck",
            f"Balance line cycle times, adjust transfer speeds, and schedule preventive maintenance.",
        )


# ── Main correlator ───────────────────────────────────────────────────────────

def correlate(
    analysis: dict | None,
    defect_summary: dict | None = None,
) -> list[dict]:
    """
    Return a comprehensive list of evidence cards for any simulation dataset.

    Each card:
        cause           str   probable process cause
        evidence        str   which metric triggered the rule
        action          str   recommended corrective action
        confidence      str   HIGH | MEDIUM | LOW
        related_defects list  defect types linked to this cause
        defect_observed bool  were these defects actually seen in inspection?
    """
    if not analysis:
        return []

    active_defects = set(
        (defect_summary or {}).get('by_type', {}).keys()
    )

    cards: list[dict] = []
    seen_causes = set()

    # 1. Evaluate Static Rules from Constants
    flat_row = dict(analysis.get('raw_means', {}))
    flat_row['Demand']      = analysis.get('demand', 0)
    flat_row['Total parts'] = analysis.get('total_parts', 0)
    for station, data in analysis.get('stations', {}).items():
        flat_row[f'{station} Util']         = data.get('utilization', 0)
        flat_row[f'{station} Waiting Time'] = data.get('waiting_time', 0)

    for rule in CORRELATION_RULES:
        try:
            triggered = rule['condition'](flat_row)
        except Exception:
            triggered = False

        if triggered:
            related = rule.get('defects', [])
            observed = bool(set(related) & active_defects) if related else False
            card_key = rule['cause']
            if card_key not in seen_causes:
                seen_causes.add(card_key)
                cards.append({
                    'cause':           rule['cause'],
                    'evidence':        rule['evidence'],
                    'action':          rule['action'],
                    'confidence':      rule['confidence'],
                    'related_defects': related,
                    'defect_observed': observed,
                })

    # 2. Dynamic Station & Theory of Constraints (TOC) Constraint Analysis
    stations = analysis.get('stations', {})
    bottleneck_st = analysis.get('bottleneck_station')
    bottleneck_ut = analysis.get('bottleneck_util', 0)
    bottleneck_wt = analysis.get('bottleneck_wait', 0)

    # 2a. Primary Constraint Station Analysis (for ANY model)
    if bottleneck_st and bottleneck_st in stations:
        st_data = stations[bottleneck_st]
        u_val = st_data.get('utilization', bottleneck_ut)
        w_val = st_data.get('waiting_time', bottleneck_wt)
        
        defects, default_cause, default_action = _infer_defect_modes(bottleneck_st)
        observed = bool(set(defects) & active_defects) if defects else False

        if u_val >= 0.80:
            conf = 'HIGH'
            ev_str = f"Primary Constraint: {bottleneck_st} Utilisation @ {u_val*100:.1f}% (Critical >80%)"
            cause_str = f"{default_cause} (Critical TOC Bottleneck)"
        elif u_val >= 0.60:
            conf = 'MEDIUM'
            ev_str = f"Primary Constraint: {bottleneck_st} Utilisation @ {u_val*100:.1f}% with queue delay {w_val:.1f} min"
            cause_str = f"{default_cause} (Moderate Line Constraint)"
        else:
            conf = 'LOW'
            ev_str = f"{bottleneck_st} Operating Load @ {u_val*100:.1f}%"
            cause_str = f"{default_cause} (Nominal Line Flow)"

        if cause_str not in seen_causes:
            seen_causes.add(cause_str)
            cards.append({
                'cause':           cause_str,
                'evidence':        ev_str,
                'action':          default_action,
                'confidence':      conf,
                'related_defects': defects,
                'defect_observed': observed,
            })

    # 2b. Secondary Machine Load & Queue Diagnostics across All Stations
    for st_name, st_data in stations.items():
        if st_name == bottleneck_st:
            continue
        u = st_data.get('utilization', 0)
        w = st_data.get('waiting_time', 0)

        # High Overload (>80%)
        if u >= 0.80:
            defects, d_cause, d_act = _infer_defect_modes(st_name)
            observed = bool(set(defects) & active_defects) if defects else False
            cause_str = f"{d_cause} (High Load Overload)"
            if cause_str not in seen_causes:
                seen_causes.add(cause_str)
                cards.append({
                    'cause':           cause_str,
                    'evidence':        f"{st_name} Utilisation @ {u*100:.1f}% (Overload >80%)",
                    'action':          d_act,
                    'confidence':      'HIGH',
                    'related_defects': defects,
                    'defect_observed': observed,
                })
        
        # Moderate Operational Load (>50%)
        elif u >= 0.50:
            defects, d_cause, d_act = _infer_defect_modes(st_name)
            observed = bool(set(defects) & active_defects) if defects else False
            cause_str = f"{d_cause} (Active Station Workload)"
            if cause_str not in seen_causes:
                seen_causes.add(cause_str)
                cards.append({
                    'cause':           cause_str,
                    'evidence':        f"{st_name} Utilisation @ {u*100:.1f}% (Nominal operating envelope)",
                    'action':          d_act,
                    'confidence':      'LOW',
                    'related_defects': defects,
                    'defect_observed': observed,
                })

        # Extended Queue Waiting Time (> 6 min)
        if w >= 6.0:
            defects, d_cause, _ = _infer_defect_modes(st_name)
            observed = bool(set(defects) & active_defects) if defects else False
            cause_str = f"WIP buffer accumulation upstream of {st_name}"
            if cause_str not in seen_causes:
                seen_causes.add(cause_str)
                cards.append({
                    'cause':           cause_str,
                    'evidence':        f"{st_name} Queue Delay = {w:.1f} min (Buffer Accumulation)",
                    'action':          f"Implement kanban pacing and reduce batch transfer sizes into {st_name}.",
                    'confidence':      'MEDIUM',
                    'related_defects': ['rust', 'scratch'],
                    'defect_observed': observed,
                })

    # 3. Sort: HIGH confidence first, then MEDIUM, then LOW; observed defects first within tier
    order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    cards.sort(key=lambda c: (order.get(c['confidence'], 9), not c['defect_observed']))
    return cards


# ── Drift analysis ────────────────────────────────────────────────────────────

def drift_analysis(registry_df: pd.DataFrame) -> dict | None:
    """
    Detect batch-to-batch defect-rate trend.

    Returns dict with:
        df              enriched DataFrame with rolling_defect_rate
        trend           RISING | IMPROVING | STABLE
        slope           float  (pp per unit)
        novel_count     int
        uncertain_count int
    """
    if registry_df.empty or len(registry_df) < 2:
        return None

    df = registry_df.copy()

    # Compute rolling defect rate
    if 'rolling_defect_rate' not in df.columns:
        df['is_defect'] = (df['status'] == 'DEFECT').astype(int)
        df['rolling_defect_rate'] = (
            df['is_defect'].rolling(3, min_periods=1).mean() * 100
        )

    rates = df['rolling_defect_rate'].tolist()
    n     = len(rates)
    slope = (rates[-1] - rates[0]) / max(n - 1, 1)

    if slope > 2.0:
        trend = 'RISING'
    elif slope < -2.0:
        trend = 'IMPROVING'
    else:
        trend = 'STABLE'

    return {
        'df':             df,
        'trend':          trend,
        'slope':          round(slope, 3),
        'novel_count':    int((df['status'] == 'NOVEL').sum()),
        'uncertain_count':int((df['status'] == 'UNCERTAIN').sum()),
    }


# ── Novel-defect flag ─────────────────────────────────────────────────────────

def flag_novel(registry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return subset of inspected units requiring human-in-the-loop review.
    Units with status UNCERTAIN or NOVEL.
    """
    if registry_df.empty:
        return pd.DataFrame()
    return registry_df[registry_df['status'].isin(['UNCERTAIN', 'NOVEL'])].copy()

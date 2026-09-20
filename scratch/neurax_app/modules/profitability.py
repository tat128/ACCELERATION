"""
Profitability & economic modelling module.

Converts operational telemetry into financial loss estimates and
supports what-if scenario simulation via parameter sliders.
"""
from __future__ import annotations

# Assumed unit revenue multiplier (material cost → selling price)
# Adjust via sidebar — this is the default fallback
_DEFAULT_REVENUE_MULTIPLIER = 4.0


def compute_financials(
    analysis: dict | None,
    defect_summary: dict | None,
    scrap_cost_per_unit: float = 15.0,
    downtime_cost_per_min: float = 2.50,
    revenue_multiplier: float = _DEFAULT_REVENUE_MULTIPLIER,
) -> dict:
    """
    Compute financial KPIs from operational telemetry.

    Parameters
    ----------
    analysis              Output of production_analyzer.analyze()
    defect_summary        Output of DefectRegistry.summary()
    scrap_cost_per_unit   Material cost lost per scrapped unit ($)
    downtime_cost_per_min Cost of bottleneck idle time per minute ($)
    revenue_multiplier    Selling price = scrap_cost × multiplier

    Returns
    -------
    dict with financial KPIs (all in USD)
    """
    if analysis is None:
        return _empty_financials()

    total_parts    = analysis.get('total_parts', 0)
    demand         = analysis.get('demand', 0)
    throughput_gap = analysis.get('throughput_gap', 0)
    bn_wait        = analysis.get('bottleneck_wait', 0)   # avg wait (min) at bottleneck

    defect_rate_pct = (defect_summary or {}).get('defect_rate', 0)   # 0-100
    defect_rate     = defect_rate_pct / 100.0

    # Units
    good_units  = total_parts * (1 - defect_rate)
    scrap_units = total_parts * defect_rate

    # Unit revenue and rework cost estimates
    unit_revenue = scrap_cost_per_unit * revenue_multiplier
    rework_cost  = scrap_cost_per_unit * 0.4  # 40% of material cost to rework

    # ── Loss components ────────────────────────────────────────────────────────
    scrap_material_loss = scrap_units * scrap_cost_per_unit
    bottleneck_idle_loss = bn_wait * downtime_cost_per_min   # per simulation run
    opportunity_loss    = throughput_gap * unit_revenue      # revenue missed from unmet demand
    rework_cost_total   = scrap_units * rework_cost * 0.3    # assume 30% reworked, rest scrapped

    total_loss = scrap_material_loss + bottleneck_idle_loss + opportunity_loss + rework_cost_total

    # ── Revenue / Margin ───────────────────────────────────────────────────────
    gross_revenue = good_units * unit_revenue
    net_margin    = gross_revenue - total_loss
    margin_pct    = (net_margin / gross_revenue * 100) if gross_revenue > 0 else 0

    return {
        'total_parts':          round(total_parts, 1),
        'good_units':           round(good_units, 1),
        'scrap_units':          round(scrap_units, 1),
        'defect_rate_pct':      round(defect_rate_pct, 2),
        'gross_revenue':        round(gross_revenue, 2),
        'scrap_loss':           round(scrap_material_loss, 2),
        'bottleneck_idle_loss': round(bottleneck_idle_loss, 2),
        'opportunity_loss':     round(opportunity_loss, 2),
        'rework_cost':          round(rework_cost_total, 2),
        'total_loss':           round(total_loss, 2),
        'net_margin':           round(net_margin, 2),
        'margin_pct':           round(margin_pct, 1),
    }


def what_if(
    analysis: dict | None,
    defect_summary: dict | None,
    scrap_cost_per_unit: float,
    downtime_cost_per_min: float,
    util_improvement_pct: float = 0.0,
    scrap_reduction_pct: float = 0.0,
    revenue_multiplier: float = _DEFAULT_REVENUE_MULTIPLIER,
) -> dict:
    """
    Simulate the financial impact of process improvements.

    Parameters
    ----------
    util_improvement_pct   % reduction in bottleneck waiting time (0-100)
    scrap_reduction_pct    % reduction in defect rate (0-100)
    """
    if analysis is None:
        return _empty_financials()

    # Clone and adjust analysis
    adj_analysis = dict(analysis)
    adj_analysis['bottleneck_wait']     = analysis['bottleneck_wait']     * (1 - util_improvement_pct / 100)
    adj_analysis['throughput_gap']      = analysis['throughput_gap']      * (1 - util_improvement_pct / 100)

    # Clone and adjust defect summary
    base_rate = (defect_summary or {}).get('defect_rate', 0)
    adj_defect = dict(defect_summary) if defect_summary else {}
    adj_defect['defect_rate'] = base_rate * (1 - scrap_reduction_pct / 100)

    return compute_financials(
        adj_analysis, adj_defect,
        scrap_cost_per_unit, downtime_cost_per_min,
        revenue_multiplier,
    )


def impact_delta(baseline: dict, whatif: dict) -> dict:
    """Compute absolute and relative improvement from baseline to what-if."""
    delta_margin = whatif['net_margin'] - baseline['net_margin']
    delta_loss   = baseline['total_loss'] - whatif['total_loss']
    return {
        'delta_margin':     round(delta_margin, 2),
        'delta_loss_saved': round(delta_loss, 2),
        'margin_improvement_pct': round(
            delta_margin / abs(baseline['net_margin']) * 100
            if baseline['net_margin'] != 0 else 0, 1
        ),
    }


def _empty_financials() -> dict:
    keys = [
        'total_parts', 'good_units', 'scrap_units', 'defect_rate_pct',
        'gross_revenue', 'scrap_loss', 'bottleneck_idle_loss',
        'opportunity_loss', 'rework_cost', 'total_loss', 'net_margin', 'margin_pct',
    ]
    return {k: 0 for k in keys}

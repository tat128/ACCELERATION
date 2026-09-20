"""
Plotly chart factory with futuristic industrial cyber styling.
"""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

__all__ = [
    'defect_donut',
    'defect_type_bar',
    'confidence_histogram',
    'defect_trend',
    'utilization_bar',
    'waiting_time_bar',
    'multi_model_radar',
    'financial_waterfall',
    'margin_comparison_bar',
    'ann_prediction_gauge',
    'ann_sweep_curve',
    'ann_validation_scatter',
    'margin_surface_3d',
    'render_production_animation',
]

# ── Modern Classy Executive Palette ──────────────────────────────────────────
_CYBER = {
    'cyan':    '#38bdf8',   # Sky / Ice blue
    'green':   '#10b981',   # Modern Emerald
    'red':     '#f43f5e',   # Classy Rose / Crimson
    'orange':  '#f59e0b',   # Warm Amber
    'blue':    '#6366f1',   # Modern Indigo
    'purple':  '#a855f7',   # Soft Purple
    'gray':    '#64748b',   # Slate Neutral
    'dark':    '#0f172a',   # Deep Slate
    'teal':    '#14b8a6',   # Muted Teal
    'yellow':  '#eab308',   # Warm Gold
}

_BG = 'rgba(0,0,0,0)'   # transparent background for dark-mode Streamlit


def _base_layout(xaxis=None, yaxis=None, **kwargs):
    default_xaxis = dict(
        gridcolor='rgba(148, 163, 184, 0.08)',
        zerolinecolor='rgba(148, 163, 184, 0.15)',
        tickfont=dict(color='#94a3b8', size=11, family='Plus Jakarta Sans, sans-serif'),
        title_font=dict(color='#cbd5e1', size=12, family='Plus Jakarta Sans, sans-serif'),
    )
    default_yaxis = dict(
        gridcolor='rgba(148, 163, 184, 0.08)',
        zerolinecolor='rgba(148, 163, 184, 0.15)',
        tickfont=dict(color='#94a3b8', size=11, family='Plus Jakarta Sans, sans-serif'),
        title_font=dict(color='#cbd5e1', size=12, family='Plus Jakarta Sans, sans-serif'),
    )
    if xaxis is not None:
        default_xaxis.update(xaxis)
    if yaxis is not None:
        default_yaxis.update(yaxis)

    return dict(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color='#e2e8f0', family='Plus Jakarta Sans, -apple-system, sans-serif'),
        margin=dict(t=45, b=30, l=40, r=20),
        xaxis=default_xaxis,
        yaxis=default_yaxis,
        **kwargs,
    )


# ── Quality Charts ────────────────────────────────────────────────────────────

def defect_donut(summary: dict) -> go.Figure:
    labels = ['Pass', 'Defect', 'Uncertain', 'Novel']
    values = [
        summary.get('pass', 0),
        summary.get('defect', 0),
        summary.get('uncertain', 0),
        summary.get('novel', 0),
    ]
    colors = [_CYBER['green'], _CYBER['red'], _CYBER['orange'], _CYBER['purple']]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color='#0b0f19', width=2.5)),
        textinfo='label+percent',
        textfont=dict(size=11, color='#ffffff'),
        hovertemplate='<b>%{label}</b>: %{value} units (%{percent})<extra></extra>',
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text='Quality Status Distribution', font=dict(size=14, color='#f8fafc')),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.22, font=dict(color='#94a3b8')),
            height=320,
        )
    )
    return fig


def defect_type_bar(by_type: dict) -> go.Figure:
    if not by_type:
        return go.Figure()
    df = pd.DataFrame(list(by_type.items()), columns=['Defect', 'Count'])
    color_map = {
        'crack':   _CYBER['red'],
        'hole':    _CYBER['purple'],
        'rust':    _CYBER['orange'],
        'scratch': _CYBER['yellow'],
    }
    colors = [color_map.get(d.lower(), _CYBER['gray']) for d in df['Defect']]
    fig = go.Figure(go.Bar(
        x=df['Defect'], y=df['Count'],
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.15)', width=1)),
        text=df['Count'], textposition='outside',
        textfont=dict(color='#ffffff', size=11),
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text='Defect Class Signature Count', font=dict(size=14, color='#f8fafc')),
            xaxis=dict(title='Defect Signature'),
            yaxis=dict(title='Detected Units'),
            height=280,
        )
    )
    return fig


def confidence_histogram(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    fig = px.histogram(
        df[df['status'] != 'PASS'], x='confidence', nbins=20,
        color='status',
        color_discrete_map={'DEFECT': _CYBER['red'], 'UNCERTAIN': _CYBER['orange'], 'NOVEL': _CYBER['purple']},
        title='Neural Confidence Distribution (Flagged Units)',
    )
    fig.update_layout(**_base_layout(height=260))
    return fig


def defect_trend(registry_df: pd.DataFrame) -> go.Figure:
    if registry_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=registry_df['unit_id'], y=registry_df['rolling_defect_rate'],
        mode='lines+markers', name='Defect Rate',
        line=dict(color=_CYBER['cyan'], width=2.5),
        marker=dict(size=6, color=_CYBER['cyan'], line=dict(color='#ffffff', width=1)),
        fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.12)',
    ))
    fig.add_hline(y=10, line_dash='dash', line_color=_CYBER['orange'],
                  annotation_text='10% Threshold', annotation_position='bottom right',
                  annotation_font=dict(color=_CYBER['orange'], size=10))
    fig.update_layout(
        **_base_layout(
            title=dict(text='Process Drift Telemetry (Rolling 3-Unit Window)', font=dict(size=14, color='#f8fafc')),
            xaxis=dict(title='Inspection Unit #'),
            yaxis=dict(title='Defect Rate (%)'),
            height=280,
        )
    )
    return fig


# ── Production Flow Charts ────────────────────────────────────────────────────

def utilization_bar(station_metrics: dict, model_name: str) -> go.Figure:
    stations = list(station_metrics.keys())
    utils = [v['utilization'] * 100 for v in station_metrics.values()]
    colors = [
        _CYBER['red'] if u > 85 else _CYBER['orange'] if u > 70 else _CYBER['green']
        for u in utils
    ]
    fig = go.Figure(go.Bar(
        x=stations, y=utils,
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.15)', width=1)),
        text=[f'{u:.1f}%' for u in utils], textposition='outside',
        textfont=dict(color='#ffffff', size=11),
        hovertemplate='<b>%{x}</b><br>Utilisation: %{y:.1f}%<extra></extra>',
    ))
    fig.add_hline(y=85, line_dash='dash', line_color=_CYBER['red'],
                  annotation_text='Bottleneck Cutoff (85%)',
                  annotation_font=dict(color=_CYBER['red'], size=10))
    fig.add_hline(y=70, line_dash='dot', line_color=_CYBER['orange'],
                  annotation_text='Warning Threshold (70%)',
                  annotation_font=dict(color=_CYBER['orange'], size=10))
    fig.update_layout(
        **_base_layout(
            title=dict(text=f'{model_name} — Station Capacity Utilisation', font=dict(size=14, color='#f8fafc')),
            xaxis=dict(title='Production Station'),
            yaxis=dict(title='Utilisation (%)', range=[0, 115], gridcolor='rgba(0, 240, 255, 0.07)'),
            height=320,
        )
    )
    return fig


def waiting_time_bar(station_metrics: dict, model_name: str) -> go.Figure:
    stations = list(station_metrics.keys())
    waits = [v['waiting_time'] for v in station_metrics.values()]
    max_wait = max(waits) if waits else 0
    colors = [_CYBER['red'] if w == max_wait else _CYBER['cyan'] for w in waits]

    fig = go.Figure(go.Bar(
        x=stations, y=waits,
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.15)', width=1)),
        text=[f'{w:.1f}m' for w in waits], textposition='outside',
        textfont=dict(color='#ffffff', size=11),
        hovertemplate='<b>%{x}</b><br>Queue Delay: %{y:.1f} min<extra></extra>',
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text=f'{model_name} — Queue & Wait Latency', font=dict(size=14, color='#f8fafc')),
            xaxis=dict(title='Production Station'),
            yaxis=dict(title='Queue Delay (minutes)'),
            height=300,
        )
    )
    return fig


def multi_model_radar(model_analyses: dict) -> go.Figure:
    """Radar chart comparing models across key KPIs."""
    categories = ['Throughput %', 'Drill Idle', 'Mill Idle', 'Assembly Idle', 'Overall Line Output']
    fig = go.Figure()
    palette_list = [_CYBER['cyan'], _CYBER['green'], _CYBER['purple'], _CYBER['orange']]

    for i, (model_name, analysis) in enumerate(model_analyses.items()):
        if analysis is None:
            continue
        stations = analysis.get('stations', {})
        values = [
            min(analysis.get('throughput_efficiency', 0), 100),
            100 - stations.get('Drilling', {}).get('utilization', 0) * 100,
            100 - stations.get('Milling', {}).get('utilization', 0) * 100,
            100 - stations.get('Assembly', {}).get('utilization', 0) * 100,
            min(analysis.get('throughput_efficiency', 0), 100),
        ]
        col = palette_list[i % len(palette_list)]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself', name=model_name,
            line=dict(color=col, width=2),
            opacity=0.45,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 110], gridcolor='rgba(0, 240, 255, 0.1)'),
            angularaxis=dict(gridcolor='rgba(0, 240, 255, 0.1)', tickfont=dict(color='#94a3b8', size=10)),
            bgcolor='rgba(13, 19, 36, 0.5)',
        ),
        title=dict(text='Multi-Scenario Tactical Radar Benchmark', font=dict(size=14, color='#f8fafc')),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color='#e2e8f0'),
        height=380,
    )
    return fig


# ── Financial Charts ──────────────────────────────────────────────────────────

def financial_waterfall(financials: dict, model_name: str) -> go.Figure:
    labels  = ['Gross Revenue', 'Scrap Loss', 'Idle Loss', 'Opp. Loss', 'Net Margin']
    measure = ['absolute', 'relative', 'relative', 'relative', 'total']
    values  = [
        financials.get('gross_revenue', 0),
        -financials.get('scrap_loss', 0),
        -financials.get('bottleneck_idle_loss', 0),
        -financials.get('opportunity_loss', 0),
        financials.get('net_margin', 0),
    ]

    fig = go.Figure(go.Waterfall(
        name='', orientation='v',
        measure=measure, x=labels, y=values,
        connector=dict(line=dict(color='rgba(56, 189, 248, 0.7)', width=2)),
        decreasing=dict(marker=dict(color='#f43f5e', line=dict(color='#fda4af', width=1.5))),
        increasing=dict(marker=dict(color='#10b981', line=dict(color='#6ee7b7', width=1.5))),
        totals=dict(marker=dict(color='#38bdf8', line=dict(color='#bae6fd', width=1.5))),
        text=[f'${abs(v):,.0f}' for v in values],
        textposition='outside',
        textfont=dict(color='#f8fafc', size=12, family='JetBrains Mono, monospace'),
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text=f'{model_name} — Economic Waterfall Telemetry (USD)', font=dict(size=14, color='#f8fafc')),
            yaxis=dict(title='Financial Impact ($)'),
            height=360,
        )
    )
    return fig


def margin_comparison_bar(arg1, arg2=None, model_name: str = "Active Model") -> go.Figure:
    """
    Renders a high-contrast side-by-side comparison between Baseline and What-If.
    Supports both:
      margin_comparison_bar(baseline_margin, whatif_margin, model_name)
      margin_comparison_bar({'Model 1': {'baseline': ..., 'whatif': ...}})
    """
    if isinstance(arg1, dict):
        if not arg1:
            return go.Figure()
        first_key = list(arg1.keys())[0]
        if isinstance(arg1[first_key], dict) and 'baseline' in arg1[first_key]:
            baseline_val = float(arg1[first_key]['baseline'].get('net_margin', 0))
            whatif_val = float(arg1[first_key]['whatif'].get('net_margin', 0))
            model_name = str(first_key)
        else:
            baseline_val = float(arg1.get('baseline', 0))
            whatif_val = float(arg1.get('whatif', 0))
    else:
        baseline_val = float(arg1)
        whatif_val = float(arg2) if arg2 is not None else float(arg1)

    scenarios = ['Baseline Scenario', 'What-If Optimized']
    margins = [baseline_val, whatif_val]
    colors = ['#6366f1', '#10b981']

    fig = go.Figure([
        go.Bar(
            x=scenarios,
            y=margins,
            marker=dict(color=colors, line=dict(color=['#a5b4fc', '#6ee7b7'], width=1.5)),
            text=[f'${v:,.0f}' for v in margins],
            textposition='outside',
            textfont=dict(color='#f8fafc', size=13, family='JetBrains Mono, monospace'),
            width=[0.45, 0.45],
        )
    ])
    fig.update_layout(
        **_base_layout(
            title=dict(text=f'Margin Delta: Baseline vs What-If ({model_name})', font=dict(size=13, color='#f8fafc')),
            yaxis=dict(title='Operating Margin ($)'),
            height=260,
        )
    )
    return fig


# ── ANN Surrogate & Digital Twin Charts ───────────────────────────────────────

def ann_prediction_gauge(val: float, title: str, threshold: float = 0.85) -> go.Figure:
    """Renders a high-contrast gauge for ANN predicted station load with visible threshold."""
    pct = val * 100.0 if val <= 1.0 else val
    thresh_pct = threshold * 100.0 if threshold <= 1.0 else threshold
    bar_color = '#f43f5e' if pct >= thresh_pct else ('#f59e0b' if pct >= thresh_pct * 0.85 else '#38bdf8')

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number=dict(suffix="%", font=dict(color='#f8fafc', size=26, family='JetBrains Mono, monospace')),
        delta=dict(reference=thresh_pct, increasing=dict(color='#f43f5e'), decreasing=dict(color='#10b981')),
        title=dict(text=f"<b>{title}</b>", font=dict(color='#f8fafc', size=13, family='Plus Jakarta Sans, sans-serif')),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1.5, tickcolor="#94a3b8", tickfont=dict(color="#cbd5e1", size=10)),
            bar=dict(color=bar_color, thickness=0.45),
            bgcolor="rgba(30, 41, 59, 0.7)",
            borderwidth=1.5,
            bordercolor="rgba(148, 163, 184, 0.3)",
            steps=[
                dict(range=[0, thresh_pct * 0.75], color="rgba(16, 185, 129, 0.25)"),
                dict(range=[thresh_pct * 0.75, thresh_pct], color="rgba(245, 158, 11, 0.25)"),
                dict(range=[thresh_pct, 100], color="rgba(244, 63, 94, 0.35)"),
            ],
            threshold=dict(
                line=dict(color='#f43f5e', width=4),
                thickness=0.85,
                value=thresh_pct
            )
        )
    ))
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color='#e2e8f0', family='Plus Jakarta Sans, sans-serif'),
        margin=dict(t=35, b=10, l=20, r=20),
        height=200
    )
    return fig


def ann_sweep_curve(sweep_x: list, sweep_y: list, output_names: list, current_input: float = None, title: str = "Neural Metamodel Response Curve") -> go.Figure:
    """Plots multi-output surrogate response curves across parameter sweep with bold threshold."""
    fig = go.Figure()
    colors = ['#38bdf8', '#10b981', '#a855f7', '#f59e0b', '#eab308']

    import numpy as np
    y_arr = np.array(sweep_y)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)

    for i, name in enumerate(output_names):
        col_color = colors[i % len(colors)]
        series_y = y_arr[:, i] * 100.0 if np.max(y_arr[:, i]) <= 1.0 else y_arr[:, i]
        fig.add_trace(go.Scatter(
            x=sweep_x,
            y=series_y,
            mode='lines',
            name=name,
            line=dict(color=col_color, width=3),
            hovertemplate=f"Input: %{{x:.1f}}<br>{name}: %{{y:.2f}}%<extra></extra>"
        ))

    # Bold High-Contrast Threshold line
    fig.add_hline(
        y=85.0, line_dash="dash", line_color="#f43f5e", line_width=2.5,
        annotation_text="85% Bottleneck Limit",
        annotation_font=dict(color="#f43f5e", size=12, family='JetBrains Mono, monospace'),
        annotation_bgcolor="rgba(15, 23, 42, 0.85)",
        annotation_bordercolor="#f43f5e",
        annotation_borderwidth=1,
        annotation_position="top right"
    )

    if current_input is not None:
        fig.add_vline(
            x=current_input, line_dash="dot", line_color="#ffffff", line_width=2,
            annotation_text=f"Current Input: {current_input:.1f}",
            annotation_font=dict(color="#38bdf8", size=11, family='JetBrains Mono, monospace'),
            annotation_bgcolor="rgba(15, 23, 42, 0.85)",
            annotation_bordercolor="#38bdf8",
            annotation_borderwidth=1,
            annotation_position="bottom right"
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=14, color='#f8fafc')),
            xaxis=dict(title='Input Parameter Value'),
            yaxis=dict(title='Predicted Output / Utilization (%)'),
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    )
    return fig


def ann_validation_scatter(y_true: list, y_pred: list, title: str = "Arena Simulation vs ANN Surrogate", r2: float = 0.0) -> go.Figure:
    """Plots actual Rockwell Arena simulation observations vs ANN surrogate predictions."""
    import numpy as np
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Observations',
        marker=dict(color=_CYBER['cyan'], size=6, opacity=0.75, line=dict(color='#ffffff', width=0.5)),
        hovertemplate="Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>"
    ))

    # Ideal 1:1 diagonal line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Ideal Parity (y=x)',
        line=dict(color=_CYBER['green'], dash='dash', width=2)
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(text=f"{title} (R² = {r2:.4f})", font=dict(size=14, color='#f8fafc')),
            xaxis=dict(title='Rockwell Arena Actual Metric'),
            yaxis=dict(title='ANN Surrogate Metamodel Prediction'),
            height=330,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    )
    return fig


# ── 3D What-If Optimization Surface ──────────────────────────────────────────

def margin_surface_3d(
    analysis: dict,
    defect_sum: dict,
    scrap_cost: float,
    downtime_cost: float,
    revenue_mult: float,
    current_util_impr: float = 15.0,
    current_scrap_redn: float = 20.0,
) -> go.Figure:
    """
    Renders an interactive 3D Surface Landscape showing Net Margin ($)
    across variations of Scrap Reduction % and Bottleneck Idle Reduction %.
    Highlights the current slider operating point in real time.
    """
    import numpy as np
    from modules.profitability import what_if

    scrap_vals = np.linspace(0, 60, 20)
    util_vals = np.linspace(0, 50, 20)
    
    Z = np.zeros((len(util_vals), len(scrap_vals)))
    for i, u in enumerate(util_vals):
        for j, s in enumerate(scrap_vals):
            res = what_if(analysis, defect_sum, scrap_cost, downtime_cost, u, s, revenue_mult)
            Z[i, j] = res.get('net_margin', 0)

    # Calculate active point
    cur_res = what_if(analysis, defect_sum, scrap_cost, downtime_cost, current_util_impr, current_scrap_redn, revenue_mult)
    cur_z = cur_res.get('net_margin', 0)

    # Executive Classy Colorscale: Deep Obsidian -> Slate -> Steel Cobalt -> Muted Emerald -> Soft Ice Blue
    colorscale = [
        [0.0, "#0b1120"],
        [0.25, "#1e293b"],
        [0.5, "#0369a1"],
        [0.75, "#059669"],
        [1.0, "#38bdf8"],
    ]

    fig = go.Figure()

    # 3D Surface
    fig.add_trace(go.Surface(
        x=scrap_vals,
        y=util_vals,
        z=Z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=dict(text="Net Margin ($)", font=dict(color="#cbd5e1", size=11, family="Plus Jakarta Sans")),
            tickfont=dict(color="#94a3b8", size=10, family="JetBrains Mono"),
            len=0.7,
            thickness=14,
            outlinewidth=0,
        ),
        hovertemplate="Scrap Redn: %{x:.1f}%<br>Idle Redn: %{y:.1f}%<br>Margin: $%{z:,.0f}<extra></extra>",
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="#ffffff", project_z=True)
        ),
        opacity=0.94,
    ))

    # Active Operating Point Marker
    fig.add_trace(go.Scatter3d(
        x=[current_scrap_redn],
        y=[current_util_impr],
        z=[cur_z],
        mode="markers+text",
        marker=dict(
            size=7,
            color="#f43f5e",
            symbol="diamond",
            line=dict(color="#ffffff", width=1.5)
        ),
        text=[f"  Active Target: ${cur_z:,.0f}"],
        textposition="top center",
        textfont=dict(color="#f8fafc", size=11, family="JetBrains Mono"),
        name="Operating Point",
        hovertemplate="<b>ACTIVE SELECTION</b><br>Scrap Redn: %{x:.1f}%<br>Idle Redn: %{y:.1f}%<br>Margin: $%{z:,.0f}<extra></extra>"
    ))

    camera = dict(
        eye=dict(x=1.55, y=-1.55, z=0.95),
        center=dict(x=0, y=0, z=-0.1)
    )

    fig.update_layout(
        title=dict(
            text="3D Profitability Optimization Frontier (What-If Response Surface)",
            font=dict(size=14, color="#f8fafc", family="Plus Jakarta Sans, sans-serif"),
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="Scrap Reduction (%)", font=dict(color="#cbd5e1", size=11)),
                tickfont=dict(color="#94a3b8", size=10, family="JetBrains Mono"),
                backgroundcolor="rgba(15, 23, 42, 0.3)",
                gridcolor="rgba(148, 163, 184, 0.12)",
                showbackground=True,
            ),
            yaxis=dict(
                title=dict(text="Bottleneck Idle Red. (%)", font=dict(color="#cbd5e1", size=11)),
                tickfont=dict(color="#94a3b8", size=10, family="JetBrains Mono"),
                backgroundcolor="rgba(15, 23, 42, 0.3)",
                gridcolor="rgba(148, 163, 184, 0.12)",
                showbackground=True,
            ),
            zaxis=dict(
                title=dict(text="Net Margin ($)", font=dict(color="#cbd5e1", size=11)),
                tickfont=dict(color="#94a3b8", size=10, family="JetBrains Mono"),
                backgroundcolor="rgba(15, 23, 42, 0.3)",
                gridcolor="rgba(148, 163, 184, 0.12)",
                showbackground=True,
            ),
            camera=camera,
        ),
        paper_bgcolor=_BG,
        font=dict(color="#e2e8f0", family="Plus Jakarta Sans, sans-serif"),
        margin=dict(t=40, b=20, l=20, r=20),
        height=460,
    )
    return fig


# ── Animated HTML5 Canvas Production Line Simulator ──────────────────────────

def render_production_animation(analysis: dict, model_name: str = "Active Model") -> str:
    """
    Generates a standalone HTML5/CSS/JavaScript canvas component rendering a real-time
    animated conveyor line with moving WIP particle dynamics, queue congestion physics,
    and pulsating constraint/bottleneck nodes.
    """
    import json

    if not analysis or 'stations' not in analysis:
        return "<div style='color:#94a3b8; padding:20px; text-align:center;'>No simulation data available for animation.</div>"

    stations_data = []
    bn_station = analysis.get('bottleneck_station', '')
    
    for s_name, s_info in analysis['stations'].items():
        u_pct = round(s_info.get('utilization', 0) * 100, 1)
        wait_m = round(s_info.get('waiting_time', 0), 2)
        is_bn = (s_name == bn_station)
        stations_data.append({
            'name': s_name,
            'util': u_pct,
            'wait': wait_m,
            'is_bottleneck': is_bn
        })

    throughput = round(analysis.get('parts_per_hour', 10.0), 1)
    stations_json = json.dumps(stations_data)

    html_code = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: transparent;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    color: #f8fafc;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 4px;
  }}
  .sim-container {{
    position: relative;
    width: 100%;
    min-width: 750px;
    background: #0f1522;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 12px 16px 10px 16px;
  }}
  .sim-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }}
  .sim-title {{
    font-size: 12px;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: 0.4px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .sim-badge {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.25);
  }}
  .sim-stats {{
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: #64748b;
  }}
  #simCanvas {{
    display: block;
    width: 100%;
    height: 190px;
    border-radius: 7px;
    background: #090e17;
    border: 1px solid rgba(255, 255, 255, 0.05);
  }}
</style>
</head>
<body>
<div class="sim-container">
  <div class="sim-header">
    <div class="sim-title">
      <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#38bdf8;"></span>
      REAL-TIME PHYSICAL CONVEYOR TELEMETRY — {model_name.upper()}
    </div>
    <div class="sim-stats">
      LINE RATE: <span style="color:#f8fafc; font-weight:600;">{throughput} parts/hr</span> |
      <span class="sim-badge">PHYSICS SIM ACTIVE</span>
    </div>
  </div>
  <canvas id="simCanvas"></canvas>
</div>

<script>
(function() {{
  const canvas = document.getElementById('simCanvas');
  const ctx = canvas.getContext('2d');
  const stations = {stations_json};

  let width, height;
  function resize() {{
    width = canvas.parentElement.clientWidth - 32;
    if (width < 720) width = 720;
    height = 190;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }}
  resize();
  window.addEventListener('resize', resize);

  // Layout calculations
  const padX = 70;
  const trackY = 95;
  const numStations = stations.length;
  const stepX = (width - 2 * padX) / (numStations + 1);

  // Station Node Objects
  const nodes = [];
  // Inflow Station (Raw Stock)
  nodes.push({{
    name: 'Raw Stock',
    util: 100,
    isBN: false,
    x: padX,
    y: trackY,
    type: 'inflow'
  }});

  // Process Stations
  stations.forEach((s, idx) => {{
    nodes.push({{
      name: s.name,
      util: s.util,
      wait: s.wait,
      isBN: s.is_bottleneck,
      x: padX + (idx + 1) * stepX,
      y: trackY,
      type: 'station'
    }});
  }});

  // Outflow Station (Finished Goods)
  nodes.push({{
    name: 'Shipped',
    util: 100,
    isBN: false,
    x: padX + (numStations + 1) * stepX,
    y: trackY,
    type: 'outflow'
  }});

  // Particles (WIP units moving on conveyor)
  const particles = [];
  const maxParticles = Math.min(36, Math.max(16, numStations * 4));

  function createParticle() {{
    return {{
      progress: Math.random(), // 0 to 1 along conveyor
      speed: 0.0012 + Math.random() * 0.0006,
      radius: 3.5,
      hue: Math.random() > 0.15 ? '#38bdf8' : '#34d399'
    }};
  }}

  for (let i = 0; i < maxParticles; i++) {{
    particles.push(createParticle());
  }}

  let pulseAngle = 0;

  function render() {{
    ctx.clearRect(0, 0, width, height);
    pulseAngle += 0.045;

    // 1. Draw Conveyor Track Base
    const startX = nodes[0].x;
    const endX = nodes[nodes.length - 1].x;

    // Outer glow for track
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.15)';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(startX, trackY);
    ctx.lineTo(endX, trackY);
    ctx.stroke();

    // Inner conveyor rail
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(startX, trackY);
    ctx.lineTo(endX, trackY);
    ctx.stroke();

    // Conveyor tick marks / movement illusion
    const dashOffset = (Date.now() / 35) % 20;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 6]);
    ctx.lineDashOffset = -dashOffset;
    ctx.beginPath();
    ctx.moveTo(startX, trackY);
    ctx.lineTo(endX, trackY);
    ctx.stroke();
    ctx.setLineDash([]); // Reset dash

    // 2. Draw & Update Moving WIP Particles
    particles.forEach(p => {{
      p.progress += p.speed;
      if (p.progress > 1) p.progress = 0;

      // Calculate current X position
      const curX = startX + p.progress * (endX - startX);

      // Check proximity to bottleneck stations to dynamically slow down (queue up)
      nodes.forEach(n => {{
        if (n.isBN && Math.abs(curX - n.x) < 45 && curX < n.x) {{
          p.speed = 0.0004; // Queue slowdown
        }} else if (Math.abs(curX - n.x) > 50) {{
          p.speed = 0.0012; // Normal conveyor velocity
        }}
      }});

      // Draw WIP bead
      ctx.beginPath();
      ctx.arc(curX, trackY, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = p.hue;
      ctx.shadowColor = p.hue;
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0; // Reset
    }});

    // 3. Draw Station Nodes & Status Badges
    nodes.forEach(n => {{
      const nx = n.x;
      const ny = n.y;

      if (n.type === 'inflow') {{
        // Inflow terminal
        ctx.fillStyle = 'rgba(30, 41, 59, 0.9)';
        ctx.strokeStyle = '#0284c7';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(nx - 36, ny - 32, 72, 64, 8);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('INFLOW', nx, ny - 14);

        ctx.fillStyle = '#38bdf8';
        ctx.font = 'bold 11px "Plus Jakarta Sans", sans-serif';
        ctx.fillText('Raw Stock', nx, ny + 4);

        ctx.fillStyle = '#64748b';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillText('BUFFER', nx, ny + 18);
      }} else if (n.type === 'outflow') {{
        // Outflow terminal
        ctx.fillStyle = 'rgba(30, 41, 59, 0.9)';
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(nx - 36, ny - 32, 72, 64, 8);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('OUTFLOW', nx, ny - 14);

        ctx.fillStyle = '#34d399';
        ctx.font = 'bold 11px "Plus Jakarta Sans", sans-serif';
        ctx.fillText('Shipped', nx, ny + 4);

        ctx.fillStyle = '#64748b';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillText('OUTPUT', nx, ny + 18);
      }} else {{
        // Production Workstation Node
        const isBN = n.isBN;
        const u = n.util;
        let strokeCol = '#10b981';
        let fillGlow = 'rgba(16, 185, 129, 0.15)';
        let statusText = 'STABLE';
        let statusCol = '#34d399';

        if (isBN) {{
          strokeCol = '#f43f5e';
          fillGlow = 'rgba(244, 63, 94, 0.35)';
          statusText = 'BOTTLENECK';
          statusCol = '#fb7185';
        }} else if (u > 70) {{
          strokeCol = '#f59e0b';
          fillGlow = 'rgba(245, 158, 11, 0.2)';
          statusText = 'WARNING';
          statusCol = '#fbbf24';
        }}

        // Pulsing Ring for Bottleneck
        if (isBN) {{
          const pulseR = 44 + Math.sin(pulseAngle) * 6;
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(nx, ny, pulseR, 0, Math.PI * 2);
          ctx.stroke();

          // Radar beacon sweep effect
          ctx.beginPath();
          ctx.arc(nx, ny, pulseR + 4, pulseAngle, pulseAngle + 0.8);
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.85)';
          ctx.lineWidth = 3;
          ctx.stroke();
        }}

        // Station Card Box
        const cardW = Math.max(78, Math.min(105, stepX - 12));
        const cardH = 74;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
        ctx.strokeStyle = strokeCol;
        ctx.lineWidth = isBN ? 2.5 : 1.5;

        ctx.beginPath();
        ctx.roundRect(nx - cardW / 2, ny - cardH / 2, cardW, cardH, 8);
        ctx.fill();
        ctx.stroke();

        // Station Header / Name
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 11px "Plus Jakarta Sans", sans-serif';
        ctx.textAlign = 'center';
        
        let dispName = n.name;
        if (dispName.length > 11) dispName = dispName.substring(0, 10) + '..';
        ctx.fillText(dispName, nx, ny - 18);

        // Utilization metric
        ctx.fillStyle = statusCol;
        ctx.font = 'bold 14px "JetBrains Mono", monospace';
        ctx.fillText(u + '%', nx, ny - 1);

        // Status Badge Pill
        ctx.fillStyle = fillGlow;
        ctx.strokeStyle = strokeCol;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(nx - cardW/2 + 6, ny + 13, cardW - 12, 16, 4);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = statusCol;
        ctx.font = 'bold 8px "JetBrains Mono", monospace';
        ctx.fillText(statusText, nx, ny + 24);
      }}
    }});

    requestAnimationFrame(render);
  }}

  render();
}})();
</script>
</body>
</html>
"""
    return html_code



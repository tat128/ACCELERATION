# Feature Implementation Plan

## What We're Building

Three things in one go:
1. **Animated Production Line Simulator** — live particle-flow animation in Tab 2, with bottleneck pulsing red
2. **3D What-If Surface Plot** — replaces the flat margin bar in Tab 4 with a full 3D margin landscape
3. **UI Classy Upgrade** — polish pass across the whole app

---

## 1. Animated Production Line Simulator

### How it works
- Uses `st.components.v1.html()` to embed a **custom HTML5 Canvas + JavaScript** animation directly in Tab 2
- Particles (small glowing dots) flow left-to-right through the station pipeline
- Each station node is sized/colored by its **real utilization %** from Arena data:
  - `<70%` — teal/green glow (healthy)
  - `70–85%` — amber warning glow
  - `>85%` — **pulsing red** bottleneck glow with ring animation
- Particles slow down or queue up at bottleneck stations (proportional to wait time)
- Throughput rate (parts/hr) drives the particle spawn rate
- Station labels show live utilization % and name
- Fully **driven by real data** — changes when you switch between Model 1/2/3

### Implementation
- New function `render_production_animation(station_metrics, analysis, model_name)` in `utils/chart_helpers.py`
- Returns a raw HTML string injected via `st.components.v1.html(html, height=320)`
- Added just **above** the existing SCADA strip in Tab 2 (replaces the static strip)

---

## 2. 3D Surface Plot — What-If Margin Landscape

### How it works
- X axis: **Scrap Reduction %** (0 → 50%, 20 steps)
- Y axis: **Bottleneck Idle Reduction %** (0 → 50%, 20 steps)
- Z axis: **Net Margin ($)** — computed via `what_if()` across the full grid (400 points)
- Rendered as `go.Surface` with a custom colorscale: dark navy → teal → gold for margin levels
- Current slider position is highlighted with a red sphere marker on the surface
- Rotating 3D view — judges can drag it
- Replaces (or sits beside) the flat `margin_comparison_bar` in Tab 4

### Implementation
- New function `margin_surface_3d(analysis, defect_sum, scrap_cost, downtime_cost, revenue_mult, current_util, current_scrap)` in `utils/chart_helpers.py`
- Computes 20×20 grid using `what_if()`, builds surface + current-position marker
- Added in Tab 4 below the What-If sliders, replacing the flat bar chart

---

## 3. UI Classy Upgrade

Key polish changes:
- **Tab bar styling** — custom CSS to give tabs a pill/underline style with active glow
- **Section headers** — gradient text effect on h4 headings
- **Stat cards** — add subtle shimmer/hover animation
- **Bottleneck card** — add animated left border pulse when CRITICAL
- **Scrollbar** — custom styled thin scrollbar
- **HUD banner** — add a slow scrolling marquee effect to the telemetry strip
- **Overall spacing** — tighten gaps, better padding rhythm

---

## Files Changed

| File | Changes |
|---|---|
| `utils/chart_helpers.py` | + `render_production_animation()` + `margin_surface_3d()` |
| `app.py` | Import new functions, replace static strip with animation in Tab 2, replace flat bar with 3D surface in Tab 4, CSS polish |

---

## Verification

- App compiles with `py_compile`
- Animation renders with all 3 models (1, 2, and 13 stations)
- 3D surface renders and updates live when sliders move
- No emojis, no "NEURAX 3.0" title anywhere

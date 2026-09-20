# Autonomous Quality & Process Intelligence Platform

> Executive-Grade Cyber-Physical Decision Support System for High-Throughput Manufacturing

---

## Problem Understanding

High-throughput industrial manufacturing environments face compounding economic losses from three critical challenges that conventional tools treat in silos:

| Operational Challenge | Direct Industrial Consequence |
|---|---|
| Subtle micro-defects undetected at production line speed | Defect escapes, customer warranty claims, expensive field recalls |
| Dynamic bottlenecks and queue accumulation across multi-stage lines | Starvation, idle work-in-progress (WIP), missed delivery schedules |
| Disconnect between defect manifestation and upstream process physics | Recurring quality failure modes without systemic remediation |

This platform delivers a **unified cyber-physical AI intelligence suite** that integrates real-time computer vision, Theory of Constraints (TOC) production telemetry, dynamic causal root-cause reasoning, 3D economic optimization landscapes, and neural network surrogate digital twins.

---

## End-to-End System Architecture

```
+---------------------------------------------------------------------------------------+
|                    Autonomous Quality & Process Intelligence Platform                 |
|                                                                                       |
|  +--------------------+     +------------------------+     +-----------------------+  |
|  |   Surface Images   |     |  Arena Simulation Logs |     |  Economic Parameters  |  |
|  |   (Crack, Rust,    |     |  (Model 1, 2, 3 CSVs   |     |  (Scrap, Downtime,    |  |
|  |    Hole, Scratch)  |     |   + MATLAB Weights)    |     |   Revenue Multiplier) |  |
|  +---------+----------+     +-----------+------------+     +-----------+-----------+  |
|            |                            |                              |              |
|  +---------v----------+     +-----------v------------+                 |              |
|  | YOLOv8 Classifier  |     | Theory of Constraints  |                 |              |
|  | + Grad-CAM Heatmap |     | Capacity & WIP Engine  |                 |              |
|  +---------+----------+     +-----------+------------+                 |              |
|            |                            |                              |              |
|  +---------v----------+     +-----------v------------+     +-----------v-----------+  |
|  | Defect Registry    |     | Real-Time Conveyor     |     | Profitability Engine  |  |
|  | + Drift Detection  |     | Physics Simulator      |     | + 3D Optimization     |  |
|  +---------+----------+     +-----------+------------+     +-----------+-----------+  |
|            |                            |                              |              |
|            +-------------+--------------+                              |              |
|                          |                                             |              |
|                 +--------v---------+                                   |              |
|                 | Dynamic Causal   |<----------------------------------+              |
|                 | Root-Cause Engine|                                                  |
|                 +--------+---------+                                                  |
|                          |                                                            |
|  +-----------------------v---------------------------------------------------------+  |
|  |                       Executive Cockpit Dashboard                                |  |
|  |  [Tab 1] Visual Quality & Grad-CAM Neural Inspection                            |  |
|  |  [Tab 2] Real-Time Conveyor Physics & TOC Line Health                           |  |
|  |  [Tab 3] Dynamic Multi-Tier Root Cause Correlation Engine                       |  |
|  |  [Tab 4] 3D Profitability Optimization Frontier & Economic Waterfall            |  |
|  |  [Tab 5] Neural Digital Twin (13 MATLAB Surrogate Models)                       |  |
|  +---------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------+
```

---

## Subsystem Capabilities

### 1. Vision & Neural Explainability (Tab 1)
- **Architecture**: Ultralytics YOLOv8 surface defect classifier coupled with Grad-CAM neural activation heatmaps.
- **Defect Classes**: `crack`, `hole`, `rust`, `scratch`, `normal`.
- **Tri-State Classification Gate**:
  - `Confidence >= 0.45` -> Deterministic Quality Decision (PASS / DEFECT).
  - `0.25 <= Confidence < 0.45` -> UNCERTAIN (diverted to automated review queue).
  - `Confidence < 0.25` -> NOVEL (out-of-distribution anomaly detection).
- **Inspection Modes**: Bounding Box View, Grad-CAM Activation Heatmaps, Clean Industrial View, and Raw Visual View.

### 2. Conveyor Physics & Theory of Constraints (Tab 2)
- **Live Canvas Physics**: Real-time conveyor animation simulating moving WIP beads, dynamic buffer accumulation, and constraint queuing.
- **TOC Analysis**: Automated detection of primary bottleneck stations, capacity utilization percentage, and queue delay latency.
- **Little's Law WIP Modeling**: Buffer estimations quantifying station inventory accumulation.
- **Multi-Model Benchmark**: Cross-scenario radar benchmarks comparing Model 1, Model 2, and Model 3 topologies.

### 3. Dynamic Root Cause Correlation (Tab 3)
- **Causal Inference**: Multi-tier thresholding matching station taxonomy against physical defect classes.
- **Evidence Cards**: Actionable engineering recommendations mapping process overloads to specific maintenance actions with HIGH, MEDIUM, and LOW confidence rankings.
- **Process Drift Telemetry**: Rolling window defect rate trajectory monitoring.

### 4. 3D Profitability Optimization Frontier (Tab 4)
- **Economic Waterfall**: Decomposition of Gross Revenue, Scrap Material Loss, Bottleneck Starvation Loss, and Net Operating Margin.
- **3D Response Surface**: Continuous 3D landscape modeling projected margin ($) across 400 combinations of scrap reduction and downtime mitigation.
- **Dynamic What-If Engine**: Real-time sensitivity sliders with instantaneous operating point visualization on the 3D frontier.

### 5. Neural Digital Twin Surrogate Metamodels (Tab 5)
- **Architecture**: Feedforward Artificial Neural Networks trained on 3,000 Rockwell Arena discrete-event simulation runs (`3000Samplesv3.mat`).
- **13 Pre-Loaded Models**: Multi-output neural models for Model 1, Model 2, and 13-station Model 3 topologies.
- **Interactive Metamodel Cockpit**: Sub-millisecond response curve sweeps, bottleneck risk gauges, and Arena vs ANN validation scatter plots with high R-squared metrics.

---

## Quick Start & Installation

Website link : https://acceleration-kvdiepxrn63pylpm4bj3zm.streamlit.app/

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (.env)
Copy the template configuration and customize paths if needed:
```bash
cp .env.example .env
```

Key environment variables:
```env
MODEL_DIR=D:/acceleration
MODEL_1_PATH=D:/acceleration/Model_1.csv
MODEL_2_PATH=D:/acceleration/Model_2.csv
MODEL_3_PATH=D:/acceleration/Model_3.csv
YOLO_WEIGHTS_PATH=D:/acceleration/best.pt
MATLAB_MAT_PATH=C:/Users/win-10/Downloads/3000Samplesv3.mat
DEFAULT_SCRAP_COST=15.0
DEFAULT_DOWNTIME_COST=2.50
DEFAULT_REVENUE_MULTIPLIER=4.0
```

### 3. Launch Platform
```bash
streamlit run app.py
```

---

## Project Structure

```
neurax_app/
├── app.py                          # Streamlit application entry point
├── .env                            # Machine-specific environment configuration
├── .env.example                    # Sample environment template
├── requirements.txt                # Python package dependencies
├── README.md                       # Platform documentation
├── modules/
│   ├── defect_detector.py          # YOLOv8 inference & Grad-CAM engine
│   ├── defect_registry.py          # Inspection session store & drift analytics
│   ├── production_analyzer.py      # Arena simulation parser & TOC bottleneck engine
│   ├── root_cause.py               # Dynamic multi-tier causal correlation engine
│   ├── profitability.py            # Financial waterfall & What-If simulator
│   └── ann_surrogate.py            # MATLAB ANN neural surrogate parser & inference
├── utils/
│   ├── chart_helpers.py            # Plotly 3D charts & HTML5 canvas conveyor simulator
│   └── constants.py                # System thresholds, palettes, and rules
└── data/
    └── matlab_models/              # Parsed MATLAB ANN surrogate weights & biases
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Deep Learning & Vision | Ultralytics YOLOv8, PyTorch, Grad-CAM, Pillow, OpenCV |
| Surrogate Metamodeling | Artificial Neural Networks (NumPy forward-pass runtime), SciPy |
| Interactive Simulation | HTML5 Canvas API, JavaScript RequestAnimationFrame Physics |
| Telemetry & Data Science | Pandas, NumPy, Theory of Constraints (TOC), Little's Law |
| Visual Analytics | Plotly 3D Surface / Graph Objects / Express |
| Application Framework | Streamlit 1.35+, Python-Dotenv |

---

*Autonomous Quality & Process Intelligence Platform -- Enterprise Industrial AI Decision Support*

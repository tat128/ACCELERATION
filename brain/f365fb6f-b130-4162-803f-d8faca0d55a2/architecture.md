# Autonomous Quality & Process Intelligence Platform
## Complete Technical Architecture & System Flowcharts

---

## 1. High-Level End-to-End System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. INGESTION & DATA SOURCES"]
        A1["Camera Feed / Product Surface Images"]
        A2["Rockwell Arena Simulation Data (.csv / .xlsx)"]
        A3["MATLAB Neural Models (.m / 3000Samplesv3.mat)"]
    end

    subgraph INFERENCE["2. DUAL INFERENCE & PROCESSING PIPELINES"]
        B1["YOLOv8-cls Deep Neural Classifier"]
        B2["Grad-CAM Explainability Engine"]
        B3["Dynamic Telemetry & TOC Analyzer"]
        B4["Pure-Python ANN Surrogate Engine"]
    end

    subgraph REASONING["3. DECISION & CAUSAL INTELLIGENCE CORE"]
        C1["Defect Registry & OOD Novelty Filter"]
        C2["Theory of Constraints Bottleneck Engine"]
        C3["Root-Cause Causal Matrix (Process <-> Defect)"]
        C4["Financial Loss Waterfall & What-If Engine"]
        C5["Prescriptive Capacity Optimizer (<0.05ms)"]
    end

    subgraph UI_LAYER["4. EXECUTIVE MISSION CONTROL COCKPIT"]
        D1["Tab 1: Quality Inspection Gallery & CAM Overlays"]
        D2["Tab 2: Plant Process Flow SCADA Topology"]
        D3["Tab 3: Root Cause Evidence & Defect Drift"]
        D4["Tab 4: Profitability & ROI Simulator"]
        D5["Tab 5: Digital Twin Metamodel & Gauges"]
    end

    A1 --> B1
    B1 --> B2
    B2 --> C1
    
    A2 --> B3
    B3 --> C2
    
    A3 --> B4
    B4 --> C5

    C1 --> C3
    C2 --> C3
    C2 --> C4
    C1 --> C4

    C1 --> D1
    C2 --> D2
    C3 --> D3
    C4 --> D4
    C5 --> D5
    B4 --> D5
```

---

## 2. Detailed Subsystem Workflows

### Subsystem A: Computer Vision & Explainability Pipeline

```mermaid
flowchart LR
    IMG["Raw Product Image"] --> PRE["Preprocessing (Resize & RGB Normalization)"]
    PRE --> YOLO["YOLOv8 Feature Extraction"]
    YOLO --> SOFTMAX["Softmax Multi-Class Probability"]
    
    SOFTMAX --> DECISION{"Confidence Score"}
    DECISION -->|">= 0.45"| CONF_PASS["Classified: Crack / Scratch / Rust / Hole / Normal"]
    DECISION -->|"0.25 to 0.44"| CONF_UNC["Flagged: UNCERTAIN (Human Review)"]
    DECISION -->|"< 0.25"| CONF_NOV["Flagged: NOVEL Signature (OOD Alert)"]
    
    YOLO --> CAM["Grad-CAM Backpropagation"]
    CAM --> HEATMAP["Target Layer Activation Heatmap"]
    HEATMAP --> BBOX["Adaptive Bounding Box Localization"]
    
    CONF_PASS --> REG["Session Defect Registry"]
    CONF_UNC --> REG
    CONF_NOV --> REG
    BBOX --> REG
```

---

### Subsystem B: Theory of Constraints (TOC) Telemetry Engine

```mermaid
flowchart TD
    RAW["Arena Simulation Stream (Model 1 / 2 / 3)"] --> PARSE["Dynamic Column & Header Discovery"]
    PARSE --> STATS["Station Metrics Calculation (Util %, Queue Delay, PPH)"]
    
    STATS --> TOC{"TOC Constraint Evaluator"}
    TOC -->|"argmax(Utilisation)"| BN["Identify Primary Bottleneck Station"]
    TOC --> LITTLE["Calculate Little's Law WIP Buffer: WIP = Throughput x CycleTime"]
    
    BN --> GAP["Calculate Throughput Gap: Demand - Total Parts"]
    GAP --> SCADA["Interactive Process Flow Topology Map"]
    LITTLE --> SCADA
```

---

### Subsystem C: Neural Digital Twin & Surrogate Pipeline

```mermaid
flowchart TD
    ARENA["Rockwell Arena (3,000 Stochastic Runs)"] --> MAT["Dataset Packaging: 3000Samplesv3.mat"]
    MAT --> MATLAB["MATLAB Neural Fitting (fitnet / tansig)"]
    MATLAB --> M_FILE["Exported Function Scripts (*.m)"]
    
    M_FILE --> PARSER["Python Regex & Weight Matrix Parser"]
    PARSER --> ENG["Vectorized ANN Inference Engine (NumPy)"]
    
    subgraph FORWARD_PASS["Real-Time Forward Pass (<0.05ms)"]
        INP["User Slider Inputs: x"] --> NORM["mapminmax Input Normalization: x_norm = 2*(x - xmin)/(xmax - xmin) - 1"]
        NORM --> HIDDEN["Hidden Layer: a1 = tansig(IW * x_norm + b1)"]
        HIDDEN --> OUT["Output Layer: a2 = purelin(LW * a1 + b2)"]
        OUT --> DENORM["mapminmax Reverse Scaling: y_hat = 0.5*(a2 + 1)*(ymax - ymin) + ymin"]
    end
    
    ENG --> FORWARD_PASS
    DENORM --> GAUGE["Real-Time Cyber Gauge Telemetry"]
    DENORM --> OPT["Prescriptive Capacity Optimization Algorithm"]
    DENORM --> VAL["Validation Matrix vs Ground Truth (R2, RMSE, MAE)"]
```

---

### Subsystem D: Root-Cause Causal Correlation Matrix

```mermaid
flowchart LR
    subgraph PROCESS_STREAM["PROCESS STREAM (Arena Telemetry)"]
        P1["Station Utilisation > 80%"]
        P2["Queue Delay > 6.0 min"]
        P3["Pacing Starvation"]
    end

    subgraph VISION_STREAM["VISION STREAM (YOLOv8 Inspection)"]
        V1["Crack Signatures"]
        V2["Scratch / Abrasion"]
        V3["Surface Rust / Oxidation"]
        V4["Hole / Impact Fracture"]
    end

    subgraph CAUSAL_MATRIX["EVIDENCE-BASED CAUSAL MATRIX"]
        R1["Drilling/Milling Overload -> Spindle Vibration & Tool Wear -> Micro-Cracks"]
        R2["Press/Blanking Impact -> Die Mechanical Fatigue -> Fractures & Holes"]
        R3["Extended Queue Dwell -> Oxidation Exposure -> Surface Rust"]
        R4["Assembly Pacing Stress -> Handling Abrasion -> Scratches"]
    end

    P1 & V1 --> R1
    P1 & V4 --> R2
    P2 & V3 --> R3
    P1 & V2 --> R4
    
    R1 & R2 & R3 & R4 --> EVIDENCE["Ranked Evidence Cards + Corrective Engineering Actions"]
```

---

### Subsystem E: Financial Loss Waterfall & What-If Simulator

```mermaid
flowchart TD
    DEMAND["Production Demand Target"] & OUTPUT["Finished Goods Output"] --> GROSS["Gross Revenue = Parts Produced x Selling Price"]
    
    DEFECTS["Defective Units Detected"] --> SCRAP["Scrap Loss = Defective Units x Scrap Cost"]
    BOTTLENECK["Constraint Starvation Time"] --> IDLE["Idle Loss = Idle Time x Downtime Cost"]
    GAP["Unmet Market Demand"] --> OPP["Opportunity Loss = Shortage Units x Lost Margin"]
    
    GROSS --> NET["Net Operating Margin = Gross Revenue - (Scrap + Idle + Opportunity Losses)"]
    SCRAP & IDLE & OPP --> NET
    
    NET --> WHATIF["What-If Optimization Engine"]
    SLIDER1["Idle Reduction Slider (%)"] --> WHATIF
    SLIDER2["Scrap Reduction Slider (%)"] --> WHATIF
    
    WHATIF --> RECOV["Projected Net Margin Lift ($) + Loss Recovered ($)"]
    RECOV --> EXPORT["1-Click Executive Intelligence Report (.md)"]
```

---

## 3. Technology & Latency Matrix

| Platform Subsystem | Core Technology | Mathematical / AI Model | Execution Latency |
| :--- | :--- | :--- | :--- |
| **Surface Inspection** | Ultralytics YOLOv8, PyTorch | Deep Convolutional Backbone + Softmax | ~12 ms / image |
| **Explainable Localization** | OpenCV, Grad-CAM Engine | Activation Map Gradient Backprop | ~18 ms / image |
| **Process Diagnostics** | Pandas, NumPy | Theory of Constraints (TOC), Little's Law | ~2 ms / dataset |
| **Digital Twin (ANN)** | Pure-Python Vectorized NumPy | 2-Layer Feedforward Net (10 Tansig Hidden) | **&lt; 0.05 ms** (Real-Time) |
| **Causal Correlation** | Rule Engine + Physics of Failure | Cross-Domain Evidence Mapping | ~1 ms |
| **Financial Engine** | Python Floating-Point Engine | Economic Loss Waterfall & Sensitivity | &lt; 0.5 ms |
| **Frontend UI** | Streamlit, Plotly, CSS3 Glassmorphism | Responsive Reactive Web App | 60 FPS Refresh |

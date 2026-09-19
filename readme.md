# Visual Inspection & Defect Root-Cause Assistant

**Neurax Hackathon 3.0 — Domain 2: AI in Industry and Automation**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8-green.svg)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

In high-throughput industrial manufacturing, optimizing production lines requires balancing **product quality**, **stage throughput**, and **overall profitability**. Traditional quality monitoring systems rely on isolated image classifiers or static KPI dashboards. These systems often fail to connect visual defect signatures with underlying machine bottlenecks, queue buildups, and batch-to-batch process drift.

This project presents a **Unified Industrial Decision-Support System** that bridges computer vision quality inspection, multi-stage discrete-event line analytics, and economic profit-margin modeling. By ingesting visual inspection streams along with multi-model operational logs (Drilling, Milling, and Assembly stages), the system delivers real-time diagnostic insights, flags novel defects, identifies active line constraints, and provides evidence-based financial and operational advisories.

---

## 🛠️ Key Architectural Pillars

### 1. Vision & Novelty Detection Engine (Quality)
* **High-Speed Localization:** Utilizes **YOLOv8** to classify and localize product surface defects (e.g., scratches, dents, pinholes, cracks).
* **Uncertainty & Novelty Thresholding:** Calculates prediction confidence bounds. Images yielding confidence scores below a customizable safety threshold ($\text{Conf} < 0.70$) are flagged as **Uncertain / Novel Defect** for human operator escalation rather than forcing low-confidence classifications.

### 2. Multi-Stage Operations & Bottleneck Engine (Process Flow)
* **Cross-Model Operational Tracking:** Evaluates process performance across **Model 1 (Baseline)**, **Model 2 (High Demand)**, and **Model 3 (Process Drift)** scenarios.
* **Bottleneck Identification Logic:** Dynamically monitors station capacity, work-in-process (WIP) queues, and machine utilization rates ($U \ge 0.85$). Identifies active constraints across Drilling, Milling, and Assembly lines based on queue growth and cycle-time imbalances.

### 3. Financial & Economic Loss Engine (Profitability)
* **Direct Scrap & Rework Quantification:** Maps visual defect rates to direct material scrap losses and rework labor costs.
* **Downtime & Capacity Cost Mapping:** Translates bottleneck wait times and queue delays into dollar losses based on machine idle overhead and lost target margins.
* **What-If Scenario Simulation:** Simulates operational adjustments (e.g., reducing Assembly cycle time by 10% or rebalancing Drilling feed rates) to project net margin recovery.

### 4. Executive Advisory Agent (Root-Cause Guidance)
* **Multimodal Synthesis:** Combines visual defect frequency with machine log anomalies into contextualized prompts for LLM-driven root-cause recommendations.
* **Evidence-Based Action Items:** Recommends targeted shop-floor maintenance interventions to prevent defect recurrence and alleviate line congestion.

---

## 🏗️ System Architecture

```
[ Inspection Images Zip ] --------> [ YOLOv8 Vision Engine ]
                                               |
                                     (Defect Rates & Novel Flags)
                                               |
                                               v
[ Models 1, 2, 3 Data Logs ] ----> [ Bottleneck & Flow Engine ]
                                               |
                                     (Station Util & Wait Times)
                                               |
                                               v
                                  [ Financial Loss Engine ]
                                               |
                                     (Scrap & Delay Costs)
                                               |
                                               v
                                 [ Streamlit Decision Center ]
                                 [ LLM Root-Cause Advisory ]
```

---

## 📊 Dataset Overview

The system ingests and processes multi-tier manufacturing datasets from the Mendeley Industrial Synthetic Line benchmark:

| Data Component | Format / Files | Description & Purpose |
| :--- | :--- | :--- |
| **Visual Inspection** | ZIP / `.jpg`, `.png` | Surface inspection images representing acceptable units, known defect types, and edge cases. |
| **Model 1 Data** | `Model_1.csv` | Baseline line performance tracking demand, throughput, and station metrics across Drilling, Milling, and Assembly. |
| **Model 2 Data** | `Model_2.csv` | High-throughput stress scenario highlighting line congestion and exponential queue growth at peak capacity. |
| **Model 3 Data** | `Model_3.csv` | Process drift scenario capturing tool wear, shift-to-shift variances, and yield fluctuations. |
| **Line Topology** | PDF / Flowcharts | Multi-stage manufacturing flow diagram used to define upstream/downstream station dependencies. |

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.9** or higher
* `pip` package manager
* (Optional) NVIDIA GPU with CUDA support for accelerated YOLO inference

### 1. Repository Setup
```bash
git clone https://github.com/your-username/neurax-defect-assistant.git
cd neurax-defect-assistant
```

### 2. Environment Setup
Create and activate a isolated Python virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the template configuration and supply your environment details:

```bash
cp .env.example .env
```

Edit your `.env` file:
```env
OPENAI_API_KEY=your_llm_api_key_here
CONFIDENCE_THRESHOLD=0.70
DATA_DIR=./data
MODEL_PATH=./weights/best.pt
```

### 5. Run the Application
Launch the interactive decision-support interface:

```bash
streamlit run app.py
```

---

## 📁 Directory Structure

```
neurax-defect-assistant/
├── data/
│   ├── raw_images/               # Unzipped product surface inspection images
│   ├── Model_1.csv               # Baseline process metrics dataset
│   ├── Model_2.csv               # High-throughput capacity dataset
│   └── Model_3.csv               # Process drift & variance dataset
├── models/
│   └── vision_detector.py        # YOLOv8 inference & uncertainty classification
├── modules/
│   ├── bottleneck_engine.py      # Multi-model station queue & utilization engine
│   ├── economic_engine.py        # Financial loss & profit margin estimator
│   └── advisory_agent.py         # Root-cause recommendation prompt pipeline
├── weights/
│   └── yolo_defects.pt           # Fine-tuned YOLOv8 weights
├── .env.example                  # Environment configuration template
├── app.py                        # Streamlit multi-tab dashboard interface
├── requirements.txt              # Dependencies (ultralytics, streamlit, pandas, etc.)
└── README.md                     # Project documentation
```

---

## ⚙️ Operational Advisory & Simulation Disclaimer

* **Software-Only & Simulation Scope:** This software is designed exclusively as an offline advisory decision-support system. All line interventions, bottleneck optimization strategies, and financial estimates are generated as simulated scenarios.
* **No Live Hardware Interaction:** The system does not connect directly to live programmable logic controllers (PLCs), robotic sorting arms, live camera feeds, or industrial automation hardware.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
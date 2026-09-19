# Visual Inspection & Defect Root-Cause Assistant
**Neurax Hackathon 3.0 — Domain 2: AI in Industry and Automation**

## 📌 Executive Summary
High-throughput manufacturing environments face complex tradeoffs between product quality, process capacity, and economic outcomes. Isolated defect classifiers or passive KPI dashboards fail to capture cross-stage operational dynamics. 

This project provides a **unified industrial decision-support system** that integrates computer vision quality inspection, discrete-event line analysis, and dynamic economic modeling. By linking defect detection directly to machine bottleneck identification and financial loss metrics, the system delivers real-time, evidence-based process recommendations to maximize net profitability.

---

## 🛠️ Key Features

1. **Defect & Novelty Detection Engine (Quality):**
   * Classifies products as acceptable vs. defective using vision models (YOLOv8/FastSAM).
   * Calculates confidence thresholds to flag uncertain or novel defect signatures for human review rather than forcing incorrect classifications.

2. **Operations & Bottleneck Engine (Process Flow):**
   * Analyzes multi-stage process metrics across Drilling, Milling, and Assembly stages.
   * Tracks cycle times, queue delays, and station utilization rates ($U \ge 0.85$) to isolate active line bottlenecks dynamically.

3. **Profitability & Economic Engine (Financial Analytics):**
   * Translates scrap counts, rework overhead, and bottleneck downtime into net dollar impact.
   * Runs "What-If" scenario simulations to predict how process modifications influence overall operating margin.

4. **Executive Advisory & Recommendations:**
   * Synthesizes cross-module diagnostics into actionable, evidence-based root-cause reports for shop-floor engineers.

---

## 🏗️ System Architecture

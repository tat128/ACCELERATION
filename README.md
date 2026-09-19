Neurax 3.0 Hackathon- AI in Industry and Automation
Visual Inspection & Defect Root-Cause Assistant

Problem understanding:
Throughout manufacturing, lines are challenging to optimize because product quality, process capacity, and economics are interrelated. Defect signatures are often obscured and difficult to identify at production rates, and process bottlenecks arise from imbalanced cycle-time, excessive work-in-process, downtime, changeovers, low utilization, scrap, and rework. 
We are building a unified industrial decision-support system that answers 
1) What is wrong with the product? (Identifying the defect)
2) Where is the defect? (Localizing the defect)
3) Is this a known defect? If known, which type? (Classification of the defect)
4) What process is most likely to cause the defect? (Corelates defects with Data)
5) **Where is the production flow constrained, and what does it cost?** — Detect bottlenecks, quantify throughput loss and profitability impact
6) What should be changed? (Evidence/Research based optimization)

Architecture:
flowchart TD
    A[Vision] --> D[Combining]
    B[Process] --> D[Combining]
    C[Economics] --> D[Combining]
    D --> E[Dashboard Display]

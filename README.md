Neurax 3.0 Hackathon- AI in Industry and Automation \n
Visual Inspection & Defect Root-Cause Assistant

_*Problem understanding:*_
Throughout manufacturing, lines are challenging to optimize because product quality, process capacity, and economics are interrelated. Defect signatures are often obscured and difficult to identify at production rates, and process bottlenecks arise from imbalanced cycle-time, excessive work-in-process, downtime, changeovers, low utilization, scrap, and rework. 
We are building a unified industrial decision-support system that answers 6 questions:
1) What is wrong with the product? (Identifying the defect)
2) Where is the defect? (Localizing the defect)
3) Is this a known defect? If known, which type? (Classification of the defect)
4) What process is most likely to cause the defect? (Corelates defects with Data)
5) What are the constrains in the production flow, and what do they cost? (Detecting bottlenecks and quantifying loss)
6) What should be changed? (Evidence/Research based optimization)

_Architecture:_
There are three major input models. Vision, Process, and Economics.

Vision:
1) Deep learning classifier for defect category, localized by heatmaps or bounding-box detector.
2) Has an uncertainty gate to identify any new defect categories.
   
Process Module:
1) Analyzes multi-stage production logs (cycle time, downtime, changeovers etc).
2) Identifies bottlenecks via queue time, downtime analysis
3) Correlates defects with relevant processes
   
Economics Module:
1) Looks at the revenue, scrap cost, rework cost, margins
2) Simulates profit margins with changing processes and quality conditions

Then we combine the three modules and display them in a dashboard.

*Approach:*
Vision:
Transfer a baseline model with augmentation for lightning, orientation, and batch variation. Train with temperature scaling and point accuracy. 

Process:
Analyze per-station cycle time, queue length, downtime statistics. Determine the major steps that slows down the process. Correlate the root cause of defects against processes.

Economics:
A transparent cost model with documented assumptions (like price per unit, cost of scrap, downtime per hour) and the simulate the profit margins for different scenarios

Combination: 
Corelates all the predictions from the three modules and optimizes it as per the available evidence and research.



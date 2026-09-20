# ─────────────────────────────────────────────────────────────────────────────
# Constants & Configuration
# ─────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

# ── YOLOv8 Model ─────────────────────────────────────────────────────────────
DEFECT_CLASSES = ['crack', 'hole', 'normal', 'rust', 'scratch']
DEFECT_CLASSES_CONCERN = ['crack', 'hole', 'rust', 'scratch']   # non-normal classes

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
NOVEL_THRESHOLD      = float(os.getenv("NOVEL_THRESHOLD",      "0.25"))

_yolo_env = os.getenv("YOLO_WEIGHTS_PATH", r"D:\acceleration\best.pt")
MODEL_PATHS = [
    _yolo_env,
    r"D:\acceleration\runs\classify\defect_classifier\model_v1\weights\best.pt",
]

# ── Visual Colours ────────────────────────────────────────────────────────────
DEFECT_COLORS = {
    'crack':     '#e74c3c',
    'hole':      '#9b59b6',
    'rust':      '#e67e22',
    'scratch':   '#f1c40f',
    'normal':    '#2ecc71',
    'UNCERTAIN': '#95a5a6',
    'NOVEL':     '#8e44ad',
}

STATUS_COLORS = {
    'PASS':      '#2ecc71',
    'DEFECT':    '#e74c3c',
    'UNCERTAIN': '#f39c12',
    'NOVEL':     '#9b59b6',
    'ERROR':     '#95a5a6',
}

STATION_COLORS = {
    'Drilling': '#e74c3c',
    'Milling':  '#f39c12',
    'Assembly': '#3498db',
    'Blanking': '#1abc9c',
    'Press':    '#e67e22',
    'Cell':     '#9b59b6',
    'Paint':    '#34495e',
    'Quality':  '#2ecc71',
    'Forklift': '#f1c40f',
}

# ── Production Thresholds ─────────────────────────────────────────────────────
HIGH_UTIL_THRESHOLD = 0.85   # flag station as overloaded

# ── Column Aliases (flexible CSV/XLSX parsing) ─────────────────────────────────
STATION_COLS = {
    'Drilling': {
        'util': ['Drilling Util', 'Drilling Utilization', 'Drill Util', 'Drilling_Util'],
        'wait': ['Drilling Waiting Time', 'Drilling Queue Time', 'Drill Waiting Time', 'Drilling_Waiting_Time'],
    },
    'Milling': {
        'util': ['Milling Util', 'Milling Utilization', 'Mill Util', 'Milling_Util'],
        'wait': ['Milling Waiting Time', 'Milling Queue Time', 'Mill Waiting Time', 'Milling_Waiting_Time'],
    },
    'Assembly': {
        'util': ['Assembly Util', 'Assembly Utilization', 'Asm Util', 'Assembly_Util'],
        'wait': ['Assembly Waiting Time', 'Assembly Queue Time', 'Assembly Time', 'Asm Waiting Time', 'Assembly_Waiting_Time'],
    },
}

DEMAND_COLS   = ['Demand', 'demand', 'Total Demand', 'c_TotalProducts']
PARTS_COLS    = ['Total parts', 'Total Parts', 'total_parts', 'Parts Produced', 'Entities Out', 'c_TotalProducts', 'c_Total_Products']
PPH_COLS      = ['Parts per hour', 'Parts Per Hour', 'parts_per_hour', 'PPH', 'Parts/Hour']
VA_TIME_COLS  = ['VA Time', 'Value Added Time', 'va_time', 'Part 1 VA Time']

# ── Root-Cause Correlation Rules ──────────────────────────────────────────────
# Each rule: condition lambda on flat_row dict, related defect types, explanation
CORRELATION_RULES = [
    {
        'id': 'DRILL_HIGH_UTIL',
        'condition': lambda r: r.get('Drilling Util', 0) > 0.85,
        'defects': ['crack', 'scratch'],
        'cause': 'Drilling station overloaded  -  tool vibration / wear',
        'evidence': 'Drilling Utilisation > 85%',
        'action': 'Inspect drill bits; reduce feed rate by 10%; schedule preventive maintenance.',
        'confidence': 'HIGH',
    },
    {
        'id': 'MILLING_HIGH_UTIL',
        'condition': lambda r: r.get('Milling Util', 0) > 0.85,
        'defects': ['crack', 'scratch'],
        'cause': 'Milling station overloaded  -  thermal stress / coolant insufficiency',
        'evidence': 'Milling Utilisation > 85%',
        'action': 'Increase coolant flow rate; add milling capacity buffer; inspect spindle bearings.',
        'confidence': 'HIGH',
    },
    {
        'id': 'ASSEMBLY_HIGH_UTIL',
        'condition': lambda r: r.get('Assembly Util', 0) > 0.85,
        'defects': ['scratch', 'hole'],
        'cause': 'Assembly station overloaded  -  rushed handling causes surface damage',
        'evidence': 'Assembly Utilisation > 85%',
        'action': 'Reduce WIP upstream; cross-train additional assembly operator; add buffer stock.',
        'confidence': 'HIGH',
    },
    {
        'id': 'DRILL_LONG_WAIT',
        'condition': lambda r: r.get('Drilling Waiting Time', 0) > 15,
        'defects': ['crack'],
        'cause': 'Parts waiting long at drilling  -  thermal cycling causes micro-cracks',
        'evidence': 'Drilling Waiting Time > 15 min',
        'action': 'Balance batch size entering drilling; investigate tool changeover frequency.',
        'confidence': 'MEDIUM',
    },
    {
        'id': 'MILLING_LONG_WAIT',
        'condition': lambda r: r.get('Milling Waiting Time', 0) > 12,
        'defects': ['rust', 'scratch'],
        'cause': 'WIP piling upstream of milling  -  surface oxidation during extended idle time',
        'evidence': 'Milling Waiting Time > 12 min',
        'action': 'Reduce inter-station transfer time; apply protective coating to WIP; improve scheduling.',
        'confidence': 'MEDIUM',
    },
    {
        'id': 'ASSEMBLY_VERY_LONG_WAIT',
        'condition': lambda r: r.get('Assembly Waiting Time', 0) > 50,
        'defects': ['scratch', 'rust'],
        'cause': 'Extreme assembly queue  -  surface degradation and contamination during idle',
        'evidence': 'Assembly Waiting Time > 50 min',
        'action': 'Expedite upstream throughput; protect queued WIP with covers; investigate scheduling policy.',
        'confidence': 'MEDIUM',
    },
    {
        'id': 'THROUGHPUT_DEFICIT',
        'condition': lambda r: (r.get('Demand', 0) - r.get('Total parts', 0)) / max(r.get('Demand', 1), 1) > 0.05,
        'defects': [],
        'cause': 'Throughput deficit >5%  -  systemic capacity constraint',
        'evidence': 'Parts Produced < 95% of Demand',
        'action': 'Conduct TOC analysis; elevate bottleneck capacity first; avoid sub-optimising non-bottleneck stations.',
        'confidence': 'HIGH',
    },
]

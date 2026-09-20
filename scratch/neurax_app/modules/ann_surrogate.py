"""
ANN Surrogate Metamodel & Digital Twin Engine.
Ported from MATLAB Neural Network Toolbox (fitnet / genFunction) models.
Executes pure vectorized NumPy neural forward passes with zero MATLAB runtime dependency.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class NormalizationStep:
    """mapminmax scaling settings."""
    xoffset: np.ndarray
    gain: np.ndarray
    ymin: float = -1.0


@dataclass
class ANNModelConfig:
    """Configuration and weights for a 2-layer Feedforward Neural Metamodel."""
    name: str
    description: str
    num_inputs: int
    num_outputs: int
    input_names: List[str]
    output_names: List[str]
    input_norm: NormalizationStep
    b1: np.ndarray          # (hidden_size,)
    IW1_1: np.ndarray       # (hidden_size, num_inputs)
    b2: np.ndarray          # (num_outputs,)
    LW2_1: np.ndarray       # (num_outputs, hidden_size)
    output_norm: NormalizationStep
    source_file: Optional[str] = None


class ANNParser:
    """Parses MATLAB genFunction .m files into ANNModelConfig objects."""

    @staticmethod
    def _extract_vector(text: str, var_name: str) -> Optional[np.ndarray]:
        # Match var_name = [...]; or var_name = number;
        pattern = rf"{re.escape(var_name)}\s*=\s*(\[[^\]]+\]|[-0-9.eE+]+)\s*;"
        match = re.search(pattern, text)
        if not match:
            return None
        val_str = match.group(1).strip()
        if val_str.startswith('[') and val_str.endswith(']'):
            val_str = val_str[1:-1].strip()
            # Split by ';' or whitespace/newlines
            elements = [float(x) for x in re.split(r'[;\s]+', val_str) if x.strip()]
            return np.array(elements, dtype=np.float64)
        else:
            return np.array([float(val_str)], dtype=np.float64)

    @staticmethod
    def _extract_matrix(text: str, var_name: str) -> Optional[np.ndarray]:
        # Match var_name = [ row1; row2; ... ];
        pattern = rf"{re.escape(var_name)}\s*=\s*\[([^\]]+)\]\s*;"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None
        matrix_str = match.group(1).strip()
        rows = [r.strip() for r in matrix_str.split(';') if r.strip()]
        matrix_data = []
        for r in rows:
            cols = [float(c) for c in re.split(r'[\s,]+', r) if c.strip()]
            matrix_data.append(cols)
        return np.array(matrix_data, dtype=np.float64)

    @classmethod
    def parse_m_file(cls, filepath: Union[str, Path]) -> Optional[ANNModelConfig]:
        path = Path(filepath)
        if not path.exists():
            return None

        content = path.read_text(encoding='utf-8', errors='ignore')

        # 1. Input normalization
        xoffset_in = cls._extract_vector(content, "x1_step1.xoffset")
        gain_in = cls._extract_vector(content, "x1_step1.gain")
        ymin_in_arr = cls._extract_vector(content, "x1_step1.ymin")
        ymin_in = float(ymin_in_arr[0]) if ymin_in_arr is not None else -1.0

        if xoffset_in is None or gain_in is None:
            return None

        input_norm = NormalizationStep(xoffset=xoffset_in, gain=gain_in, ymin=ymin_in)
        num_inputs = len(xoffset_in)

        # 2. Layer 1 (Hidden)
        b1 = cls._extract_vector(content, "b1")
        IW1_1 = cls._extract_matrix(content, "IW1_1")
        if IW1_1 is None:
            IW1_1 = cls._extract_vector(content, "IW1_1")
            if IW1_1 is not None:
                IW1_1 = IW1_1.reshape(-1, 1)

        # 3. Layer 2 (Output)
        b2 = cls._extract_vector(content, "b2")
        LW2_1 = cls._extract_matrix(content, "LW2_1")
        if LW2_1 is None:
            LW2_1 = cls._extract_vector(content, "LW2_1")
            if LW2_1 is not None:
                LW2_1 = LW2_1.reshape(1, -1)

        # 4. Output normalization
        xoffset_out = cls._extract_vector(content, "y1_step1.xoffset")
        gain_out = cls._extract_vector(content, "y1_step1.gain")
        ymin_out_arr = cls._extract_vector(content, "y1_step1.ymin")
        ymin_out = float(ymin_out_arr[0]) if ymin_out_arr is not None else -1.0

        if xoffset_out is None or gain_out is None or b1 is None or IW1_1 is None or b2 is None or LW2_1 is None:
            return None

        output_norm = NormalizationStep(xoffset=xoffset_out, gain=gain_out, ymin=ymin_out)
        num_outputs = len(xoffset_out)

        # Friendly naming heuristics
        stem = path.stem
        input_names = [f"Input {i+1}" for i in range(num_inputs)]
        output_names = [f"Output {i+1}" for i in range(num_outputs)]

        if stem == "Model1ANN":
            desc = "Model 1 Arena Metamodel — 3 Station Capacity Estimator"
            input_names = ["Part Interarrival / Volume Parameter"]
            output_names = ["Station 1 Utilization", "Station 2 Utilization", "Station 3 Utilization"]
        elif stem == "Model2ANN":
            desc = "Model 2 Arena Metamodel — Multi-Station (Drilling, Milling, Assembly)"
            input_names = ["Batch Processing Parameter"]
            output_names = ["Drilling Utilization", "Milling Utilization", "Assembly Utilization"]
        elif "Drilling" in stem:
            desc = "Model 2 Arena Metamodel — Dedicated Drilling Station"
            input_names = ["Drilling Inflow Parameter"]
            output_names = ["Drilling Utilization"]
        elif "Milling" in stem:
            desc = "Model 2 Arena Metamodel — Dedicated Milling Station"
            input_names = ["Milling Inflow Parameter"]
            output_names = ["Milling Utilization"]
        elif "Assembly" in stem:
            desc = "Model 2 Arena Metamodel — Dedicated Assembly Station"
            input_names = ["Assembly Inflow Parameter"]
            output_names = ["Assembly Utilization"]
        elif stem.startswith("c"):
            desc = f"Cell/Scenario Metamodel {stem} — Multi-Parameter Surrogate"
            input_names = [f"Station {i+1} Parameter" for i in range(num_inputs)]
            output_names = [f"Target Metric ({stem})"]
        else:
            desc = f"Neural Surrogate Metamodel ({stem})"

        return ANNModelConfig(
            name=stem,
            description=desc,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            input_names=input_names,
            output_names=output_names,
            input_norm=input_norm,
            b1=b1,
            IW1_1=IW1_1,
            b2=b2,
            LW2_1=LW2_1,
            output_norm=output_norm,
            source_file=str(path)
        )


class ANNSurrogateEngine:
    """High-performance vectorized neural forward-pass runtime."""

    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        self.models: Dict[str, ANNModelConfig] = {}
        if models_dir is not None:
            self.load_models_from_dir(models_dir)

    def load_models_from_dir(self, directory: Union[str, Path]) -> int:
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0

        count = 0
        for m_file in dir_path.glob("*ANN*.m"):
            config = ANNParser.parse_m_file(m_file)
            if config is not None:
                self.models[config.name] = config
                count += 1
        return count

    def predict(self, model_name: str, x: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Executes pure vectorized neural forward pass matching MATLAB genFunction:
        1. Xp1 = (X - xoffset) * gain + ymin
        2. a1 = 2 / (1 + exp(-2 * (b1 + IW1_1 * Xp1))) - 1   [tansig]
        3. a2 = b2 + LW2_1 * a1                               [purelin]
        4. Y = (a2 - ymin) / gain + xoffset
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in registry.")

        cfg = self.models[model_name]

        # Format input X into shape (num_inputs, Q) where Q is batch size
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1, 1)
        elif x_arr.ndim == 1:
            if cfg.num_inputs == 1:
                x_arr = x_arr.reshape(1, -1)
            else:
                if len(x_arr) == cfg.num_inputs:
                    x_arr = x_arr.reshape(cfg.num_inputs, 1)
                else:
                    x_arr = x_arr.reshape(1, -1)
        elif x_arr.ndim == 2:
            if x_arr.shape[0] != cfg.num_inputs and x_arr.shape[1] == cfg.num_inputs:
                x_arr = x_arr.T

        Q = x_arr.shape[1]

        # 1. mapminmax_apply for inputs
        # Xp1 = bsxfun(@minus, x, settings.xoffset) * settings.gain + settings.ymin
        xoffset_in = cfg.input_norm.xoffset.reshape(-1, 1)
        gain_in = cfg.input_norm.gain.reshape(-1, 1)
        ymin_in = cfg.input_norm.ymin

        Xp1 = (x_arr - xoffset_in) * gain_in + ymin_in

        # 2. Layer 1: Hyperbolic tangent sigmoid
        # a1 = tansig(repmat(b1, 1, Q) + IW1_1 * Xp1)
        b1 = cfg.b1.reshape(-1, 1)
        n1 = b1 + cfg.IW1_1 @ Xp1
        a1 = np.tanh(n1)  # Exact mathematical equivalent to 2 / (1 + exp(-2n)) - 1

        # 3. Layer 2: Linear output layer
        # a2 = repmat(b2, 1, Q) + LW2_1 * a1
        b2 = cfg.b2.reshape(-1, 1)
        a2 = b2 + cfg.LW2_1 @ a1

        # 4. mapminmax_reverse for outputs
        # Y = (a2 - settings.ymin) / settings.gain + settings.xoffset
        xoffset_out = cfg.output_norm.xoffset.reshape(-1, 1)
        gain_out = cfg.output_norm.gain.reshape(-1, 1)
        ymin_out = cfg.output_norm.ymin

        Y = (a2 - ymin_out) / gain_out + xoffset_out

        # Return shape (Q, num_outputs) if batch, or (num_outputs,) if single sample
        if Q == 1:
            return Y.flatten()
        return Y.T

    def evaluate_against_data(
        self,
        model_name: str,
        input_series: np.ndarray,
        actual_outputs: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates R^2, RMSE, MAE between surrogate model and real Arena simulation runs."""
        preds = self.predict(model_name, input_series)
        
        actual_outputs = np.asarray(actual_outputs)
        if actual_outputs.ndim == 1:
            actual_outputs = actual_outputs.reshape(-1, 1)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        metrics = []
        num_targets = min(preds.shape[1], actual_outputs.shape[1])

        for col in range(num_targets):
            y_true = actual_outputs[:, col]
            y_pred = preds[:, col]

            # Filter out NaNs
            valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            y_true_clean = y_true[valid_mask]
            y_pred_clean = y_pred[valid_mask]

            if len(y_true_clean) == 0:
                continue

            ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
            ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
            rmse = float(np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2)))
            mae = float(np.mean(np.abs(y_true_clean - y_pred_clean)))

            metrics.append({
                "target_index": col,
                "r2": float(np.clip(r2, 0.0, 1.0)),
                "rmse": rmse,
                "mae": mae,
                "y_true": y_true_clean,
                "y_pred": y_pred_clean
            })

        return {
            "model_name": model_name,
            "sample_count": len(input_series),
            "targets": metrics
        }

    def prescriptive_optimization(
        self,
        model_name: str,
        param_bounds: Optional[List[Tuple[float, float]]] = None,
        target_max_util: float = 0.85,
        steps: int = 200
    ) -> Dict[str, Any]:
        """
        Finds the optimal input setpoints that maximize throughput while maintaining
        all predicted station utilizations safely below target_max_util (e.g. 85%).
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")

        cfg = self.models[model_name]

        if param_bounds is None:
            # Derive reasonable bounds from normalization offsets
            param_bounds = []
            for idx in range(cfg.num_inputs):
                xoff = float(cfg.input_norm.xoffset[idx])
                param_bounds.append((xoff * 0.2, xoff * 3.0))

        if cfg.num_inputs == 1:
            x_min, x_max = param_bounds[0]
            x_vals = np.linspace(x_min, x_max, steps)
            preds = self.predict(model_name, x_vals)  # (steps, num_outputs)

            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)

            # Max utilization across all predicted stations at each setpoint
            max_utils = np.max(preds, axis=1)
            safe_mask = max_utils <= target_max_util

            if np.any(safe_mask):
                # Highest throughput within safe envelope
                optimal_idx = np.where(safe_mask)[0][-1]
                opt_input = float(x_vals[optimal_idx])
                opt_preds = preds[optimal_idx].tolist()
                status = "OPTIMAL"
            else:
                # Minimum risk setpoint
                optimal_idx = int(np.argmin(max_utils))
                opt_input = float(x_vals[optimal_idx])
                opt_preds = preds[optimal_idx].tolist()
                status = "SUB-OPTIMAL (Overloaded)"

            return {
                "status": status,
                "optimal_input": [opt_input],
                "predicted_utilization": opt_preds,
                "max_utilization": float(max_utils[optimal_idx]),
                "safe_range": (float(x_vals[safe_mask][0]), float(x_vals[safe_mask][-1])) if np.any(safe_mask) else None,
                "sweep_x": x_vals.tolist(),
                "sweep_y": preds.tolist(),
                "output_names": cfg.output_names
            }
        else:
            # Multi-input grid scan
            grid_points = [np.linspace(b[0], b[1], 15) for b in param_bounds]
            mesh = np.meshgrid(*grid_points)
            coords = np.vstack([m.flatten() for m in mesh]).T  # (N, num_inputs)

            preds = self.predict(model_name, coords)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)

            max_utils = np.max(preds, axis=1)
            safe_mask = max_utils <= target_max_util

            if np.any(safe_mask):
                opt_idx = np.where(safe_mask)[0][-1]
                status = "OPTIMAL"
            else:
                opt_idx = int(np.argmin(max_utils))
                status = "SUB-OPTIMAL (Overloaded)"

            return {
                "status": status,
                "optimal_input": coords[opt_idx].tolist(),
                "predicted_utilization": preds[opt_idx].tolist(),
                "max_utilization": float(max_utils[opt_idx]),
                "output_names": cfg.output_names
            }


# Singleton engine instance
_ENGINE_INSTANCE: Optional[ANNSurrogateEngine] = None

def get_surrogate_engine(models_dir: Optional[Union[str, Path]] = None) -> ANNSurrogateEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        if models_dir is None:
            # Look inside default locations
            base_dir = Path(__file__).resolve().parent.parent
            models_dir = base_dir / "data" / "matlab_models"
            if not models_dir.exists():
                models_dir = Path("C:/Users/win-10/Downloads/Matlab Models")
        _ENGINE_INSTANCE = ANNSurrogateEngine(models_dir)
    return _ENGINE_INSTANCE

"""YOLOv8 inference wrapper with clean HUD annotation, Grad-CAM Heatmaps, and probability breakdown."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from utils.constants import (
    DEFECT_CLASSES,
    DEFECT_COLORS,
    STATUS_COLORS,
    CONFIDENCE_THRESHOLD,
    NOVEL_THRESHOLD,
    MODEL_PATHS,
)

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

_model_cache: dict[str, object] = {}


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model(custom_path: Optional[str] = None) -> tuple[object | None, str | None]:
    """Return (model, error_msg). Model is cached per path."""
    if not _YOLO_AVAILABLE:
        return None, (
            "ultralytics not installed.\n"
            "Run: pip install ultralytics"
        )

    search_paths = ([custom_path] if custom_path else []) + MODEL_PATHS
    for path in search_paths:
        if not path:
            continue
        if path in _model_cache:
            return _model_cache[path], None
        if os.path.exists(path):
            try:
                model = YOLO(path)
                _model_cache[path] = model
                return model, None
            except Exception as exc:
                return None, f"Failed to load {path}: {exc}"

    checked = [p for p in search_paths if p]
    return None, (
        f"Model weights not found.\n"
        f"Checked:\n" + "\n".join(f"  • {p}" for p in checked) +
        "\n\nPlace best.pt at D:\\acceleration\\best.pt or upload below."
    )


# ── PyTorch Grad-CAM Neural Activation ─────────────────────────────────────────

def compute_gradcam(
    yolo_model: object,
    img_pil: Image.Image,
    target_class_idx: int,
) -> tuple[np.ndarray | None, list[list[float]]]:
    """Computes exact pre-softmax Grad-CAM activation heatmap from YOLOv8's feature layer."""
    h1 = None
    h2 = None
    try:
        pytorch_model = yolo_model.model
        pytorch_model.eval()
        
        orig_w, orig_h = img_pil.size
        img_resized = img_pil.resize((224, 224)).convert('RGB')
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        tensor.requires_grad = True

        features_list = []
        logits_list = []

        def f_feat(module, inp, out):
            features_list.append(out)

        def f_lin(module, inp, out):
            logits_list.append(out)

        # Hook last C2f feature extractor and linear classification head
        target_feat_layer = pytorch_model.model[8] if hasattr(pytorch_model, 'model') and len(pytorch_model.model) > 8 else pytorch_model.model[-2]
        h1 = target_feat_layer.register_forward_hook(f_feat)

        if hasattr(pytorch_model, 'model') and hasattr(pytorch_model.model[-1], 'linear'):
            h2 = pytorch_model.model[-1].linear.register_forward_hook(f_lin)

        out = pytorch_model(tensor)

        if logits_list:
            raw_logits = logits_list[-1]
            score = raw_logits[0, target_class_idx]
        else:
            if isinstance(out, tuple):
                out = out[0]
            score = out[0, target_class_idx]

        feat = features_list[-1]
        grads = torch.autograd.grad(score, feat, retain_graph=False)[0]

        h1.remove()
        h1 = None
        if h2 is not None:
            h2.remove()
            h2 = None

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feat).sum(dim=1)[0]
        cam = F.relu(cam).detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        cam_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)

        # Multi-percentile bounding box extraction
        boxes = []
        if cam.max() > 0.01:
            for p_cut in [85, 75, 65, 50]:
                thresh_val = max(0.20, float(np.percentile(cam_resized, p_cut)))
                _, bin_mask = cv2.threshold(np.uint8(cam_resized * 255), int(thresh_val * 255), 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                min_area = (orig_w * orig_h) * 0.005
                for cnt in contours:
                    if cv2.contourArea(cnt) < min_area:
                        continue
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw > (orig_w * 0.90) and bh > (orig_h * 0.90):
                        continue
                    pad = 4
                    bx1 = max(0, x - pad)
                    by1 = max(0, y - pad)
                    bx2 = min(orig_w, x + bw + pad)
                    by2 = min(orig_h, y + bh + pad)
                    boxes.append([float(bx1), float(by1), float(bx2), float(by2)])
                if boxes:
                    break

        return cam_resized, boxes[:2]
    except Exception:
        if h1 is not None:
            try: h1.remove()
            except Exception: pass
        if h2 is not None:
            try: h2.remove()
            except Exception: pass
        return None, []


def localize_defect_fallback(img_pil: Image.Image, cls_name: str) -> list[list[float]]:
    """Guaranteed Computer Vision bounding box localizer based on gradient energy and defect morphology."""
    orig_w, orig_h = img_pil.size
    img_gray = np.array(img_pil.convert('L'))
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Gradient energy
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, thresh = cv2.threshold(grad_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    min_area = (orig_w * orig_h) * 0.005
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if valid_contours:
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        for cnt in valid_contours[:2]:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > (orig_w * 0.90) and bh > (orig_h * 0.90):
                continue
            pad = 6
            bx1 = max(0, x - pad)
            by1 = max(0, y - pad)
            bx2 = min(orig_w, x + bw + pad)
            by2 = min(orig_h, y + bh + pad)
            boxes.append([float(bx1), float(by1), float(bx2), float(by2)])

    if not boxes:
        # Default centered focus box (25% area) if anomalous without distinct edge
        margin_x = int(orig_w * 0.25)
        margin_y = int(orig_h * 0.25)
        boxes.append([float(margin_x), float(margin_y), float(orig_w - margin_x), float(orig_h - margin_y)])

    return boxes[:2]


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    image_input,
    model=None,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Run YOLOv8 on one image (PIL Image, file path, or uploaded bytes).
    """
    if model is None:
        model, err = load_model()
        if model is None:
            return {'status': 'ERROR', 'error': err,
                    'defect_type': 'unknown', 'confidence': 0.0,
                    'class_probs': {}, 'bbox': None, 'all_detections': [],
                    'annotated_image': None, 'heatmap_image': None, 'boxed_image': None, 'raw_image': None}

    if isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        try:
            img = Image.open(image_input).convert('RGB')
        except Exception as exc:
            return {'status': 'ERROR', 'error': str(exc),
                    'defect_type': 'unknown', 'confidence': 0.0,
                    'class_probs': {}, 'bbox': None, 'all_detections': [],
                    'annotated_image': None, 'heatmap_image': None, 'boxed_image': None, 'raw_image': None}

    try:
        results = model(img, verbose=False)
    except Exception as exc:
        return {'status': 'ERROR', 'error': f'Inference error: {exc}',
                'defect_type': 'unknown', 'confidence': 0.0,
                'class_probs': {}, 'bbox': None, 'all_detections': [],
                'annotated_image': None, 'heatmap_image': None, 'boxed_image': None, 'raw_image': img}

    result = results[0]
    detections: list[dict] = []
    class_probs: dict[str, float] = {}
    cam_heatmap: np.ndarray | None = None
    has_native_boxes = False

    # Detection model mode
    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        has_native_boxes = True
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = (result.names or {}).get(cls_id, DEFECT_CLASSES[cls_id] if cls_id < len(DEFECT_CLASSES) else 'unknown').lower()
            detections.append({
                'defect_type': cls_name,
                'confidence': conf,
                'bbox': box.xyxy[0].tolist(),
            })

    # Classification model mode
    elif hasattr(result, 'probs') and result.probs is not None:
        probs_raw = result.probs.data.cpu().numpy()
        top_cls_idx = int(result.probs.top1)
        top_conf = float(result.probs.top1conf.item())
        cls_name = (result.names or {}).get(top_cls_idx, DEFECT_CLASSES[top_cls_idx] if top_cls_idx < len(DEFECT_CLASSES) else 'unknown').lower()

        for idx, p in enumerate(probs_raw):
            c_name = (result.names or {}).get(idx, DEFECT_CLASSES[idx] if idx < len(DEFECT_CLASSES) else f'class_{idx}').lower()
            class_probs[c_name] = round(float(p), 4)

        if cls_name not in ('normal', 'acceptable', 'pass'):
            cam_heatmap, local_boxes = compute_gradcam(model, img, top_cls_idx)
            if not local_boxes:
                local_boxes = localize_defect_fallback(img, cls_name)
            for b in local_boxes:
                detections.append({
                    'defect_type': cls_name,
                    'confidence': top_conf,
                    'bbox': b,
                })
        else:
            detections.append({
                'defect_type': cls_name,
                'confidence': top_conf,
                'bbox': None,
            })

    # Determine status
    if not detections:
        status = 'PASS'
        primary_type = 'normal'
        primary_conf = 1.0
        primary_bbox = None
    else:
        concern_first = sorted(
            detections,
            key=lambda d: (d['defect_type'] not in ('normal', 'acceptable', 'pass'), d['confidence']),
            reverse=True,
        )
        top = concern_first[0]
        primary_type = top['defect_type']
        primary_conf = top['confidence']
        primary_bbox = top['bbox']

        if primary_conf < NOVEL_THRESHOLD:
            status = 'NOVEL'
        elif primary_conf < conf_threshold:
            status = 'UNCERTAIN'
        elif primary_type in ('normal', 'acceptable', 'pass'):
            status = 'PASS'
        else:
            status = 'DEFECT'

    # Clean Industrial HUD Image (Clean, sharp, no fake bounding boxes)
    clean_hud_img = annotate_clean_hud(img.copy(), primary_type, primary_conf, status)

    # Boxed Image
    boxed_img = annotate_boxes_hud(img.copy(), detections, status)

    # Grad-CAM Heatmap Image
    heatmap_img = generate_heatmap_overlay(img.copy(), cam_heatmap, status, primary_type)

    return {
        'status': status,
        'defect_type': primary_type,
        'confidence': primary_conf,
        'class_probs': class_probs,
        'bbox': primary_bbox,
        'all_detections': detections,
        'annotated_image': clean_hud_img,   # Clean HUD view by default
        'boxed_image': boxed_img,           # Targeted Bounding Box view
        'heatmap_image': heatmap_img,       # Grad-CAM Heatmap view
        'raw_image': img,                   # Untouched Raw Image
    }


# ── Clean Industrial HUD Annotator (No Distortion / Zoom) ─────────────────────

def annotate_clean_hud(img: Image.Image, defect_type: str, confidence: float, status: str) -> Image.Image:
    """Draws a clean top HUD telemetry bar onto the image without distorting or zooming."""
    w, h = img.size
    draw = ImageDraw.Draw(img)

    color_hex = STATUS_COLORS.get(status, '#95a5a6')
    bar_h = max(24, int(h * 0.12))

    # Top banner background
    draw.rectangle([0, 0, w, bar_h], fill='#111625')
    draw.rectangle([0, bar_h - 2, w, bar_h], fill=color_hex)

    # Status Badge
    badge_text = f" {status} "
    label_text = f"{defect_type.upper()} ({confidence*100:.1f}%)"

    try:
        font_b = ImageFont.truetype("arialbd.ttf", max(11, int(bar_h * 0.45)))
        font_l = ImageFont.truetype("arial.ttf", max(11, int(bar_h * 0.42)))
    except Exception:
        font_b = ImageFont.load_default()
        font_l = ImageFont.load_default()

    # Draw badge
    draw.rectangle([6, 4, 6 + len(badge_text)*8, bar_h - 6], fill=color_hex)
    draw.text((8, 6), badge_text, fill='white', font=font_b)

    # Draw class & conf
    draw.text((16 + len(badge_text)*8, 6), label_text, fill='#ecf0f1', font=font_l)

    return img


# ── Bounding Box HUD Annotator ────────────────────────────────────────────────

def annotate_boxes_hud(img: Image.Image, detections: list[dict], status: str) -> Image.Image:
    """Draws bounding boxes around localized defect regions."""
    w, h = img.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img)

    for det in detections:
        bbox = det.get('bbox')
        dtype = det['defect_type'].lower()
        if bbox is None or dtype in ('normal', 'acceptable', 'pass'):
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color_hex = DEFECT_COLORS.get(dtype, '#e74c3c')
        c_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Semi-transparent fill
        draw_overlay.rectangle([x1, y1, x2, y2], fill=c_rgb + (35,))
        
        # Bounding box outline
        box_thickness = max(2, int(min(w, h) / 180))
        draw.rectangle([x1, y1, x2, y2], outline=color_hex, width=box_thickness)

        # Target corner brackets (industrial HUD look)
        corner_len = max(6, min(int((x2 - x1) * 0.22), int((y2 - y1) * 0.22), 16))
        c_th = box_thickness + 1
        draw.line([(x1, y1), (x1 + corner_len, y1)], fill='#ffffff', width=c_th)
        draw.line([(x1, y1), (x1, y1 + corner_len)], fill='#ffffff', width=c_th)
        draw.line([(x2, y1), (x2 - corner_len, y1)], fill='#ffffff', width=c_th)
        draw.line([(x2, y1), (x2, y1 + corner_len)], fill='#ffffff', width=c_th)
        draw.line([(x1, y2), (x1 + corner_len, y2)], fill='#ffffff', width=c_th)
        draw.line([(x1, y2), (x1, y2 - corner_len)], fill='#ffffff', width=c_th)
        draw.line([(x2, y2), (x2 - corner_len, y2)], fill='#ffffff', width=c_th)
        draw.line([(x2, y2), (x2, y2 - corner_len)], fill='#ffffff', width=c_th)

        # Tag
        label = f" {dtype.upper()} {det['confidence']*100:.0f}% "
        try:
            font = ImageFont.truetype("arialbd.ttf", max(10, int(min(w, h) / 45)))
        except Exception:
            font = ImageFont.load_default()

        tag_h = max(16, int(min(w, h) / 35))
        lbl_y1 = max(0, y1 - tag_h - 2)
        lbl_y2 = lbl_y1 + tag_h
        lbl_w = len(label) * int(tag_h * 0.62)
        draw.rectangle([x1, lbl_y1, x1 + lbl_w, lbl_y2], fill=color_hex)
        draw.text((x1 + 3, lbl_y1 + 2), label, fill='#ffffff', font=font)

    return Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')


# ── Heatmap Overlay ───────────────────────────────────────────────────────────

def generate_heatmap_overlay(
    img: Image.Image,
    cam: np.ndarray | None,
    status: str,
    defect_type: str,
) -> Image.Image:
    """Creates a smooth thermal activation heatmap overlay on the original image."""
    if cam is None or status == 'PASS':
        return img

    img_rgb = np.array(img.convert('RGB'))
    cam_uint = np.uint8(np.clip(cam * 255, 0, 255))
    heatmap_color = cv2.applyColorMap(cam_uint, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_rgb, 0.65, heatmap_color, 0.35, 0)
    return Image.fromarray(blended)

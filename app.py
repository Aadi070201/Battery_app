# app.py
import os
import tempfile
import warnings
import cv2
import numpy as np
import torch
import gradio as gr

from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.data import PredictDataset
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
MODEL_PATH = os.getenv("MODEL_PATH") or hf_hub_download(
    repo_id="Aadi070201/Aadi070201padimcylindrical",  # <- your HF model repo
    filename="padim_model_cylindrical.pth"
)
#MODEL_PATH = "padim_model_cylindrical.pth"

# ---------------------- Utilities ----------------------
def _load_model(path: str) -> Padim:
    model = Padim()
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)  # robust to version diffs
    model.eval()
    return model

def _to_uint8_rgb(img_np: np.ndarray) -> np.ndarray:
    """Ensure image is RGB uint8 [H,W,3]. Gradio gives RGB float/uint8; normalize if needed."""
    if img_np.dtype != np.uint8:
        # assume 0..255 floats/ints, clip and convert
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    return img_np

def _get_scalar(x):
    if hasattr(x, "item"):
        return x.item()
    if isinstance(x, (list, tuple)) and x:
        return _get_scalar(x[0])
    return float(x)

def _get_pred_fields(batch):
    """Handle both singular/plural field names across anomalib versions."""
    # score
    if hasattr(batch, "pred_score"):
        score = _get_scalar(batch.pred_score)
    elif hasattr(batch, "pred_scores"):
        score = _get_scalar(batch.pred_scores)
    else:
        score = None

    # label
    if hasattr(batch, "pred_label"):
        label = int(_get_scalar(batch.pred_label))
    elif hasattr(batch, "pred_labels"):
        label = int(_get_scalar(batch.pred_labels))
    else:
        label = None

    # anomaly map (pixel-level scores)
    amap = None
    if hasattr(batch, "anomaly_map"):
        amap = batch.anomaly_map
    elif hasattr(batch, "anomaly_maps"):
        amap = batch.anomaly_maps

    # predicted mask (binary), if available
    pmask = None
    if hasattr(batch, "pred_mask"):
        pmask = batch.pred_mask
    elif hasattr(batch, "pred_masks"):
        pmask = batch.pred_masks

    return score, label, amap, pmask

def _normalize_map(am_np: np.ndarray, robust: bool = True) -> np.ndarray:
    """Normalize anomaly map to [0, 1] similar to PaDiM visualizations."""
    am = am_np.astype(np.float32)
    if robust:
        # robust percentile normalization to avoid outlier washout
        lo = np.percentile(am, 2)
        hi = np.percentile(am, 98)
        if hi <= lo:
            lo, hi = am.min(), am.max()
    else:
        lo, hi = am.min(), am.max()
    denom = (hi - lo) if (hi - lo) > 1e-8 else 1.0
    am = np.clip((am - lo) / denom, 0.0, 1.0)
    return am

def _overlay_heatmap(image_rgb_u8: np.ndarray, amap_np: np.ndarray, alpha: float = 0.45, blur_ksize: int = 11) -> np.ndarray:
    """Create PaDiM-like overlay: upsample -> (optional) blur -> colormap -> blend."""
    H, W = image_rgb_u8.shape[:2]
    # amap could be [H,W], [1,H,W], or [1,1,H,W] depending on version
    am = amap_np
    if am.ndim == 3 and am.shape[0] in (1, 3):  # [C,H,W]
        am = am[0]
    if am.ndim == 4:  # [N,1,H,W] -> take first
        am = am[0, 0]
    am = cv2.resize(am, (W, H), interpolation=cv2.INTER_LINEAR)
    am = _normalize_map(am, robust=True)

    if blur_ksize and blur_ksize > 1:
        am = cv2.GaussianBlur(am, (blur_ksize, blur_ksize), 0)

    am_u8 = (255.0 * am).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(am_u8, cv2.COLORMAP_JET)            # BGR
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)             # RGB

    overlay = cv2.addWeighted(image_rgb_u8, 1.0 - alpha, heat_rgb, alpha, 0)
    return overlay

def _draw_mask_contours(overlay_rgb_u8: np.ndarray, pmask: torch.Tensor | np.ndarray) -> np.ndarray:
    """If a binary predicted mask exists, draw its contour(s) like many PaDiM demos do."""
    H, W = overlay_rgb_u8.shape[:2]
    if isinstance(pmask, torch.Tensor):
        m = pmask.detach().cpu().numpy()
    else:
        m = pmask
    # pmask can be [H,W], [1,H,W], [N,1,H,W]; squeeze
    while m.ndim > 2:
        m = m[0]
    if m.dtype != np.uint8:
        m = (m > 0.5).astype(np.uint8) * 255
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = overlay_rgb_u8.copy()
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)   # white contour
    return out

# ---------------------- Model / Engine ----------------------
model = _load_model(MODEL_PATH)
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
engine = Engine(accelerator=accelerator, devices=1)

# ---------------------- Gradio handler ----------------------
def run_inference(image_np: np.ndarray):
    """Gradio passes a numpy RGB image. Save to temp, run PredictDataset, build proper overlay."""
    # Ensure RGB uint8
    image_rgb = _to_uint8_rgb(image_np)

    # Save uploaded image to a temp file for PredictDataset
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    # Predict with anomalib
    with torch.no_grad():
        batches = engine.predict(model=model, dataset=PredictDataset(path=tmp_path))
    batch = batches[0]

    # Get fields robustly across versions
    score, label, amap, pmask = _get_pred_fields(batch)
    if score is None or label is None:
        return image_rgb, "Prediction fields not found."

    label_text = "Anomalous ❌" if label == 1 else "Normal ✅"

    # Build PaDiM-like overlay
    overlay = image_rgb
    if amap is not None:
        # convert to numpy
        if isinstance(amap, torch.Tensor):
            amap_np = amap.detach().cpu().numpy()
        else:
            amap_np = np.array(amap)
        overlay = _overlay_heatmap(image_rgb, amap_np, alpha=0.45, blur_ksize=11)

    # Draw predicted mask contours if present
    if pmask is not None:
        overlay = _draw_mask_contours(overlay, pmask)

    # Compose info line
    info = f"Score: {score:.4f} | Prediction: {label_text}"

    # Clean temp
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return overlay, info

# ---------------------- UI ----------------------
demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=[
        gr.Image(type="numpy", label="Heatmap Overlay"),
        gr.Textbox(label="Prediction"),
    ],
    title="🔍 PaDiM Anomaly Detection",
    description="Upload an image (phone/desktop). Returns PaDiM-like heatmap overlay and prediction.",
)

if __name__ == "__main__":
    # Access from your phone on the same Wi-Fi: http://<your-PC-IP>:7860
    demo.launch(server_name="0.0.0.0", server_port=7860)

import hashlib
import io
import json
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Prevent transformers from eagerly importing TensorFlow/Flax, which on Colab
# pulls in googleapiclient -> httplib2 -> pyparsing and can fail with version
# mismatches. We only use the PyTorch backend.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from transformers import AutoProcessor

from gui_actor.constants import chat_template  # noqa: F401  (imported for parity with demo/app.py)
from gui_actor.inference import inference
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer

from ui_detr import UIDetr

MAX_PIXELS = 1920 * 1080

LOG_DIR = Path(os.environ.get("PREDICT_LOG_DIR", Path(__file__).resolve().parent / "logs"))
LOG_IMAGES_DIR = LOG_DIR / "images"
LOG_JSONL = LOG_DIR / "requests.jsonl"
LOGGING_ENABLED = os.environ.get("PREDICT_LOG_DISABLED", "0") != "1"


def resize_image(image: Image.Image, resize_to_pixels: int = MAX_PIXELS) -> Image.Image:
    w, h = image.size
    if resize_to_pixels is not None and (w * h) != resize_to_pixels:
        ratio = (resize_to_pixels / (w * h)) ** 0.5
        image = image.resize((int(w * ratio), int(h * ratio)))
    return image


@asynccontextmanager
async def lifespan(app: FastAPI):
    if torch.cuda.is_available():
        gui_actor_id = "microsoft/GUI-Actor-7B-Qwen2.5-VL"
        device_map = "cuda"
        device = "cuda"
    else:
        gui_actor_id = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
        device_map = "cpu"
        device = "cpu"

    print(f"[startup] Loading GUI-Actor: {gui_actor_id} on {device}")
    data_processor = AutoProcessor.from_pretrained(gui_actor_id)
    tokenizer = data_processor.tokenizer
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        gui_actor_id,
        torch_dtype=torch.float16,
        device_map=device_map,
    ).eval()
    print("[startup] GUI-Actor loaded")

    print("[startup] Loading UI-DETR: racineai/UI-DETR-1")
    ui_detr = UIDetr(repo_id="racineai/UI-DETR-1", device=device)
    print("[startup] UI-DETR loaded")

    app.state.device = device
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.data_processor = data_processor
    app.state.ui_detr = ui_detr

    yield

    del app.state.model
    del app.state.ui_detr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="GUI-Actor + UI-DETR", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": getattr(app.state, "device", None),
        "guiactor_loaded": hasattr(app.state, "model"),
        "uidetr_loaded": hasattr(app.state, "ui_detr"),
    }


async def _read_image(upload: UploadFile):
    """Return (PIL Image, raw bytes) so callers can both decode and hash without
    re-encoding (which would change the digest)."""
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty upload: {upload.filename}")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image {upload.filename}: {e}")
    return img, data


def _save_image_bytes(data: bytes, original_filename: Optional[str]) -> str:
    """Store an uploaded image under logs/images/<sha256><ext>, deduped. Returns
    the relative filename (not absolute path) so it stays portable in the log."""
    LOG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()
    ext = ""
    if original_filename:
        ext = Path(original_filename).suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            ext = ""
    filename = f"{digest}{ext}"
    dest = LOG_IMAGES_DIR / filename
    if not dest.exists():
        dest.write_bytes(data)
    return filename


def _log_request(entry: dict) -> None:
    if not LOGGING_ENABLED:
        return
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[log] failed to write request log: {e}")


def _build_conversation(input_image: Image.Image, reference_image: Optional[Image.Image], instruction: str):
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a GUI agent. Given a screenshot of the current GUI (image 1) and a "
                        "reference image (image 2) / human instruction, your task is to locate the screen "
                        "element that corresponds to the instruction. You should output a PyAutoGUI action "
                        "that performs a click on the correct position. To indicate the click location, we "
                        "will use some special tokens, which is used to refer to a visual patch later. For "
                        "example, you can output: pyautogui.click(<your_special_token_here>)."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": input_image}],
        },
    ]
    if reference_image is not None:
        conversation[-1]["content"].append({"type": "image", "image": reference_image})
    conversation[-1]["content"].append({"type": "text", "text": instruction})
    return conversation


def _select_bbox(detections, point_px):
    """Return (selected_det, source) per the user-confirmed rule."""
    if not detections:
        return None, "no_detections"

    px, py = point_px
    containing = [
        d for d in detections
        if d["bbox"][0] <= px <= d["bbox"][2] and d["bbox"][1] <= py <= d["bbox"][3]
    ]
    if containing:
        best = max(containing, key=lambda d: d["score"])
        return best, "contained_highest_score"

    def dist(d):
        cx = (d["bbox"][0] + d["bbox"][2]) / 2.0
        cy = (d["bbox"][1] + d["bbox"][3]) / 2.0
        return math.hypot(cx - px, cy - py)

    return min(detections, key=dist), "nearest_center_fallback"


@app.post("/predict")
@torch.inference_mode()
async def predict(
    input_image: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None),
    instruction: Optional[str] = Form(None),
    score_threshold: float = Form(0.3),
):
    t_start = time.time()
    img, input_bytes = await _read_image(input_image)
    ref_bytes: Optional[bytes] = None
    if reference_image is not None:
        ref, ref_bytes = await _read_image(reference_image)
    else:
        ref = None

    if img.size[0] * img.size[1] > MAX_PIXELS:
        img = resize_image(img)
    if ref is not None and ref.size[0] * ref.size[1] > MAX_PIXELS:
        ref = resize_image(ref)

    instruction_text = instruction.strip() if instruction and instruction.strip() else "Locate the matching UI element."
    conversation = _build_conversation(img, ref, instruction_text)

    try:
        pred = inference(
            conversation,
            app.state.model,
            app.state.tokenizer,
            app.state.data_processor,
            use_placeholder=True,
            topk=3,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GUI-Actor inference failed: {e}")

    px_norm, py_norm = pred["topk_points"][0]
    w, h = img.size
    point_px = (px_norm * w, py_norm * h)

    detections = app.state.ui_detr.detect(img, score_threshold=score_threshold)
    selected, source = _select_bbox(detections, point_px)

    response = {
        "point": {"x": float(px_norm), "y": float(py_norm)},
        "point_pixel": {"x": float(point_px[0]), "y": float(point_px[1])},
        "bbox": None,
        "bbox_pixel": None,
        "bbox_score": None,
        "bbox_label": None,
        "bbox_source": source,
        "image_size": {"width": w, "height": h},
        "num_detections": len(detections),
    }

    if selected is not None:
        x1, y1, x2, y2 = selected["bbox"]
        response["bbox_pixel"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        response["bbox"] = {"x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h}
        response["bbox_score"] = selected["score"]
        response["bbox_label"] = selected["label"]

    if LOGGING_ENABLED:
        input_saved = _save_image_bytes(input_bytes, input_image.filename)
        ref_saved = (
            _save_image_bytes(ref_bytes, reference_image.filename)
            if ref_bytes is not None and reference_image is not None
            else None
        )
        _log_request({
            "ts": time.time(),
            "latency_s": round(time.time() - t_start, 3),
            "request": {
                "input_image_file": input_saved,
                "input_image_filename": input_image.filename,
                "reference_image_file": ref_saved,
                "reference_image_filename": reference_image.filename if reference_image else None,
                "instruction": instruction,
                "score_threshold": score_threshold,
            },
            "response": response,
        })

    return response

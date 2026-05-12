from ui_detr import UIDetr
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference
from transformers import AutoProcessor
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import torch
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


from gui_actor.constants import chat_template  # noqa: F401  (imported for parity with demo/app.py)


MAX_PIXELS = 1280 * 720

LOG_DIR = Path(os.environ.get("PREDICT_LOG_DIR",
               Path(__file__).resolve().parent / "logs"))
LOG_IMAGES_DIR = LOG_DIR / "images"
LOG_OVERLAYS_DIR = LOG_DIR / "overlays"
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
    attn_implementation = None
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
        attn_implementation=attn_implementation
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


class WireTimingMiddleware:
    """Logs wire-level timing for /predict: bytes/time spent receiving the
    request body, server think-time between last byte in and first byte out,
    and bytes/time spent sending the response."""

    def __init__(self, app, path_prefix: str = "/predict"):
        self.app = app
        self.path_prefix = path_prefix

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not scope.get("path", "").startswith(self.path_prefix):
            await self.app(scope, receive, send)
            return

        first_in = last_in = first_out = last_out = None
        bytes_in = 0
        bytes_out = 0

        async def recv():
            nonlocal first_in, last_in, bytes_in
            msg = await receive()
            if msg["type"] == "http.request":
                now = time.perf_counter()
                if first_in is None:
                    first_in = now
                bytes_in += len(msg.get("body", b""))
                if not msg.get("more_body"):
                    last_in = now
            return msg

        async def snd(msg):
            nonlocal first_out, last_out, bytes_out
            if msg["type"] == "http.response.start":
                first_out = time.perf_counter()
            elif msg["type"] == "http.response.body":
                bytes_out += len(msg.get("body", b""))
                if not msg.get("more_body"):
                    last_out = time.perf_counter()
            await send(msg)

        await self.app(scope, recv, snd)

        def ms(a, b):
            return (b - a) * 1000 if (a is not None and b is not None) else 0.0

        wire_in_ms = ms(first_in, last_in)
        think_ms = ms(last_in, first_out)
        wire_out_ms = ms(first_out, last_out)
        total_ms = ms(first_in, last_out)
        print(f"[wire] in={wire_in_ms:.0f}ms ({bytes_in/1024:.0f}KB) "
              f"think={think_ms:.0f}ms "
              f"out={wire_out_ms:.0f}ms ({bytes_out/1024:.1f}KB) "
              f"total={total_ms:.0f}ms")


app.add_middleware(WireTimingMiddleware)


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
        raise HTTPException(
            status_code=400, detail=f"Empty upload: {upload.filename}")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Could not decode image {upload.filename}: {e}")
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


def _save_overlay(
    base_image: Image.Image,
    detections: list,
    point_px: tuple,
    selected_bbox: Optional[list],
    bbox_source: str,
    input_sha: str,
    request_ts: float,
) -> str:
    """Draw all UI-DETR detections, the GUI-Actor point, and highlight the
    selected bbox. Saved under logs/overlays/<sha>_<ts_ms>.png."""
    LOG_OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

    canvas = base_image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)

    # All detections in muted yellow
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 200, 0), width=2)
        label = f"{det['label']}:{det['score']:.2f}"
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(255, 200, 0))

    # Selected bbox in green (if any) — drawn after so it sits on top
    if selected_bbox is not None:
        x1, y1, x2, y2 = selected_bbox
        color = (0, 200, 0) if bbox_source == "contained_highest_score" else (
            255, 140, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        draw.text((x1 + 4, max(0, y1 - 14)),
                  f"selected ({bbox_source})", fill=color)

    # GUI-Actor point in red, drawn last so it always shows
    px, py = point_px
    r = 10
    draw.ellipse([px - r, py - r, px + r, py + r],
                 outline=(255, 0, 0), width=4)
    draw.line([px - r - 4, py, px + r + 4, py], fill=(255, 0, 0), width=2)
    draw.line([px, py - r - 4, px, py + r + 4], fill=(255, 0, 0), width=2)

    ts_ms = int(request_ts * 1000)
    filename = f"{input_sha[:12]}_{ts_ms}.png"
    canvas.save(LOG_OVERLAYS_DIR / filename, "PNG")
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
        conversation[-1]["content"].append(
            {"type": "image", "image": reference_image})
    conversation[-1]["content"].append({"type": "text", "text": instruction})
    return conversation


def _attention_bbox(region_points, n_width, n_height, image_size):
    """Build a pixel bbox covering all patches in the highest-attention region.

    `region_points` is a list of (cx, cy) normalized patch-center coords (already
    shifted by +0.5/n). Invert that shift to recover integer (col, row) patch
    indices, then take the union of the patch cells as the bbox.
    """
    if not region_points:
        return None
    w, h = image_size
    cols, rows = [], []
    for cx, cy in region_points:
        col = int(round(cx * n_width - 0.5))
        row = int(round(cy * n_height - 0.5))
        cols.append(max(0, min(n_width - 1, col)))
        rows.append(max(0, min(n_height - 1, row)))
    x1 = min(cols) / n_width * w
    y1 = min(rows) / n_height * h
    x2 = (max(cols) + 1) / n_width * w
    y2 = (max(rows) + 1) / n_height * h
    return [x1, y1, x2, y2]


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
    use_ui_detr: bool = Form(False),
):
    t_start = time.time()
    print(f"[predict] hit @ {time.strftime('%H:%M:%S')} "
          f"use_ui_detr={use_ui_detr} threshold={score_threshold}")

    t_upload_start = time.perf_counter()
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
    upload_ms = (time.perf_counter() - t_upload_start) * 1000

    instruction_text = instruction.strip() if instruction and instruction.strip(
    ) else "Locate the matching UI element."
    conversation = _build_conversation(img, ref, instruction_text)

    t_fwd_start = time.perf_counter()
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
        raise HTTPException(
            status_code=500, detail=f"GUI-Actor inference failed: {e}")
    forward_ms = (time.perf_counter() - t_fwd_start) * 1000

    t_post_start = time.perf_counter()
    px_norm, py_norm = pred["topk_points"][0]
    w, h = img.size
    point_px = (px_norm * w, py_norm * h)

    if use_ui_detr:
        detections = app.state.ui_detr.detect(
            img, score_threshold=score_threshold)
        selected, source = _select_bbox(detections, point_px)
        selected_bbox = selected["bbox"] if selected is not None else None
        selected_score = selected["score"] if selected is not None else None
        selected_label = selected["label"] if selected is not None else None
    else:
        detections = []
        top_region_patches = pred.get("topk_points_all") or []
        attn_bbox = _attention_bbox(
            top_region_patches[0] if top_region_patches else [],
            pred["n_width"],
            pred["n_height"],
            (w, h),
        )
        if attn_bbox is None:
            selected_bbox = None
            selected_score = None
            source = "no_attention_region"
        else:
            selected_bbox = attn_bbox
            selected_score = pred["topk_values"][0] if pred.get(
                "topk_values") else None
            source = "attention_region"
        selected_label = None

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

    if selected_bbox is not None:
        x1, y1, x2, y2 = selected_bbox
        response["bbox_pixel"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        response["bbox"] = {"x1": x1 / w,
                            "y1": y1 / h, "x2": x2 / w, "y2": y2 / h}
        response["bbox_score"] = selected_score
        response["bbox_label"] = selected_label

    post_ms = (time.perf_counter() - t_post_start) * 1000

    if LOGGING_ENABLED:
        ts_now = time.time()
        t_send_start = time.perf_counter()
        input_saved = _save_image_bytes(input_bytes, input_image.filename)
        ref_saved = (
            _save_image_bytes(ref_bytes, reference_image.filename)
            if ref_bytes is not None and reference_image is not None
            else None
        )
        try:
            overlay_saved = _save_overlay(
                base_image=img,
                detections=detections,
                point_px=point_px,
                selected_bbox=selected_bbox,
                bbox_source=source,
                input_sha=Path(input_saved).stem,
                request_ts=ts_now,
            )
        except Exception as e:
            print(f"[log] failed to save overlay: {e}")
            overlay_saved = None

        _log_request({
            "ts": ts_now,
            "latency_s": round(ts_now - t_start, 3),
            "timings_ms": {
                "upload": round(upload_ms, 1),
                "forward": round(forward_ms, 1),
                "post": round(post_ms, 1),
                "total": round((ts_now - t_start) * 1000, 1),
            },
            "request": {
                "input_image_file": input_saved,
                "input_image_filename": input_image.filename,
                "reference_image_file": ref_saved,
                "reference_image_filename": reference_image.filename if reference_image else None,
                "instruction": instruction,
                "score_threshold": score_threshold,
            },
            "overlay_file": overlay_saved,
            "response": response,
        })
        save_ms = (time.perf_counter() - t_send_start) * 1000
    else:
        save_ms = 0.0

    total_ms = (time.time() - t_start) * 1000
    print(f"[predict] done upload={upload_ms:.0f}ms forward={forward_ms:.0f}ms "
          f"post={post_ms:.0f}ms save={save_ms:.0f}ms total={total_ms:.0f}ms")

    return response

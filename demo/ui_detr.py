import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from rfdetr.detr import RFDETRMedium


class UIDetr:
    def __init__(
        self,
        repo_id: str = "racineai/UI-DETR-1",
        weights_filename: str = "model.pth",
        resolution: int = 1600,
        device: str = "cuda",
    ):
        weights_path = hf_hub_download(repo_id=repo_id, filename=weights_filename)
        self.model = RFDETRMedium(pretrain_weights=weights_path, resolution=resolution)
        self.device = device

    def detect(self, image: Image.Image, score_threshold: float = 0.3):
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image)
        detections = self.model.predict(arr, threshold=score_threshold)

        boxes = detections.xyxy
        scores = detections.confidence
        class_ids = getattr(detections, "class_id", None)

        results = []
        for idx, (box, score) in enumerate(zip(boxes, scores)):
            label = str(int(class_ids[idx])) if class_ids is not None else "ui_element"
            results.append({
                "bbox": [float(v) for v in box.tolist()],
                "score": float(score),
                "label": label,
            })
        return results

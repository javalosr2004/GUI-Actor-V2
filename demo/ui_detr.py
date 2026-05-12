import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


class UIDetr:
    def __init__(self, model_id: str = "yuyangstatistics/UI-DETR", device: str = "cuda"):
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def detect(self, image: Image.Image, score_threshold: float = 0.3):
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=score_threshold
        )[0]
        id2label = self.model.config.id2label
        return [
            {
                "bbox": [float(v) for v in box.tolist()],
                "score": float(score),
                "label": id2label[int(label)],
            }
            for box, score, label in zip(results["boxes"], results["scores"], results["labels"])
        ]

"""Model loading, preprocessing, radiograph gate, and prediction."""
import hashlib, os
from pathlib import Path
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pt"
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.75"))
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SIZE = 224


def build_model(arch: str, n_classes: int, pretrained: bool = False):
    import torch.nn as nn
    import torchvision.models as m
    w = "DEFAULT" if pretrained else None
    if arch == "densenet121":
        net = m.densenet121(weights=w)
        net.classifier = nn.Linear(net.classifier.in_features, n_classes)
    elif arch == "resnet18":
        net = m.resnet18(weights=w)
        net.fc = nn.Linear(net.fc.in_features, n_classes)
    elif arch == "efficientnet_b0":
        net = m.efficientnet_b0(weights=w)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, n_classes)
    else:
        raise ValueError(f"Unknown arch {arch}")
    return net


def looks_like_xray(img: Image.Image):
    """Cheap sanity gate: radiographs are near-greyscale with real contrast.
    This is NOT a validated out-of-distribution detector."""
    a = np.asarray(img.convert("RGB").resize((128, 128))).astype(float)
    sat = (a.max(2) - a.min(2)).mean()
    contrast = a.mean(2).std()
    if sat > 14:
        return False, "Image is colourful; chest X-rays are greyscale. Upload a radiograph."
    if contrast < 15:
        return False, "Image is nearly blank or has too little contrast to analyse."
    return True, ""


class Classifier:
    def __init__(self):
        self.net = None
        self.demo = True
        self.classes = ["NORMAL", "PNEUMONIA"]
        self.version = "demo-untrained"
        self.temperature = 1.0
        self.metrics = {}
        self.load()

    def load(self):
        if not MODEL_PATH.exists():
            return
        try:
            import torch
            ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            net = build_model(ck["arch"], len(ck["classes"]))
            net.load_state_dict(ck["state_dict"])
            net.eval()
            self.net, self.classes = net, list(ck["classes"])
            self.temperature = float(ck.get("temperature", 1.0))
            self.version = ck.get("version", ck["arch"])
            self.metrics = ck.get("metrics", {})
            self.demo = False
        except Exception as e:  # keep app alive, stay in demo mode
            print(f"[WARN] Could not load model: {e}")

    def _tensor(self, img: Image.Image):
        import torch
        a = np.asarray(img.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32) / 255.0
        a = (a - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
        return torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0)

    def predict(self, img: Image.Image) -> dict:
        if self.demo:
            # Deterministic placeholder so the UI can be demonstrated. NOT a prediction.
            h = hashlib.sha256(img.convert("L").resize((16, 16)).tobytes()).digest()
            raw = np.array([h[i] + 1 for i in range(len(self.classes))], dtype=float)
            p = raw / raw.sum()
        else:
            import torch
            with torch.no_grad():
                logits = self.net(self._tensor(img)) / self.temperature
                p = torch.softmax(logits, 1)[0].numpy()
        i = int(np.argmax(p))
        conf = float(p[i])
        status = "ok" if conf >= REVIEW_THRESHOLD else "needs_review"
        return {"label": self.classes[i], "confidence": conf,
                "probs": {c: float(x) for c, x in zip(self.classes, p)},
                "status": status, "model_version": self.version, "demo": self.demo}

    def info(self):
        return {"version": self.version, "classes": self.classes, "demo": self.demo,
                "review_threshold": REVIEW_THRESHOLD, "temperature": self.temperature,
                "metrics": self.metrics}


classifier = Classifier()

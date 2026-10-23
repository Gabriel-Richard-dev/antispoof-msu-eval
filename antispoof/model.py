"""MobileNetV2 + BiLSTM multiframe (mesma arquitetura e pré-processamento do vigia)."""
import numpy as np

NUM_FRAMES = 11
INPUT_SIZE = (224, 224)
CROP_MARGIN = 0.1
ATTACK_THRESHOLD = 0.982  # ataque quando (1 - p_real) > limiar


def build_model():
    import torch
    from torch import nn
    from torchvision import models

    class CNNRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = models.mobilenet_v2(weights=None)
            self.cnn.classifier[1] = nn.Linear(1280, 512)
            self.rnn = nn.LSTM(512, 128, num_layers=2, batch_first=True, bidirectional=True)
            self.linear = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

        def forward(self, x):
            b, t, c, h, w = x.shape
            features = self.cnn(x.reshape(b * t, c, h, w)).reshape(b, t, -1)
            sequence, _ = self.rnn(features)
            return self.linear(sequence[:, -1])

    return CNNRNN()


def load_model(path, device):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state", checkpoint)
    model = build_model()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def make_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(INPUT_SIZE), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def crop_face(frame_bgr, box, margin=CROP_MARGIN):
    """Recorte idêntico ao do vigia: margem de 10% do maior lado, saída RGB."""
    import cv2

    x1, y1, x2, y2 = box
    m = int(margin * max(x2 - x1, y2 - y1))
    h, w = frame_bgr.shape[:2]
    left, top = max(0, int(x1 - m)), max(0, int(y1 - m))
    right, bottom = min(w, int(x2 + m)), min(h, int(y2 + m))
    if right <= left or bottom <= top:
        return None
    return cv2.cvtColor(frame_bgr[top:bottom, left:right], cv2.COLOR_BGR2RGB)


def windows_to_tensor(windows, transform):
    """windows: lista de janelas, cada uma com NUM_FRAMES recortes RGB uint8."""
    import torch
    from PIL import Image

    return torch.stack([
        torch.stack([transform(Image.fromarray(face)) for face in faces]) for faces in windows
    ])


def predict_real(model, batch, device):
    import torch

    with torch.inference_mode():
        out = model(batch.to(device)).squeeze(1).float().cpu().numpy()
    if not np.all(np.isfinite(out)) or out.min() < 0 or out.max() > 1:
        raise ValueError("Saída do modelo inválida.")
    return out

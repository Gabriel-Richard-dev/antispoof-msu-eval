"""Detector de rosto YOLOv8s-face (ONNX), mesma lógica do vigia."""
import cv2
import numpy as np


class YOLOFaceDetector:
    def __init__(self, model_path, conf_threshold=0.5, provider="auto"):
        import onnxruntime as ort

        available = ort.get_available_providers()
        if provider == "cpu" or "CUDAExecutionProvider" not in available:
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.provider = self.session.get_providers()[0]

    def detect_bbox(self, image_bgr):
        """Maior face acima do limiar de confiança, ou None."""
        h, w = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
        output = self.session.run(None, {self.input_name: tensor})[0]
        faces = []
        for pred in output[0].T:
            if pred[4] > self.conf_threshold:
                cx, cy, bw, bh = pred[:4]
                cx, bw = cx * w / 640, bw * w / 640
                cy, bh = cy * h / 640, bh * h / 640
                box = [max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)),
                       min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))]
                if box[2] > box[0] and box[3] > box[1]:
                    faces.append(box)
        return max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if faces else None

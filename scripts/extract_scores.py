"""Etapa 0: roda os pesos do artigo (MobileNetV2-BiLSTM) nos vídeos do MSU-MFSD e grava os scores por janela.

Não calcula métricas: isso é do report.py. Assim dá para testar outros limiares/agregações sem rodar o modelo de novo.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # importa antes do onnxruntime para ele achar as libs CUDA do torch

from antispoof import dataset, model as M
from antispoof.detector import YOLOFaceDetector

FIELDS = ["video_id", "client", "split", "device", "attack_type", "label", "window_idx", "start_frame",
          "status", "real_prob", "attack_score"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def window_indices(n_frames, stride, hop, max_windows):
    k = 0
    while True:
        start = k * hop
        idx = [start + j * stride for j in range(M.NUM_FRAMES)]
        if idx[-1] >= n_frames:
            return
        yield k, idx
        k += 1
        if max_windows and k >= max_windows:
            return


class VideoSource:
    """Leitura sequencial: get(i) deve ser chamado com i crescente."""

    def __init__(self, path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        self.opened = self.cap.isOpened()
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.opened else 0
        if self.opened and self.n_frames <= 0:  # contêiner sem contagem: conta lendo
            self.n_frames = 0
            while self.cap.grab():
                self.n_frames += 1
            self.cap.release()
            self.cap = cv2.VideoCapture(self.path)
        self.pos = 0

    def get(self, i):
        while self.pos < i:
            if not self.cap.grab():
                return None
            self.pos += 1
        ok, frame = self.cap.read()
        self.pos += 1
        return frame if ok else None

    def close(self):
        self.cap.release()


class FrameSource:
    def __init__(self, folder):
        self.files = dataset.list_frames(folder)
        self.opened = bool(self.files)
        self.n_frames = len(self.files)

    def get(self, i):
        return cv2.imread(str(self.files[i])) if i < self.n_frames else None

    def close(self):
        pass


def item_windows(item, detector, stride, hop, max_windows, frames_mode):
    """Devolve (janelas_validas, linhas_sem_decisão). Cada janela = (k, start, [11 recortes RGB]).

    frames_mode 'asis': pasta de frames já recortados no rosto, usados como estão (só BGR->RGB).
    frames_mode 'yolo': refaz detecção + recorte com margem de 10%, como no pipeline do artigo.
    Vídeos sempre passam pelo detector YOLO.
    """
    src = VideoSource(item.path) if item.kind == "video" else FrameSource(item.path)
    if not src.opened:
        return [], [(0, 0, "unreadable")]
    plan = list(window_indices(src.n_frames, stride, hop, max_windows))
    needed = sorted({i for _, idx in plan for i in idx})
    crops, nobox = {}, set()
    for i in needed:
        frame = src.get(i)
        if frame is None:
            if item.kind == "video":
                break  # fim do vídeo
            continue
        if item.kind == "frames" and frames_mode == "asis":
            face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            box = detector.detect_bbox(frame)
            face = M.crop_face(frame, box) if box is not None else None
        if face is None:
            nobox.add(i)
        else:
            crops[i] = face
    src.close()
    valid, failed = [], []
    for k, idx in plan:
        if any(j in nobox for j in idx):
            failed.append((k, idx[0], "no_face"))
        elif all(j in crops for j in idx):
            valid.append((k, idx[0], [crops[j] for j in idx]))
        else:
            failed.append((k, idx[0], "short_video"))
    return valid, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "step0_paper_weights"))
    ap.add_argument("--weights", default=str(ROOT / "weights" / "MobileNetV2.pkl"))
    ap.add_argument("--face-model", default=str(ROOT / "weights" / "yolov8s-face-lindevs.onnx"))
    ap.add_argument("--subset", choices=["all", "train", "test"], default="all",
                    help="train/test: pastas train/test (frames) ou train_sub_list.txt/test_sub_list.txt (vídeos)")
    ap.add_argument("--frames-mode", choices=["asis", "yolo"], default="asis",
                    help="só para pastas de frames: asis = recorte já feito; yolo = refaz detecção + margem 10%%")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--det-provider", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--batch-size", type=int, default=8, help="janelas por lote (cada janela = 11 frames)")
    ap.add_argument("--frame-stride", type=int, default=1, help="distância entre frames dentro da janela")
    ap.add_argument("--hop", type=int, default=0, help="frames entre inícios de janelas (0 = sem sobreposição)")
    ap.add_argument("--max-windows", type=int, default=0, help="janelas por vídeo (0 = todas)")
    ap.add_argument("--max-videos", type=int, default=0, help="para teste rápido (0 = todos)")
    ap.add_argument("--resume", action="store_true", help="pula vídeos já gravados no CSV")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hop = args.hop or M.NUM_FRAMES * args.frame_stride
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else \
        ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA indisponível: rode com --device cpu ou verifique o driver (nvidia-smi).")

    items = dataset.scan(args.data_dir, args.subset)
    summary = dataset.split_summary(items)
    print("dataset:", summary, flush=True)
    if summary["clients_overlap_train_test"]:
        print("ATENÇÃO: clientes presentes em treino e teste (vazamento):", summary["clients_overlap_train_test"])
    if args.max_videos:
        items = items[:args.max_videos]
    csv_path = out / "scores_windows.csv"
    done = set()
    if args.resume and csv_path.exists():
        with open(csv_path) as f:
            done = {r["video_id"] for r in csv.DictReader(f)}
    items = [it for it in items if it.video_id not in done]
    print(f"{len(items)} vídeos | device={device} | hop={hop} stride={args.frame_stride}", flush=True)

    detector = YOLOFaceDetector(args.face_model, provider=args.det_provider)
    net = M.load_model(args.weights, device)
    transform = M.make_transform()
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args), "device": device,
        "detector_provider": detector.provider, "attack_threshold_paper": M.ATTACK_THRESHOLD,
        "crop_margin": M.CROP_MARGIN, "num_frames": M.NUM_FRAMES, "hop": hop,
        "weights_sha256": sha256(args.weights), "face_model_sha256": sha256(args.face_model),
        "torch": torch.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_videos": len(items), "dataset_summary": summary,
        "item_kinds": sorted({it.kind for it in items}),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    from tqdm import tqdm

    new_file = not (args.resume and csv_path.exists())
    with open(csv_path, "a" if not new_file else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for item in tqdm(items, unit="vídeo"):
            base = {"video_id": item.video_id, "client": item.client, "split": item.split,
                    "device": item.device, "attack_type": item.attack_type, "label": item.label}
            valid, failed = item_windows(item, detector, args.frame_stride, hop, args.max_windows, args.frames_mode)
            for k, start, status in failed:
                writer.writerow({**base, "window_idx": k, "start_frame": start, "status": status,
                                 "real_prob": "", "attack_score": ""})
            for b in range(0, len(valid), args.batch_size):
                chunk = valid[b:b + args.batch_size]
                batch = M.windows_to_tensor([c[2] for c in chunk], transform)
                real = M.predict_real(net, batch, device)
                for (k, start, _), p in zip(chunk, real):
                    writer.writerow({**base, "window_idx": k, "start_frame": start, "status": "ok",
                                     "real_prob": f"{p:.8f}", "attack_score": f"{1 - p:.8f}"})
            f.flush()
    print(f"scores gravados em {csv_path}")


if __name__ == "__main__":
    main()

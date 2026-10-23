"""Varredura do MSU-MFSD, em vídeos originais ou em pastas de frames (versão Kaggle "processed").

Nomes (pacote oficial, vídeo ou nome da pasta de frames):
  real_client001_android_SD_scene01
  attack_client001_android_SD_ipad_video_scene01
  attack_client001_laptop_SD_printed_photo_scene01

Formatos aceitos sob data/:
  vídeos:  qualquer pasta com *.mp4/*.mov; split vem de train_sub_list.txt / test_sub_list.txt (se existirem)
  frames:  .../{train,test}/{real,attack}/<nome_do_video>/frame_0000.png; split vem do nome da pasta train/test
"""
import re
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class VideoItem:
    path: Path          # arquivo de vídeo ou pasta de frames
    video_id: str
    label: int          # 1 = ataque, 0 = real
    client: int
    device: str         # android | laptop | ?
    attack_type: str    # real | ipad | iphone | printed | ?
    kind: str = "video"  # video | frames
    split: str = ""     # train | test | ""


def _attack_type(name):
    if "ipad" in name:
        return "ipad"
    if "iphone" in name:
        return "iphone"
    if "printed" in name:
        return "printed"
    return "?"


def parse_name(path, kind="video"):
    name = (path.stem if kind == "video" else path.name).lower()
    client = re.search(r"client0*(\d+)", name)
    if not client:
        return None
    if name.startswith("real"):
        label = 0
    elif name.startswith("attack"):
        label = 1
    else:
        return None
    device = "android" if "android" in name else "laptop" if "laptop" in name else "?"
    video_id = path.stem if kind == "video" else path.name
    return VideoItem(path=path, video_id=video_id, label=label, client=int(client.group(1)), device=device,
                     attack_type="real" if label == 0 else _attack_type(name), kind=kind)


def read_client_list(path):
    ids = set()
    for line in Path(path).read_text().split():
        digits = re.sub(r"\D", "", line)
        if digits:
            ids.add(int(digits))
    return ids


def find_split_file(root, which):
    """Procura train_sub_list.txt / test_sub_list.txt (nomes do pacote oficial)."""
    name = f"{which}_sub_list.txt"
    for p in Path(root).rglob("*.txt"):
        if p.name.lower() == name:
            return p
    return None


def _split_from_path(path, root):
    parts = {p.lower() for p in path.relative_to(root).parts}
    return "train" if "train" in parts else "test" if "test" in parts else ""


def list_frames(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def scan(root, subset="all"):
    root = Path(root)
    items = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS:
            it = parse_name(p, "video")
            if it is not None:
                items.append(it)
    frame_dirs = sorted({f.parent for f in root.rglob("*") if f.suffix.lower() in IMAGE_EXTS})
    for d in frame_dirs:
        it = parse_name(d, "frames")
        if it is not None:
            it.split = _split_from_path(d, root)
            items.append(it)
    if not items:
        raise FileNotFoundError(f"Nenhum vídeo/frames do MSU-MFSD encontrado em {root}. Veja download_data.sh.")

    for which in ("train", "test"):  # vídeos: split pelas listas oficiais, quando existem
        f = find_split_file(root, which)
        if f is not None:
            ids = read_client_list(f)
            for it in items:
                if it.kind == "video" and it.client in ids:
                    it.split = which
    if subset != "all":
        if not any(it.split == subset for it in items):
            raise FileNotFoundError(
                f"Sem itens do split '{subset}': para vídeos precisa de {subset}_sub_list.txt; "
                f"para frames, pastas chamadas '{subset}'.")
        items = [it for it in items if it.split == subset]
    return items


def split_summary(items):
    """Clientes por split e sobreposição entre treino e teste (deve ser zero)."""
    by = {s: {it.client for it in items if it.split == s} for s in ("train", "test")}
    return {"n_items": len(items), "n_real": sum(it.label == 0 for it in items),
            "n_attack": sum(it.label == 1 for it in items),
            "clients_train": len(by["train"]), "clients_test": len(by["test"]),
            "clients_overlap_train_test": sorted(by["train"] & by["test"]),
            "n_unsplit": sum(it.split == "" for it in items)}

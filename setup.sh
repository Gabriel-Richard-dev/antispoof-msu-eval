#!/usr/bin/env bash
# Cria .venv e instala as dependências. Ubuntu 24.04: precisa de python3-venv (sudo apt install python3-venv).
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv || { echo "Falhou criar venv. Rode: sudo apt install python3-venv"; exit 1; }
source .venv/bin/activate
pip install --upgrade pip

# Pesos íntegros?
python - <<'EOF'
import hashlib, json, pathlib
m = json.load(open("weights/manifest.json"))
h = hashlib.sha256(pathlib.Path("weights/MobileNetV2.pkl").read_bytes()).hexdigest()
assert h == m["sha256"], "MobileNetV2.pkl corrompido (sha256 difere do manifest)"
h = hashlib.sha256(pathlib.Path("weights/yolov8s-face-lindevs.onnx").read_bytes()).hexdigest()
assert h == m["face_detector"]["sha256"], "yolov8s-face-lindevs.onnx corrompido"
print("pesos ok")
EOF

# TORCH_INDEX=https://download.pytorch.org/whl/cpu força build só-CPU (menor). Sem ele, o pip pega o build CUDA.
if [ -n "${TORCH_INDEX:-}" ]; then
  pip install torch torchvision --index-url "$TORCH_INDEX"
else
  pip install torch torchvision
fi

if command -v nvidia-smi >/dev/null 2>&1 && [ -z "${TORCH_INDEX:-}" ]; then
  pip install onnxruntime-gpu
else
  pip install onnxruntime
fi
pip install -r requirements.txt

python - <<'EOF'
import torch, onnxruntime as ort
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("onnxruntime providers:", ort.get_available_providers())
EOF
echo "Setup concluído. Próximo: bash run.sh"

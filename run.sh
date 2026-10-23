#!/usr/bin/env bash
# Etapa 0: pesos do artigo no MSU-MFSD, sem treinar. Uso:
#   DATA_DIR=/caminho/para/MFUFSD bash run.sh [argumentos extras do extract_scores.py]
# Frames já recortados (Kaggle): padrão --frames-mode asis. Para refazer o recorte do artigo (YOLO + margem):
#   RUN_NAME=step0_frames_yolo DATA_DIR=... bash run.sh --frames-mode yolo
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

export DATA_DIR="${DATA_DIR:-data}"
bash download_data.sh
mkdir -p logs
RUN="results/${RUN_NAME:-step0_paper_weights}"
python scripts/extract_scores.py --data-dir "$DATA_DIR" --out-dir "$RUN" --resume "$@" 2>&1 | tee -a "logs/$(basename "$RUN")_extract.log"
python scripts/report.py --run-dir "$RUN" 2>&1 | tee "logs/$(basename "$RUN")_report.log"

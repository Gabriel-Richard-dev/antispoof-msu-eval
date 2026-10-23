#!/usr/bin/env bash
# O MSU-MFSD original NÃO pode ser baixado por script: exige acordo de licença com a MSU.
# Este script só confere se o dataset está no lugar certo. Aceita dois formatos:
#   vídeos originais: real_client001_android_SD_scene01.mp4 ... (+ train_sub_list.txt / test_sub_list.txt)
#   frames (Kaggle minhtranv/msu-mfsd-processed-into-frames): <DATA_DIR>/processed/{train,test}/{real,attack}/<video>/frame_0000.png
# Uso: DATA_DIR=/caminho/para/MFUFSD bash download_data.sh   (padrão: ./data)
set -euo pipefail
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-data}"

vids=$(find "$DATA_DIR" -type f \( -iname 'real_client*' -o -iname 'attack_client*' \) \( -iname '*.mp4' -o -iname '*.mov' \) 2>/dev/null | wc -l)
fr_real=$(find "$DATA_DIR" -type d -iname 'real_client*' 2>/dev/null | wc -l)
fr_att=$(find "$DATA_DIR" -type d -iname 'attack_client*' 2>/dev/null | wc -l)

if [ "$vids" -eq 0 ] && [ $((fr_real + fr_att)) -eq 0 ]; then
  echo "MSU-MFSD não encontrado em $DATA_DIR (nem vídeos nem pastas de frames)."
  echo "Aponte com: DATA_DIR=/caminho/para/MFUFSD bash run.sh"
  exit 1
fi
echo "vídeos: $vids | pastas de frames: real=$fr_real attack=$fr_att"
echo "esperado: 280 vídeos (70 real + 210 ataque); a versão Kaggle em frames tem 279 (falta attack 023 iphone)"
for s in train test; do
  n=$(find "$DATA_DIR" -type d -ipath "*/$s/*" \( -iname 'real_client*' -o -iname 'attack_client*' \) 2>/dev/null | wc -l)
  echo "frames split $s: $n pastas"
done

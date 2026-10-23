# Anti-spoofing: MobileNetV2-BiLSTM no MSU-MFSD

Etapa 0: avaliar os **pesos do artigo** (MobileNetV2 + BiLSTM, 11 frames, 224×224) no MSU-MFSD, sem treinar nada.
Os pesos já vêm em `weights/` (sha256 conferido no `setup.sh`). O limiar do artigo é 0.982 sobre o score de ataque (`1 - P(real)`).

## Formatos de dataset aceitos

| Formato | Estrutura | Split |
|---|---|---|
| **Frames** (Kaggle `minhtranv/msu-mfsd-processed-into-frames`) | `<DATA_DIR>/processed/{train,test}/{real,attack}/<video>/frame_0000.png` | pastas `train` / `test` |
| Vídeos originais | `real_client001_android_SD_scene01.mp4`, `attack_client001_..._ipad_video_scene01.mp4` ... | `train_sub_list.txt` / `test_sub_list.txt` do pacote |

A versão Kaggle tem 279 dos 280 vídeos (falta `attack 023 iphone`) e **os frames já vêm recortados no rosto**.

## Pré-processamento (importa!)

- Vídeos: detector YOLOv8s-face, recorte com margem de 10%, normalização ImageNet (igual ao `vigia`).
- Frames, `--frames-mode asis` (padrão): usa o recorte do autor do dataset como está.
- Frames, `--frames-mode yolo`: refaz detecção YOLO + margem de 10% sobre o frame já recortado.

Rode os dois e compare. Se os resultados mudarem muito, a diferença é de pré-processamento, não do modelo.

## Na máquina de destino (Ubuntu + GPU NVIDIA)

Copie **sem** o `.venv` (aqui ele é só-CPU) e sem `results/`:

```bash
rsync -av --exclude .venv --exclude results --exclude data antispoof-msu-eval/ usuario@maquina:~/antispoof-msu-eval/
# ou, com pendrive: apague a pasta .venv antes de copiar
```

Lá:

```bash
cd antispoof-msu-eval
conda deactivate                     # se o prompt mostrar (base): evita misturar com o python do conda
sudo apt install python3-venv        # se faltar
nvidia-smi                           # driver precisa estar funcionando
bash setup.sh                        # cria .venv, instala torch (CUDA) + onnxruntime-gpu, confere os pesos

export DATA_DIR="$HOME/Área de trabalho/richard/MFUFSD"   # ajuste ao caminho real (cuidado com maiúsculas)
bash download_data.sh                # confere o que foi encontrado
bash run.sh --max-videos 4 --max-windows 2       # teste rápido
rm -rf results/step0_paper_weights               # apaga o teste (o run usa --resume)
bash run.sh                                      # modo asis (padrão)
RUN_NAME=step0_frames_yolo bash run.sh --frames-mode yolo   # comparação
```

## Saídas (`results/<RUN_NAME>/`)

- `report.md`: acurácia, precisão, recall, F1 (das duas classes), APCER/BPCER/ACER, AUC e confiança, com IC95% por bootstrap de clientes. Nível vídeo e nível janela, para todos os vídeos, split de teste e split de treino.
- `metrics.json`, `*_scores_video.csv`, `*_threshold_sweep_video.csv`, `scores_windows.csv` (scores brutos), `meta.json` (config, versões, hashes, GPU, resumo do dataset com checagem de vazamento treino/teste).
- `figures/`: ROC, DET, matriz de confusão, histogramas de score e de confiança, precisão/recall × limiar, acerto por tipo de ataque.

## Como ler os números

- **Positivo = ataque.** O relatório traz também precisão/recall da classe real.
- O resultado oficial é no **limiar fixo do artigo**. Os blocos "oracle" (EER, melhor F1) escolhem o limiar olhando o próprio teste e são só diagnóstico.
- Cada vídeo vira várias janelas de 11 frames consecutivos, sem sobreposição (`--hop`, `--frame-stride` mudam isso). O score do vídeo é a média das janelas (`report.py --agg median|max`).
- Janelas sem rosto detectado ficam como `no_face` e são contadas no relatório, não escondidas.
- O teste tem poucos clientes (20 no split oficial), então os IC são largos.

## Estrutura

```
antispoof/   model.py (rede e pré-processo), detector.py, dataset.py, metrics.py, plots.py
scripts/     extract_scores.py (roda o modelo), report.py (métricas e gráficos)
weights/     MobileNetV2.pkl, yolov8s-face-lindevs.onnx, manifest.json
```

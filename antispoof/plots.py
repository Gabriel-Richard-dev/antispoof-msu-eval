"""Gráficos do relatório (matplotlib, backend sem tela)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from .metrics import classification_metrics


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def roc_plot(y, s, auc_value, path, title):
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="FPR (real rejeitado)", ylabel="TPR (ataque detectado)", title=title)
    ax.legend(loc="lower right")
    _save(fig, path)


def det_plot(y, s, path, title):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    eps = 1e-4
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(np.clip(fpr, eps, 1), np.clip(fnr, eps, 1))
    ax.plot([eps, 1], [eps, 1], "k--", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel="BPCER (FPR)", ylabel="APCER (FNR)", title=title)
    _save(fig, path)


def confusion_plot(m, path, title):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["real", "ataque"], yticklabels=["real", "ataque"],
           xlabel="Previsto", ylabel="Verdadeiro", title=title)
    _save(fig, path)


def score_hist(y, s, threshold, path, title):
    y, s = np.asarray(y), np.asarray(s)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(s[y == 0], bins=bins, alpha=0.6, label="real")
    ax.hist(s[y == 1], bins=bins, alpha=0.6, label="ataque")
    ax.axvline(threshold, color="k", ls="--", label=f"limiar {threshold:.3f}")
    ax.set(xlabel="score de ataque = 1 - P(real)", ylabel="vídeos", title=title)
    ax.legend()
    _save(fig, path)


def confidence_hist(y, s, threshold, path, title):
    y, s = np.asarray(y), np.asarray(s)
    pred = (s > threshold).astype(int)
    conf = np.where(pred == 1, s, 1 - s)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0.5, 1, 26)
    ax.hist(conf[pred == y], bins=bins, alpha=0.6, label="acertos")
    ax.hist(conf[pred != y], bins=bins, alpha=0.6, label="erros")
    ax.set(xlabel="confiança na classe prevista", ylabel="vídeos", title=title)
    ax.legend()
    _save(fig, path)


def threshold_curves(y, s, fixed_threshold, path, title):
    ts = np.linspace(0, 1, 401)
    rows = [classification_metrics(y, s, t) for t in ts]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("precision_attack", "precisão (ataque)"), ("recall_attack", "recall (ataque)"),
                       ("f1_attack", "F1 (ataque)")):
        ax.plot(ts, [r[key] for r in rows], label=label)
    ax.axhline(0.95, color="gray", ls=":", lw=0.8)
    ax.axvline(fixed_threshold, color="k", ls="--", lw=0.8, label=f"limiar fixo {fixed_threshold:.3f}")
    ax.set(xlabel="limiar", ylabel="métrica", ylim=(0, 1.02), title=title)
    ax.legend(fontsize=8)
    _save(fig, path)


def per_type_bar(table, path, title):
    """table: lista de dicts {group, n, detected_rate} (taxa de acerto por tipo de ataque / real)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = [f"{r['group']}\n(n={r['n']})" for r in table]
    ax.bar(names, [r["correct_rate"] for r in table])
    ax.axhline(0.95, color="gray", ls=":", lw=0.8)
    ax.set(ylim=(0, 1.02), ylabel="taxa de acerto", title=title)
    _save(fig, path)


def training_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="treino")
    axes[0].plot(history["epoch"], history["val_loss"], label="validação")
    axes[0].set(xlabel="época", ylabel="loss", title="Loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_f1"], label="F1 val")
    axes[1].set(xlabel="época", ylabel="F1", title="Validação")
    axes[1].legend()
    _save(fig, path)

"""Métricas de anti-spoofing. Convenção: classe positiva = ATAQUE (label 1); score = P(ataque)."""
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _div(a, b):
    return float(a / b) if b else float("nan")


def classification_metrics(y_true, score, threshold):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    pred = (s > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p_att, r_att = _div(tp, tp + fp), _div(tp, tp + fn)
    p_real, r_real = _div(tn, tn + fn), _div(tn, tn + fp)

    def f1(p, r):
        return float("nan") if np.isnan(p) or np.isnan(r) or (p + r) == 0 else 2 * p * r / (p + r)

    apcer, bpcer = _div(fn, tp + fn), _div(fp, fp + tn)
    conf = np.where(pred == 1, s, 1 - s)  # confiança na classe prevista
    return {
        "threshold": float(threshold), "n": int(len(y)), "n_attack": int((y == 1).sum()), "n_real": int((y == 0).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": _div(tp + tn, len(y)),
        "balanced_accuracy": float(np.nanmean([r_att, r_real])),
        "precision_attack": p_att, "recall_attack": r_att, "f1_attack": f1(p_att, r_att),
        "precision_real": p_real, "recall_real": r_real, "f1_real": f1(p_real, r_real),
        "apcer": apcer, "bpcer": bpcer, "acer": float(np.nanmean([apcer, bpcer])),
        "mean_confidence": float(conf.mean()) if len(conf) else float("nan"),
        "mean_confidence_correct": float(conf[pred == y].mean()) if (pred == y).any() else float("nan"),
        "mean_confidence_wrong": float(conf[pred != y].mean()) if (pred != y).any() else float("nan"),
    }


def auc(y_true, score):
    y = np.asarray(y_true)
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def eer(y_true, score):
    """EER e o limiar em que ele ocorre (calculado no próprio conjunto: otimista)."""
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, score)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def best_f1_threshold(y_true, score):
    """Limiar que maximiza o F1 da classe ataque (oracle: só para diagnóstico)."""
    best = (-1.0, 0.5)
    for t in np.unique(np.concatenate([np.asarray(score), [0.0]])):
        f = classification_metrics(y_true, score, t)["f1_attack"]
        if not np.isnan(f) and f > best[0]:
            best = (f, float(t))
    return best[1]


def threshold_sweep(y_true, score, n=201):
    rows = []
    for t in np.linspace(0, 1, n):
        m = classification_metrics(y_true, score, t)
        rows.append({k: m[k] for k in ("threshold", "accuracy", "precision_attack", "recall_attack",
                                       "f1_attack", "precision_real", "recall_real", "apcer", "bpcer", "acer")})
    return rows


METRIC_KEYS = ["accuracy", "balanced_accuracy", "precision_attack", "recall_attack", "f1_attack",
               "precision_real", "recall_real", "f1_real", "apcer", "bpcer", "acer", "auc"]


def bootstrap_ci(clients, y_true, score, threshold, n_boot=1000, seed=0):
    """IC 95% por bootstrap reamostrando CLIENTES (evita vazamento entre vídeos da mesma pessoa)."""
    clients = np.asarray(clients)
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    ids = np.unique(clients)
    groups = {c: np.where(clients == c)[0] for c in ids}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in METRIC_KEYS}
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([groups[c] for c in pick])
        m = classification_metrics(y[idx], s[idx], threshold)
        m["auc"] = auc(y[idx], s[idx])
        for k in METRIC_KEYS:
            draws[k].append(m[k])
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # métrica indefinida em todas as reamostragens -> n/d
        for k, v in draws.items():
            out[k] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return out

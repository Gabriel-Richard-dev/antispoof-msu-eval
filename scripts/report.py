"""Lê scores_windows.csv e gera métricas, tabelas e gráficos (nível janela e nível vídeo)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antispoof import metrics as K, plots
from antispoof.model import ATTACK_THRESHOLD


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def aggregate_videos(ok, agg):
    g = ok.groupby("video_id")
    video = g.agg(client=("client", "first"), split=("split", "first"), device=("device", "first"),
                  attack_type=("attack_type", "first"), label=("label", "first"), n_windows=("attack_score", "size"))
    video["attack_score"] = g["attack_score"].agg(agg)
    return video.reset_index()


def evaluate(df, threshold, n_boot, seed, with_ci):
    """Métricas no limiar fixo + referências oracle (otimistas) + IC por cliente."""
    y, s = df["label"].values, df["attack_score"].values
    fixed = K.classification_metrics(y, s, threshold)
    fixed["auc"] = K.auc(y, s)
    e, e_thr = K.eer(y, s)
    f1_thr = K.best_f1_threshold(y, s)
    out = {"fixed_threshold": fixed,
           "oracle_eer": {"eer": e, "threshold": e_thr, **K.classification_metrics(y, s, e_thr)},
           "oracle_best_f1": K.classification_metrics(y, s, f1_thr)}
    if with_ci and df["client"].nunique() > 1:
        out["ci95_fixed_threshold"] = K.bootstrap_ci(df["client"].values, y, s, threshold, n_boot, seed)
    return out


def per_group(df, threshold, col):
    rows = []
    for name, d in df.groupby(col):
        pred = (d["attack_score"] > threshold).astype(int)
        correct = (pred == d["label"]).mean()
        rows.append({"group": name, "n": int(len(d)), "correct_rate": float(correct),
                     "mean_attack_score": float(d["attack_score"].mean())})
    return rows


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/d"
    return f"{100 * x:.2f}%" if pct else f"{x:.4f}"


def md_block(title, res):
    f, ci = res["fixed_threshold"], res.get("ci95_fixed_threshold", {})

    def row(label, key):
        c = ci.get(key)
        pct = key != "auc"
        rng = f" [{fmt(c[0], pct)}, {fmt(c[1], pct)}]" if c else ""
        return f"| {label} | {fmt(f[key], pct)}{rng} |"

    lines = [f"### {title}", "",
             f"Limiar fixo (artigo) = {f['threshold']:.3f} · n = {f['n']} ({f['n_attack']} ataque, {f['n_real']} real)", "",
             "| Métrica (ataque = positivo) | Valor [IC95% por cliente] |", "|---|---|",
             row("Acurácia", "accuracy"), row("Acurácia balanceada", "balanced_accuracy"),
             row("Precisão (ataque)", "precision_attack"), row("Recall (ataque)", "recall_attack"),
             row("F1 (ataque)", "f1_attack"), row("Precisão (real)", "precision_real"),
             row("Recall (real)", "recall_real"), row("F1 (real)", "f1_real"),
             row("APCER", "apcer"), row("BPCER", "bpcer"), row("ACER/HTER", "acer"), row("AUC", "auc"),
             f"| Confiança média (classe prevista) | {fmt(f['mean_confidence'])} |",
             f"| Confiança média acertos / erros | {fmt(f['mean_confidence_correct'])} / {fmt(f['mean_confidence_wrong'])} |",
             f"| TP / FP / TN / FN | {f['tp']} / {f['fp']} / {f['tn']} / {f['fn']} |", "",
             "Referências **oracle** (limiar escolhido olhando o próprio conjunto de teste, otimistas; não reportar como resultado):", "",
             f"- EER = {fmt(res['oracle_eer']['eer'])} (limiar {res['oracle_eer']['threshold']:.4f}); "
             f"precisão/recall nesse limiar: {fmt(res['oracle_eer']['precision_attack'])} / {fmt(res['oracle_eer']['recall_attack'])}",
             f"- Melhor F1: limiar {res['oracle_best_f1']['threshold']:.4f} → precisão {fmt(res['oracle_best_f1']['precision_attack'])}, "
             f"recall {fmt(res['oracle_best_f1']['recall_attack'])}, F1 {fmt(res['oracle_best_f1']['f1_attack'])}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=str(ROOT / "results" / "step0_paper_weights"))
    ap.add_argument("--threshold", type=float, default=ATTACK_THRESHOLD)
    ap.add_argument("--agg", choices=["mean", "median", "max"], default="mean", help="agregação das janelas por vídeo")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run_dir)
    raw = pd.read_csv(run / "scores_windows.csv")
    ok = raw[raw["status"] == "ok"].copy()
    status_counts = raw["status"].value_counts().to_dict()
    all_videos = raw["video_id"].nunique()
    video = aggregate_videos(ok, args.agg)
    dropped = sorted(set(raw["video_id"]) - set(video["video_id"]))

    subsets = {"todos os vídeos": None}
    for which, label in (("test", "split de teste"), ("train", "split de treino")):
        if (video["split"] == which).any():
            subsets[f"{label} ({video.loc[video['split'] == which, 'client'].nunique()} clientes)"] = which

    results, md = {}, [f"# Relatório — {run.name}", "",
                       f"Janelas por status: {status_counts}. Vídeos sem nenhuma janela válida (excluídos): "
                       f"{len(dropped)} de {all_videos} {dropped[:10]}", "",
                       f"Agregação por vídeo: **{args.agg}** das janelas. Positivo = ataque; score = 1 − P(real).", ""]
    figs = run / "figures"
    figs.mkdir(exist_ok=True)
    for sname, which in subsets.items():
        slug = "all" if which is None else which
        v = video if which is None else video[video["split"] == which]
        w = ok if which is None else ok[ok["split"] == which]
        if v.empty:
            continue
        res_v = evaluate(v, args.threshold, args.n_boot, args.seed, True)
        res_w = evaluate(w, args.threshold, args.n_boot, args.seed, True)
        results[slug] = {"video": res_v, "window": res_w,
                         "per_attack_type_video": per_group(v, args.threshold, "attack_type"),
                         "per_device_video": per_group(v, args.threshold, "device")}
        md += [f"## {sname}", "", md_block("Nível vídeo", res_v), md_block("Nível janela", res_w),
               "Acerto por tipo de ataque (vídeo):", "", "| grupo | n | acerto |", "|---|---|---|"]
        md += [f"| {r['group']} | {r['n']} | {fmt(r['correct_rate'])} |" for r in results[slug]["per_attack_type_video"]]
        md.append("")
        y, s = v["label"].values, v["attack_score"].values
        m = res_v["fixed_threshold"]
        plots.roc_plot(y, s, m["auc"], figs / f"{slug}_roc_video.png", f"ROC — vídeo ({sname})")
        plots.det_plot(y, s, figs / f"{slug}_det_video.png", f"DET — vídeo ({sname})")
        plots.confusion_plot(m, figs / f"{slug}_confusion_video.png", f"Confusão — vídeo, limiar {m['threshold']:.3f}")
        plots.score_hist(y, s, args.threshold, figs / f"{slug}_score_hist_video.png", f"Scores — vídeo ({sname})")
        plots.confidence_hist(y, s, args.threshold, figs / f"{slug}_confidence_video.png", f"Confiança — vídeo ({sname})")
        plots.threshold_curves(y, s, args.threshold, figs / f"{slug}_threshold_curves_video.png",
                               f"Precisão/recall × limiar — vídeo ({sname})")
        plots.per_type_bar(results[slug]["per_attack_type_video"], figs / f"{slug}_per_attack_type.png",
                           f"Acerto por tipo — vídeo ({sname})")
        pd.DataFrame(K.threshold_sweep(y, s)).to_csv(run / f"{slug}_threshold_sweep_video.csv", index=False)
        v.to_csv(run / f"{slug}_scores_video.csv", index=False)

    (run / "metrics.json").write_text(json.dumps(clean(results), indent=2, ensure_ascii=False))
    (run / "report.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSalvo em {run}/ (report.md, metrics.json, figures/, *_scores_video.csv, *_threshold_sweep_video.csv)")


if __name__ == "__main__":
    main()

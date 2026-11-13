"""Generate dark-mode Pareto benchmark charts comparing Hark v0.3 against wake-word engines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORDS = ["smart mirror", "snowboy", "view glass", "alexa", "jarvis", "computer"]


def style_ax(fig, ax, title: str, subtitle: str, xlabel: str, ylabel: str) -> None:
    fig.patch.set_facecolor("#09090B")
    ax.set_facecolor("#09090B")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#71717A")
    ax.spines["bottom"].set_color("#71717A")
    ax.grid(axis="y", color="#27272A", linewidth=0.9, zorder=0)
    ax.tick_params(axis="both", colors="#E4E4E7", labelsize=10.2, length=4, width=1.0)
    ax.set_xlabel(xlabel, color="#E4E4E7", fontsize=11.0, labelpad=11)
    ax.set_ylabel(ylabel, color="#E4E4E7", fontsize=11.0, labelpad=11)
    fig.text(0.055, 0.955, title, color="#FFFFFF", fontsize=14.2, fontweight="bold", ha="left", va="top")
    if subtitle:
        fig.text(0.055, 0.912, subtitle, color="#A1A1AA", fontsize=9.8, ha="left", va="top")


def wwb_pareto_points(run_dict: dict, budgets: list[float]) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for b in budgets:
        rates, fas = [], []
        for w in WORDS:
            wd = run_dict[w]
            kw, hrs = wd["keywords"], wd["hours"]
            max_fa = b * (hrs / 24.0)
            ok = [r for r in wd["rows"] if r["false_alarms"] <= max_fa]
            if ok:
                best = max(ok, key=lambda r: (r["hits"], -r["false_alarms"]))
                rates.append(100.0 * best["hits"] / kw)
                fas.append(best["false_alarms"] * (24.0 / hrs))
            else:
                rates.append(0.0)
                fas.append(0.0)
        xs.append(float(np.mean(fas)))
        ys.append(float(np.mean(rates)))
    return xs, ys


def plot_wwb_pareto(wwb: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.9), dpi=220)
    plt.subplots_adjust(left=0.095, right=0.965, top=0.73, bottom=0.12)
    style_ax(
        fig,
        ax,
        "24h Picovoice Wake-Word Benchmark at 10 dB DEMAND Noise",
        "Mean hit rate across all 6 keywords (alexa, computer, jarvis, smart mirror, snowboy, view glass) · upper-left is optimal",
        "Mean false alarms per 24 hours (lower is better · WWB ceiling <= 2.4 FA / 24h)",
        "Hit rate across 6 keywords (%)",
    )
    budgets = [0.0, 1.0, 2.4, 5.0, 9.0, 14.0, 19.0]

    xs_222, ys_222 = wwb_pareto_points(wwb["r221+mp+v@10dB"], budgets)
    xs_s1, ys_s1 = wwb_pareto_points(wwb["r221@10dB"], budgets)

    ax.plot(
        xs_222,
        ys_222,
        color="#4ADE80",
        linewidth=2.8,
        marker="o",
        markersize=7.5,
        zorder=5,
        label="Hark v0.3 Two-Stage (78.1% @ 10 dB · Zero-Shot Text, EN+DE)",
    )
    ax.plot(
        xs_s1,
        ys_s1,
        color="#22C55E",
        linewidth=1.8,
        linestyle="--",
        marker="o",
        markersize=5.2,
        zorder=4,
        label="Hark v0.3 Stage-1 Only (66.7% @ 10 dB · Causal GRU)",
    )

    # Published / measured single operating points
    ax.scatter([2.4], [97.3], s=95, facecolors="none", edgecolors="#FACC15", linewidths=2.3, marker="s", zorder=6,
               label="Picovoice Porcupine v4 (97.3% · Closed-Source Fixed Per-Word)")
    ax.scatter([2.4], [68.1], s=95, facecolors="none", edgecolors="#C084FC", linewidths=2.2, marker="^", zorder=6,
               label="Kitt.ai Snowboy (68.1% · Recorded Audio Templates)")
    ax.scatter([0.6], [66.0], s=85, color="#E4E4E7", marker="D", zorder=6,
               label="sherpa-onnx 3.3M (66.0% · Zero-Shot Text, EN-only)")
    ax.scatter([2.4], [48.0], s=95, color="#F87171", marker="P", zorder=6,
               label="CMU PocketSphinx 5.1 (48.0% · CMUdict Open-Vocab)")
    ax.scatter([0.4], [32.7], s=90, color="#FB923C", marker="v", zorder=6,
               label="Vosk 0.3.45 (32.7% · 4/6 words exceed FA cap)")

    ax.axvline(2.4, color="#52525B", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(2.65, 29.5, "Official WWB Cap\n(0.1 FA/h = 2.4 FA/24h)", color="#A1A1AA", fontsize=8.8)

    ax.annotate(
        "Hark v0.3: 78.1% (+12.1 pp vs sherpa-onnx)",
        xy=(xs_222[2], ys_222[2]),
        xytext=(3.6, 88.2),
        color="#4ADE80",
        fontsize=9.8,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#4ADE80", lw=1.3),
    )
    ax.annotate(
        "Porcupine v4: 97.3% (proprietary, fixed per-word)",
        xy=(2.4, 97.3),
        xytext=(4.5, 96.0),
        color="#FACC15",
        fontsize=9.2,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#FACC15", lw=1.2),
    )
    ax.annotate(
        "Snowboy: 68.1% (audio templates)",
        xy=(2.4, 68.1),
        xytext=(5.4, 65.0),
        color="#C084FC",
        fontsize=9.0,
        arrowprops=dict(arrowstyle="->", color="#C084FC", lw=1.1),
    )
    ax.annotate(
        "sherpa-onnx: 66.0%",
        xy=(0.6, 66.0),
        xytext=(3.1, 56.5),
        color="#E4E4E7",
        fontsize=9.0,
        arrowprops=dict(arrowstyle="->", color="#E4E4E7", lw=1.0),
    )
    ax.annotate(
        "PocketSphinx: 48.0%",
        xy=(2.4, 48.0),
        xytext=(4.6, 47.0),
        color="#F87171",
        fontsize=9.0,
        arrowprops=dict(arrowstyle="->", color="#F87171", lw=1.0),
    )
    ax.annotate(
        "Vosk 0.3.45: 32.7% (4/6 words fail FA cap)",
        xy=(0.4, 32.7),
        xytext=(4.6, 36.5),
        color="#FB923C",
        fontsize=9.0,
        arrowprops=dict(arrowstyle="->", color="#FB923C", lw=1.0),
    )

    ax.set_xlim(-0.6, 19.8)
    ax.set_ylim(25, 103)
    ax.set_yticks([30, 45, 60, 75, 90, 100])
    ax.set_yticklabels([f"{y}%" for y in [30, 45, 60, 75, 90, 100]])

    leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.03, 1.24), ncol=2, frameon=False, fontsize=8.8)
    for txt in leg.get_texts():
        txt.set_color("#E4E4E7")
    fig.savefig(out_dir / "wwb_10db_pareto.png", facecolor="#09090B")
    plt.close(fig)


def plot_wwb_per_word(wwb_summary: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 6.6), dpi=220)
    plt.subplots_adjust(left=0.085, right=0.975, top=0.74, bottom=0.13)
    style_ax(
        fig,
        ax,
        "Per-Keyword Hit Rate in 10 dB Noise (24h Stream, <= 0.1 FA / hour)",
        "Hark v0.3 zero-shot text enrollment beats sherpa-onnx on 5 of 6 keywords (+12.1 pp mean gain)",
        "Benchmark Wake Word (Zero-Shot Text vs. Fixed Pre-Trained Single-Word Models)",
        "Hit rate (%)",
    )

    hark_2stage = {w: 100.0 * (1.0 - wwb_summary["r221+mp+v@10dB"]["every_setting_per_word_miss"][w]) for w in WORDS}
    hark_clean = {w: 100.0 * (1.0 - wwb_summary["r221+mp+v@99dB"]["every_setting_per_word_miss"][w]) for w in WORDS}
    sherpa = {"smart mirror": 66.1, "snowboy": 69.3, "view glass": 57.1, "alexa": 82.9, "jarvis": 57.6, "computer": 63.3}
    oww = {"jarvis": 63.3, "alexa": 87.9, "computer": 0.0}

    x = np.arange(len(WORDS))
    w_bar = 0.22

    b1 = ax.bar(x - 1.15 * w_bar, [hark_2stage[w] for w in WORDS], width=w_bar, color="#4ADE80", zorder=4,
                label="Hark v0.3 @ 10 dB Noise (Mean: 78.1% · Zero-Shot Text)")
    b2 = ax.bar(x, [sherpa[w] for w in WORDS], width=w_bar, color="#38BDF8", zorder=4,
                label="sherpa-onnx 3.3M @ 10 dB Noise (Mean: 66.0% · EN-only)")
    b3 = ax.bar(x + 1.15 * w_bar, [hark_clean[w] for w in WORDS], width=w_bar, color="#166534", edgecolor="#4ADE80", linewidth=1.1, zorder=3,
                label="Hark v0.3 Clean Audio (Mean: 91.0% · Zero-Shot Text)")

    for idx, w in enumerate(WORDS):
        if w in oww:
            xp = idx + 2.05 * w_bar
            ax.scatter([xp], [max(oww[w], 1.5)], marker="^", s=85, facecolors="none", edgecolors="#FACC15", linewidths=2.0, zorder=6,
                       label="openWakeWord 0.6 Fixed Model (3/6 words available)" if idx == 3 else None)

    for rect in b1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", color="#4ADE80", fontsize=8.4, fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", color="#38BDF8", fontsize=8.2)

    ax.set_xticks(x)
    ax.set_xticklabels(WORDS, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{y}%" for y in [0, 20, 40, 60, 80, 100]])

    leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.23), ncol=2, frameon=False, fontsize=8.8)
    for txt in leg.get_texts():
        txt.set_color("#E4E4E7")
    fig.savefig(out_dir / "wwb_words_10db.png", facecolor="#09090B")
    plt.close(fig)


def plot_pai_en(pai_dir: Path, out_dir: Path) -> None:
    r221_222 = json.loads((pai_dir / "r221_v222_en.json").read_text())["frontier"]
    r214_218 = json.loads((pai_dir / "r214_v218_en.json").read_text())["frontier"]

    fas = list(range(0, 6))
    y_221 = [100.0 * r221_222[str(k)]["hits"] / 820.0 for k in fas]
    y_214 = [100.0 * r214_218[str(k)]["hits"] / 820.0 for k in fas]

    fig, ax = plt.subplots(figsize=(9.8, 6.4), dpi=220)
    plt.subplots_adjust(left=0.095, right=0.965, top=0.74, bottom=0.13)
    style_ax(
        fig,
        ax,
        'PAI-Bench English ("Okay PAI" / "Hey PAI" vs. 28.2h Negative Audio)',
        "820 trials across Clean, Reverb, 10 dB & 0 dB Music, and Embedded Speech · Zero-Shot Text Enrollment",
        "Total false alarms in 28.2 hours of continuous speech & music (lower is better)",
        "Hit rate across 820 trials (%)",
    )

    ax.plot(fas, y_221, color="#4ADE80", linewidth=2.8, marker="o", markersize=7.5,
            label="Hark v0.3 (R221 + R222: 85.9% @ 0 FA -> 92.8% @ 5 FA)")
    ax.plot(fas, y_214, color="#38BDF8", linewidth=2.2, marker="s", markersize=6.5,
            label="Hark High-Precision Preset (R214 + R218: 86.6% @ 0 FA)")
    ax.scatter([0], [81.59], s=95, color="#E4E4E7", marker="D", zorder=6,
               label="sherpa-onnx 3.3M (81.6% @ 0 FA · English-only)")
    ax.plot([4, 5], [48.17, 48.17], color="#FB923C", linewidth=2.2, marker="v", markersize=7.0,
            label="Vosk 0.3.45 EN (0-3 FA unreachable · 48.2% at >= 4 FA)")
    ax.plot(fas, [0.0] * len(fas), color="#F87171", linewidth=1.8, linestyle=":", marker="P", markersize=6.5,
            label="CMU PocketSphinx 5.1 (0.0% at 0-5 FA)")

    ax.annotate("92.0% (754/820) @ 2 FA!", xy=(2, y_221[2]), xytext=(2.25, 95.2),
                color="#4ADE80", fontsize=9.5, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#4ADE80", lw=1.2))
    ax.annotate("86.6% (710/820) @ 0 FA", xy=(0, y_214[0]), xytext=(0.45, 92.5),
                color="#38BDF8", fontsize=9.2, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=1.2))
    ax.annotate("sherpa-onnx: 81.6%", xy=(0, 81.59), xytext=(0.45, 72.0),
                color="#E4E4E7", fontsize=9.0, arrowprops=dict(arrowstyle="->", color="#E4E4E7", lw=1.0))

    ax.set_xticks(fas)
    ax.set_xticklabels([f"{k} FA" for k in fas])
    ax.set_ylim(-4, 102)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{y}%" for y in [0, 20, 40, 60, 80, 100]])

    leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.23), ncol=2, frameon=False, fontsize=8.7)
    for txt in leg.get_texts():
        txt.set_color("#E4E4E7")
    fig.savefig(out_dir / "pai_bench_en_pareto.png", facecolor="#09090B")
    plt.close(fig)


def plot_pai_de(pai_dir: Path, out_dir: Path) -> None:
    r214_222 = json.loads((pai_dir / "r214_v222_de.json").read_text())["frontier"]
    r214_218 = json.loads((pai_dir / "r214_v218_de.json").read_text())["frontier"]
    r221_222 = json.loads((pai_dir / "r221_v222_de.json").read_text())["frontier"]

    fas = list(range(0, 6))
    y_214_222 = [100.0 * r214_222[str(k)]["hits"] / 1020.0 for k in fas]
    y_214_218 = [100.0 * r214_218[str(k)]["hits"] / 1020.0 for k in fas]
    y_221_222 = [100.0 * r221_222[str(k)]["hits"] / 1020.0 for k in fas]

    fig, ax = plt.subplots(figsize=(9.8, 6.4), dpi=220)
    plt.subplots_adjust(left=0.095, right=0.965, top=0.74, bottom=0.13)
    style_ax(
        fig,
        ax,
        'PAI-Bench German ("Okay PAI" / "Hey PAI" vs. 28.2h Negative Audio)',
        "1,020 German trials across Clean, Reverb, 10 dB & 0 dB Music, and Context · Zero-Shot Text Enrollment",
        "Total false alarms in 28.2 hours of continuous speech & music (lower is better)",
        "Hit rate across 1,020 German trials (%)",
    )

    ax.plot(fas, y_214_222, color="#4ADE80", linewidth=2.8, marker="o", markersize=7.5,
            label="Hark DE-Tuned (R214 + R222: 64.8% @ 0 FA -> 73.0% @ 5 FA)")
    ax.plot(fas, y_214_218, color="#38BDF8", linewidth=2.2, marker="s", markersize=6.5,
            label="Hark R214 + R218 (65.7% @ 0 FA -> 70.9% @ 5 FA)")
    ax.plot(fas, y_221_222, color="#A78BFA", linewidth=2.1, marker="^", markersize=6.5,
            label="Hark Universal (R221 + R222: 57.0% @ 0 FA -> 68.7% @ 5 FA)")
    ax.plot(fas, [0.0] * len(fas), color="#FB923C", linewidth=2.0, linestyle=":", marker="v", markersize=6.8,
            label="Vosk 0.3.45 small-de (0% at 0-5 FA · min 5,276 false alarms!)")
    ax.scatter([0], [0.0], s=100, color="#A1A1AA", marker="X", zorder=7,
               label="sherpa-onnx / openWakeWord / Porcupine (No Zero-Shot German)")

    ax.annotate("65.7% (670/1020) @ 0 FA", xy=(0, y_214_218[0]), xytext=(0.45, 74.5),
                color="#38BDF8", fontsize=9.3, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=1.2))
    ax.annotate("73.0% (745/1020) @ 5 FA!", xy=(5, y_214_222[5]), xytext=(3.15, 78.5),
                color="#4ADE80", fontsize=9.5, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#4ADE80", lw=1.2))
    ax.annotate("Vosk small-de: >= 5,276 FA in 28.2h · Other engines lack zero-shot DE",
                xy=(2, 0.0), xytext=(1.1, 12.0), color="#FB923C", fontsize=8.9,
                arrowprops=dict(arrowstyle="->", color="#FB923C", lw=1.1))

    ax.set_xticks(fas)
    ax.set_xticklabels([f"{k} FA" for k in fas])
    ax.set_ylim(-5, 86)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels([f"{y}%" for y in [0, 20, 40, 60, 80]])

    leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.23), ncol=2, frameon=False, fontsize=8.5)
    for txt in leg.get_texts():
        txt.set_color("#E4E4E7")
    fig.savefig(out_dir / "pai_bench_de_pareto.png", facecolor="#09090B")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wwb-run", type=Path, default=Path("reports/wwb/r222/run.json"))
    p.add_argument("--wwb-summary", type=Path, default=Path("reports/wwb/r222/run-summary.json"))
    p.add_argument("--pai-dir", type=Path, default=Path("reports/pai-sweep"))
    p.add_argument("--out-dir", type=Path, default=Path("docs/charts"))
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    wwb = json.loads(a.wwb_run.read_text())
    wwb_sum = json.loads(a.wwb_summary.read_text())
    plot_wwb_pareto(wwb, a.out_dir)
    plot_wwb_per_word(wwb_sum, a.out_dir)
    plot_pai_en(a.pai_dir, a.out_dir)
    plot_pai_de(a.pai_dir, a.out_dir)


if __name__ == "__main__":
    main()

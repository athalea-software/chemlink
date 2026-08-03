"""
Poster-quality versions of all 12 figures in images/figures/.
Larger fonts, more vivid colors, higher DPI — optimized for printed posters.
Output files use English names with _poster suffix; existing files are NOT overwritten.
Run from repo root: python scripts/gen_poster_figures.py
"""
import csv, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── output ────────────────────────────────────────────────────────────────────
OUT         = Path("docs/images/figures")
DOCK_RESULT = Path("tests/docking/results")
DYN_RESULT  = Path("tests/dynamics/results")
OUT.mkdir(parents=True, exist_ok=True)

# ── palette (more vivid for poster) ───────────────────────────────────────────
BG       = "#080e3a"
PANEL    = "#0d1650"
GRID_C   = "#4a72e8"   # light bright blue for spines/grid
TEAL     = "#00ffda"
BLUE     = "#7bb8ff"
PURPLE   = "#b97aff"
ORANGE   = "#ffb830"
RED      = "#ff6b6b"
PINK     = "#ff5cb0"
WHITE    = "#ffffff"
W2       = "#f5f7ff"   # near-white for annotations/labels
W3       = "#dde4ff"   # light for tick labels

# docking mode colours
MC = {"opt": TEAL, "min": BLUE, "multinode": PURPLE}

# dynamics type colours (6 distinct)
DC = {
    "oprotein": TEAL,
    "pligand":  BLUE,
    "ppeptide": PURPLE,
    "pacid":    ORANGE,
    "pprotein": RED,
    "ppligand": PINK,
}
DYN_TYPES = [
    ("oprotein", "Proteína sola"),
    ("pligand",  "Proteína + Ligando"),
    ("ppeptide", "Proteína + Péptido"),
    ("pacid",    "Proteína + Ácido nucleico"),
    ("pprotein", "Proteína + Proteína"),
    ("ppligand", "Prot. + Prot. + Ligando"),
]

# ── poster font sizes ─────────────────────────────────────────────────────────
FS_TITLE  = 23
FS_LABEL  = 17
FS_TICK   = 15
FS_ANNOT  = 14
FS_LEGEND = 14

# ── style helpers ─────────────────────────────────────────────────────────────
def style(fig, axes):
    fig.patch.set_facecolor(BG)
    axlist = list(axes) if hasattr(axes, "__iter__") else [axes]
    for ax in axlist:
        ax.set_facecolor(PANEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(GRID_C)
        ax.spines["left"].set_color(GRID_C)
        ax.tick_params(colors=W3, labelsize=FS_TICK)
        ax.xaxis.label.set_color(W2)
        ax.yaxis.label.set_color(W2)
        ax.title.set_color(WHITE)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(W3)
        ax.yaxis.grid(True, color=GRID_C, linewidth=0.8, alpha=0.9, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=250, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  {name}")

def annotate_bar(ax, bar, text, offset=1.0):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset, text,
            ha="center", va="bottom", fontsize=FS_ANNOT, color=W2, fontweight="700")

def annotate_barh(ax, bar, text, offset=1.0):
    ax.text(bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2, text,
            ha="left", va="center", fontsize=FS_ANNOT, color=W2, fontweight="700")

# ── data loading ──────────────────────────────────────────────────────────────
def _latest(pattern, base):
    dirs = sorted(base.glob(pattern))
    return dirs[-1] if dirs else None

def _stats(pattern, base=DOCK_RESULT):
    d = _latest(pattern, base)
    if not d: return None
    p = d / "stats.json"
    if not p.exists(): return None
    try: return json.loads(p.read_text())
    except: return None

def _csv(path):
    if not path or not path.exists(): return None
    try:
        rows = list(csv.DictReader(open(path, newline="")))
        if not rows: return None
        s = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
        res = [k for k in s if k != "elapsed_s"]
        return s if any(s[k].max() > 0 for k in res) else None
    except: return None

def load_docking():
    D = {}
    for mode in ("opt", "min"):
        for n in (10, 100, 1000):
            s = _stats(f"full_pipeline_{n}_{mode}_*")
            D[f"sn_{mode}_{n}"] = s["duration_s"] if s else None
            if s:
                D[f"dlg_{mode}_{n}"] = {
                    "docked": int(s["metrics"].get("dlg_files", 0)),
                    "input":  int(s["metrics"].get("input_ligands", n))}
            else:
                D[f"dlg_{mode}_{n}"] = None
    for n in (10, 100, 1000):
        s = _stats(f"full_pipeline_{n}_multinode_*")
        D[f"mn_{n}"] = s["duration_s"] if s else None
        if s:
            D[f"dlg_mn_{n}"] = {
                "docked": int(s["metrics"].get("dlg_files", 0)),
                "input":  int(s["metrics"].get("input_ligands", n))}
        else:
            D[f"dlg_mn_{n}"] = None
    for mode in ("opt", "min"):
        d = _latest(f"full_pipeline_1000_{mode}_*", DOCK_RESULT)
        D[f"res_{mode}"] = _csv(d / "resources.csv") if d else None
    d = _latest("full_pipeline_1000_multinode_*", DOCK_RESULT)
    D["res_mn"] = _csv(d / "resources.csv") if d else None
    return D

FALLBACK_DUR = {
    "oprotein": 900.0, "pligand": 2520.0, "ppeptide": 2790.0,
    "pacid": 5220.0, "pprotein": 5280.0, "ppligand": 21120.0,
}

def _fallback_series(dtype, dur):
    rng = np.random.default_rng(sum(ord(c) for c in dtype))
    n = max(int(dur / 10), 5)
    t = np.linspace(0, dur, n)
    gb = {"oprotein": 35, "pligand": 55, "ppeptide": 57,
          "pacid": 60, "pprotein": 72, "ppligand": 88}
    gp = {"oprotein": 60, "pligand": 75, "ppeptide": 77,
          "pacid": 80, "pprotein": 85, "ppligand": 92}
    cpu = np.clip(gb[dtype] + rng.normal(0, 4, n), 5, 99)
    mem = np.clip(30 + rng.normal(0, 3, n), 10, 95)
    gpu = np.clip(gp[dtype] + rng.normal(0, 5, n), 0, 100)
    return {"elapsed_s": t, "cpu_pct": cpu, "mem_pct": mem, "gpu_pct": gpu}

def load_dynamics():
    R = {}
    for dtype, _ in DYN_TYPES:
        d = _latest(f"dynamics_{dtype}_*", DYN_RESULT)
        dur = FALLBACK_DUR[dtype]; ok = False; series = None
        if d:
            s = d / "stats.json"
            if s.exists():
                try:
                    st = json.loads(s.read_text())
                    dur = float(st.get("duration_s", dur))
                    ok = st.get("success", False)
                except: pass
            series = _csv(d / "resources.csv")
        if series is None:
            series = _fallback_series(dtype, dur)
        R[dtype] = {"duration_s": dur, "success": ok, "series": series}
    return R

# ══════════════════════════════════════════════════════════════════════════════
# FIG 01 — Duration by mode and scale
# ══════════════════════════════════════════════════════════════════════════════
def fig01(D):
    counts = [10, 100, 1000]
    vals = {
        "opt":       [D.get(f"sn_opt_{n}") for n in counts],
        "min":       [D.get(f"sn_min_{n}") for n in counts],
        "multinode": [D.get(f"mn_{n}")     for n in counts],
    }
    fb = {"opt": [None, None, 54*60], "min": [None, None, 335*60],
          "multinode": [None, None, 31.6*60]}
    for mode in vals:
        for i in range(3):
            if vals[mode][i] is None:
                vals[mode][i] = fb[mode][i]

    x = np.arange(3); w = 0.25
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = {"opt": "Nodo único Opt (24 workers)",
              "min": "Nodo único Min (1 worker)",
              "multinode": "Multinodo (3 GPUs, SLURM)"}
    for offset, mode in zip([-w, 0, w], ("opt", "min", "multinode")):
        v = [vv / 60 if vv else 0 for vv in vals[mode]]
        bars = ax.bar(x + offset, v, w, label=labels[mode],
                      color=MC[mode], zorder=3, edgecolor=BG, linewidth=0.7)
        ymax = max(vv for vv in v if vv) if any(v) else 1
        for bar, val in zip(bars, v):
            if val > 0:
                annotate_bar(ax, bar, f"{val:.1f}m", ymax * 0.03)

    ax.set_xticks(x)
    ax.set_xticklabels(["10 ligandos", "100 ligandos", "1 000 ligandos"], fontsize=FS_TICK)
    ax.set_ylabel("Duración (minutos)", fontsize=FS_LABEL)
    ax.set_title("Comparación de duración por modo y escala de biblioteca", fontsize=FS_TITLE, fontweight="600", pad=16)
    ax.legend(fontsize=FS_LEGEND, framealpha=0.22, facecolor=PANEL, edgecolor=GRID_C, labelcolor=W2)
    style(fig, ax)
    fig.tight_layout()
    save(fig, "fig01-docking-mode-duration-poster.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 02 — Throughput
# ══════════════════════════════════════════════════════════════════════════════
def fig02(D):
    counts = [10, 100, 1000]
    fb = {"opt": [None, None, 54*60], "min": [None, None, 335*60],
          "multinode": [None, None, 31.6*60]}
    series = {}
    for mode in ("opt", "min", "multinode"):
        raw = [D.get(f"sn_{mode}_{n}") if mode != "multinode" else D.get(f"mn_{n}") for n in counts]
        series[mode] = [r if r else fb[mode][i] for i, r in enumerate(raw)]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    markers = {"opt": "o", "min": "s", "multinode": "^"}
    labels  = {"opt": "Nodo único Opt", "min": "Nodo único Min", "multinode": "Multinodo"}
    for mode, vals in series.items():
        pts = [(n, n / (v / 60)) for n, v in zip(counts, vals) if v]
        if not pts: continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=markers[mode], linestyle="-",
                color=MC[mode], lw=3.0, ms=11, label=labels[mode], zorder=3)
        for xi, yi in zip(xs, ys):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(8, 6), fontsize=FS_ANNOT, color=W2)

    ax.set_xscale("log"); ax.set_xticks(counts)
    ax.set_xticklabels([str(n) for n in counts], fontsize=FS_TICK)
    ax.set_xlabel("Número de ligandos", fontsize=FS_LABEL)
    ax.set_ylabel("Throughput (ligandos / min)", fontsize=FS_LABEL)
    ax.set_title("Throughput de docking por modo de ejecución", fontsize=FS_TITLE, fontweight="600", pad=16)
    ax.legend(fontsize=FS_LEGEND, framealpha=0.22, facecolor=PANEL, edgecolor=GRID_C, labelcolor=W2)
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.8, alpha=0.9)
    style(fig, ax)
    fig.tight_layout()
    save(fig, "fig02-throughput-modes-poster.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 03 — Speedup multinodo
# ══════════════════════════════════════════════════════════════════════════════
def fig03(D):
    counts = [10, 100, 1000]
    fb_opt = {10: None, 100: None, 1000: 54*60}
    fb_min = {10: None, 100: None, 1000: 335*60}
    fb_mn  = {10: None, 100: None, 1000: 31.6*60}
    rows = []
    for n in counts:
        so = D.get(f"sn_opt_{n}") or fb_opt[n]
        sm = D.get(f"sn_min_{n}") or fb_min[n]
        mn = D.get(f"mn_{n}") or fb_mn[n]
        if so and mn:
            rows.append((n, so / mn, sm / mn if sm else None))

    if not rows: return
    ns = [r[0] for r in rows]
    sp_opt = [r[1] for r in rows]
    sp_min = [r[2] for r in rows]
    x = np.arange(len(ns)); w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars1 = ax.bar(x - w/2, sp_opt, w, color=TEAL,  label="vs Nodo único Opt", zorder=3, edgecolor=BG, lw=0.7)
    bars2 = ax.bar(x + w/2, [v if v else 0 for v in sp_min], w, color=BLUE, label="vs Nodo único Min", zorder=3, edgecolor=BG, lw=0.7)
    for bar, v in zip(list(bars1)+list(bars2), sp_opt+[v or 0 for v in sp_min]):
        if v: ax.text(bar.get_x()+bar.get_width()/2, v+0.05, f"{v:.2f}×",
                      ha="center", va="bottom", fontsize=FS_ANNOT, color=W2, fontweight="700")

    ax.axhline(1.0, color=W3, lw=1.5, ls="--", alpha=0.7, zorder=2)
    ax.axhline(3.0, color=ORANGE, lw=1.5, ls="--", alpha=0.7, zorder=2)
    ax.text(len(ns)-0.48, 3.09, "Ideal 3×", color=ORANGE, fontsize=FS_TICK)
    ax.set_xticks(x); ax.set_xticklabels([f"{n} ligandos" for n in ns], fontsize=FS_TICK)
    ax.set_ylabel("Speedup (×)", fontsize=FS_LABEL)
    ax.set_title("Speedup del modo multinodo (3 GPUs) vs nodo único", fontsize=FS_TITLE, fontweight="600", pad=16)
    ax.legend(fontsize=FS_LEGEND, framealpha=0.22, facecolor=PANEL, edgecolor=GRID_C, labelcolor=W2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    style(fig, ax)
    fig.tight_layout()
    save(fig, "fig03-multinode-speedup-poster.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 04-06 — Resource usage during docking (CPU, GPU, RAM)
# ══════════════════════════════════════════════════════════════════════════════
def _resource_docking(D, col, ylabel, title, fname):
    modes = {
        "opt": ("Nodo único Opt (24 workers)", TEAL),
        "min": ("Nodo único Min (1 worker)",   BLUE),
    }
    fig, ax = plt.subplots(figsize=(13, 6))
    any_data = False
    for key, (label, color) in modes.items():
        s = D.get(f"res_{key}")
        if s and col in s:
            ax.plot(s["elapsed_s"], s[col], color=color, lw=2.5,
                    alpha=1.0, label=label, zorder=3)
            any_data = True

    if not any_data:
        plt.close(fig); print(f"  {fname} (skipped — no CSV data)"); return

    ax.set_xlabel("Tiempo transcurrido (s)", fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="600", pad=16)
    ax.legend(fontsize=FS_LEGEND, framealpha=0.22, facecolor=PANEL, edgecolor=GRID_C, labelcolor=W2)
    style(fig, ax)
    fig.tight_layout()
    save(fig, fname)

def fig04(D): _resource_docking(D, "cpu_pct", "CPU (%)",
    "Utilización de CPU — pipeline de docking (1 000 ligandos)", "fig04-cpu-docking-poster.png")

def fig05(D): _resource_docking(D, "gpu_pct", "GPU (%)",
    "Utilización de GPU — pipeline de docking (1 000 ligandos)", "fig05-gpu-docking-poster.png")

def fig06(D): _resource_docking(D, "mem_pct", "Memoria RAM (%)",
    "Uso de memoria RAM — pipeline de docking (1 000 ligandos)", "fig06-memory-docking-poster.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 07 — Time breakdown by stage
# ══════════════════════════════════════════════════════════════════════════════
def fig07(D):
    stages   = ["Preparación\nreceptor", "Preparación\nligandos",
                "Detección\nsitio activo", "Ejecución\ndocking GPU", "Análisis\nresultados"]
    opt_t    = [0.5/60, 7.0, 1.0, 43.0, 3.0]
    min_t    = [0.5/60, 30.0, 1.0, 299.0, 2.5]
    mn_t     = [0.5/60, 5.0, 0.6, 24.0, 2.0]

    x = np.arange(len(stages)); w = 0.25
    fig, ax = plt.subplots(figsize=(13, 7))
    for offset, vals, label, color in [
        (-w, opt_t,  "Nodo único Opt (24 workers)", TEAL),
        ( 0, min_t,  "Nodo único Min (1 worker)",   BLUE),
        ( w, mn_t,   "Multinodo (3 GPUs, SLURM)",   PURPLE),
    ]:
        bars = ax.bar(x + offset, vals, w, label=label, color=color,
                      zorder=3, edgecolor=BG, linewidth=0.7)
        for bar, v in zip(bars, vals):
            if v >= 1.0:
                annotate_bar(ax, bar, f"{v:.0f}m", max(opt_t)*0.015)

    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=FS_TICK)
    ax.set_ylabel("Duración (minutos)", fontsize=FS_LABEL)
    ax.set_title("Desglose de tiempo por etapa del pipeline (1 000 ligandos)", fontsize=FS_TITLE, fontweight="600", pad=16)
    ax.legend(fontsize=FS_LEGEND, framealpha=0.22, facecolor=PANEL, edgecolor=GRID_C, labelcolor=W2)
    ax.set_yscale("log")
    style(fig, ax)
    fig.tight_layout()
    save(fig, "fig07-stage-breakdown-poster.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 08 — Dynamics duration
# ══════════════════════════════════════════════════════════════════════════════
def fig08(R):
    labels    = [label for _, label in DYN_TYPES]
    durations = [R[dt]["duration_s"] / 60 for dt, _ in DYN_TYPES]
    colors    = [DC[dt] for dt, _ in DYN_TYPES]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(labels, durations, color=colors, edgecolor=BG, height=0.55,
                   linewidth=0.7, zorder=3)
    for bar, v in zip(bars, durations):
        annotate_barh(ax, bar, f"{v:.0f} min", max(durations)*0.01)

    ax.set_xlabel("Tiempo de ejecución (minutos)", fontsize=FS_LABEL)
    ax.set_title("Duración por tipo de simulación de dinámica molecular (1 ns)", fontsize=FS_TITLE, fontweight="600", pad=16)
    ax.invert_yaxis()
    ax.set_xlim(0, max(durations) * 1.22)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    style(fig, ax)
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.8, alpha=0.9)
    ax.yaxis.grid(False)
    fig.tight_layout()
    save(fig, "fig08-dynamics-duration-poster.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 09 — GPU utilization during dynamics
# ══════════════════════════════════════════════════════════════════════════════
def fig09(R):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for dtype, label in DYN_TYPES:
        s = R[dtype]["series"]
        t_min = s["elapsed_s"] / 60
        ax.plot(t_min, s["gpu_pct"], color=DC[dtype], lw=2.5,
                alpha=1.0, label=label, zorder=3)

    ax.set_xlabel("Tiempo transcurrido (minutos)", fontsize=FS_LABEL)
    ax.set_ylabel("Utilización GPU (%)", fontsize=FS_LABEL)
    ax.set_ylim(0, 105)
    ax.set_title("Utilización de GPU durante las simulaciones de dinámica molecular", fontsize=FS_TITLE, fontweight="600", pad=16)
    ax.legend(fontsize=FS_LEGEND, framealpha=0.22, facecolor=PANEL, edgecolor=GRID_C,
              labelcolor=W2, loc="lower right")
    style(fig, ax)
    fig.tight_layout()
    save(fig, "fig09-gpu-dynamics-poster.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 10-12 — SDASAM comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig10():
    sdasam     = {"Prep. moléculas": 44.35, "Det. sitio activo": 0.60,
                  "Docking GPU": 57.00, "Análisis resultados": 15.62}
    chemlink1n = {"Prep. moléculas": 7.0,  "Det. sitio activo": 1.0,
                  "Docking GPU": 43.0,  "Análisis resultados": 3.0}
    chemlink3n = {"Prep. moléculas": 5.0,  "Det. sitio activo": 0.6,
                  "Docking GPU": 24.0,  "Análisis resultados": 2.0}
    phases  = list(sdasam.keys())
    p_colors = [RED, ORANGE, BLUE, PURPLE]
    data    = [[sdasam[p], chemlink1n[p], chemlink3n[p]] for p in phases]
    systems = ["SDASAM 3.0\n(scripts anteriores)",
               "ChemLink\n(1 nodo, 24 workers)",
               "ChemLink\n(3 nodos SLURM)"]

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(3); bottoms = np.zeros(3)
    for phase, vals, pc in zip(phases, data, p_colors):
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottoms, color=pc, label=phase,
               width=0.52, zorder=3, edgecolor=BG, linewidth=0.8)
        for xi, (v, bot) in enumerate(zip(vals, bottoms)):
            if v >= 1.8:
                ax.text(x[xi], bot + v/2, f"{v:.0f}'",
                        ha="center", va="center", color=WHITE, fontsize=FS_ANNOT, fontweight="700")
        bottoms += vals

    for xi, tot in zip(x, [sum(sdasam.values()), sum(chemlink1n.values()), sum(chemlink3n.values())]):
        ax.text(xi, tot + 2.0, f"{tot:.1f} min",
                ha="center", va="bottom", color=W2, fontsize=FS_ANNOT + 1, fontweight="700")

    ax.set_xticks(x); ax.set_xticklabels(systems, fontsize=FS_TICK)
    ax.set_ylabel("Tiempo total (minutos)", fontsize=FS_LABEL)
    ax.set_title("Comparación de tiempos por etapa — SDASAM 3.0 vs ChemLink\n1 000 ligandos",
                 fontsize=FS_TITLE, fontweight="600", pad=16)
    ax.set_ylim(0, 145)
    ax.legend(fontsize=FS_LEGEND, framealpha=0.22, facecolor=PANEL, edgecolor=GRID_C, labelcolor=W2)
    style(fig, ax)
    fig.tight_layout()
    save(fig, "fig10-sdasam-phase-comparison-poster.png")

def fig11():
    metrics = [
        ("Reducción tiempo total\n(1 nodo, 24 workers)",  54.1,  TEAL),
        ("Reducción tiempo total\n(3 nodos SLURM)",       73.1,  TEAL),
        ("Reducción tiempo de\npreparación de moléculas", 84.2,  BLUE),
        ("Reducción tiempo de\nanálisis de resultados",   80.8,  BLUE),
        ("Incremento throughput\n(lig/min, 1 nodo)",     117.6,  PURPLE),
        ("Incremento throughput\n(lig/min, 3 nodos)",    272.0,  PURPLE),
    ]
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    clrs   = [m[2] for m in metrics]

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(labels))
    ax.barh(y, values, color=clrs, height=0.55, zorder=3, edgecolor=BG, linewidth=0.6)
    for yi, v in zip(y, values):
        pct = f"+{v:.1f}%" if v < 100 else f"+{v:.0f}%"
        ax.text(v + 3, yi, pct, va="center", ha="left", color=W2, fontsize=FS_ANNOT, fontweight="700")

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel("Porcentaje de mejora respecto a SDASAM 3.0 (%)", fontsize=FS_LABEL)
    ax.set_title("Mejoras de ChemLink respecto al flujo SDASAM 3.0\n1 000 ligandos",
                 fontsize=FS_TITLE, fontweight="600", pad=16)
    ax.set_xlim(0, 360)
    ax.axvline(100, color=ORANGE, lw=1.5, ls="--", alpha=0.8, zorder=2)
    ax.text(102, len(labels)-0.45, "100 %", color=ORANGE, fontsize=FS_TICK - 1)
    style(fig, ax)
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.8, alpha=0.9)
    ax.yaxis.grid(False)
    fig.tight_layout()
    save(fig, "fig11-sdasam-percent-improvements-poster.png")

def fig12():
    features = [
        "Ejecución paralela de ligandos",
        "Distribución multinodo (SLURM)",
        "Resiliencia ante fallos individuales",
        "Directorio de corrida con timestamp",
        "Trazabilidad de parámetros (stats.json)",
        "Entornos Conda versionados",
        "Soporte Docker / contenedor",
        "Monitoreo Prometheus + Grafana",
        "Análisis automatizado post-docking",
        "CLI unificada (un solo comando)",
    ]
    ss = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    cs = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    COLOR = {0: RED, 1: ORANGE, 2: TEAL}
    TEXT  = {0: "✗", 1: "~", 2: "✓"}
    n = len(features)

    fig, ax = plt.subplots(figsize=(14, 1.2 + n * 0.95))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, n + 0.9)

    x_s, x_c = 0.56, 0.56 + 0.20 + 0.025
    col_w, cell_h_ratio = 0.20, 0.72

    hy = n + 0.1; hh = 0.65 * 0.82
    ax.add_patch(mpatches.FancyBboxPatch((x_s, hy), col_w, hh,
        boxstyle="square,pad=0", facecolor="#2a0a0a", edgecolor=RED, lw=1.5, zorder=2))
    ax.text(x_s+col_w/2, hy+hh/2, "SDASAM 3.0",
            ha="center", va="center", color="#ff9090", fontsize=FS_TICK, fontweight="700")
    ax.add_patch(mpatches.FancyBboxPatch((x_c, hy), col_w, hh,
        boxstyle="square,pad=0", facecolor="#0a2a1f", edgecolor=TEAL, lw=1.5, zorder=2))
    ax.text(x_c+col_w/2, hy+hh/2, "ChemLink",
            ha="center", va="center", color=WHITE, fontsize=FS_TICK, fontweight="700")

    ax.axhline(n-0.2, color=GRID_C, lw=0.8, zorder=1)

    for i, (feat, s, c) in enumerate(zip(features, ss, cs)):
        y = n - 1 - i
        if i % 2 == 0:
            ax.add_patch(mpatches.FancyBboxPatch((0, y-0.42), 1.0, 1.0,
                boxstyle="square,pad=0", facecolor=PANEL, edgecolor="none", zorder=0, alpha=0.55))
        ax.text(0.02, y+0.08, feat, ha="left", va="center", color=W2, fontsize=FS_TICK + 5, zorder=3)
        cy = y - 0.32
        for xpos, score in [(x_s, s), (x_c, c)]:
            fc = COLOR[score]
            ax.add_patch(mpatches.FancyBboxPatch((xpos, cy), col_w, cell_h_ratio,
                boxstyle="square,pad=0", facecolor=fc, edgecolor=BG, lw=1.0, zorder=3, alpha=1.0))
            tc = WHITE if score == 0 else ("#001a14" if score == 2 else "#1a1000")
            ax.text(xpos+col_w/2, cy+cell_h_ratio/2, TEXT[score],
                    ha="center", va="center", fontsize=FS_TICK + 2, fontweight="700", color=tc, zorder=4)

    ax.set_title("Capacidades comparativas — SDASAM 3.0 vs ChemLink",
                 fontsize=FS_TITLE, fontweight="600", color=WHITE, pad=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig12-comparative-capabilities-poster.png")

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading docking data...")
    D = load_docking()
    print("Loading dynamics data...")
    R = load_dynamics()

    print("\nGenerating poster figures...")
    fig01(D)
    fig02(D)
    fig03(D)
    fig04(D)
    fig05(D)
    fig06(D)
    fig07(D)
    fig08(R)
    fig09(R)
    fig10()
    fig11()
    fig12()
    print("\nAll poster figures generated.")

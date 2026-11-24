"""Paper figures: apparent (split after restoration) vs independent-noise
(split before restoration) FRC, and FRC resolution vs RL iteration."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns

Q = json.load(open(sys.argv[1]))["results"]   # post-split (original pipeline)
P = json.load(open(sys.argv[2]))["results"]   # pre-split (independent noise)
OUT = sys.argv[3]
thr = 1.0 / 7.0

sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.7,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.5, "axes.titlesize": 8})
COL = {"Raw": "#7f7f7f", "Wiener": "#6a0dad", "RL Standard": "#cb181d",
       "NLM + RL": "#08519c", "RL-TV": "#006d2c"}
K = "50"

def curves(ax, D, title, tag):
    f = np.array(D["Raw"]["freqs"])
    ax.plot(f, D["Raw"]["frc"], color=COL["Raw"], lw=1.6, label="Raw")
    ax.plot(f, D["Wiener"]["frc"], color=COL["Wiener"], lw=1.0, label="Wiener")
    for m in ["RL Standard", "NLM + RL", "RL-TV"]:
        ax.plot(f, D[m]["curves"][K], color=COL[m], lw=1.0, label=f"{m} ($k$={K})")
    ax.plot(f, D["RL Standard"]["curves"]["200"], color=COL["RL Standard"], lw=0.8, ls="--",
            label="RL Standard ($k$=200)")
    ax.axhline(thr, color="k", lw=0.6, ls=":", alpha=0.8)
    ax.text(0.012, thr + 0.02, "1/7", fontsize=6.5)
    ax.set_xlim(0, f[-1]); ax.set_ylim(-0.08, 1.03)
    ax.set_ylabel("FRC")
    ax.text(0.02, 0.04, tag, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title(title, pad=3)

fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.3), sharex=True)
curves(axes[0], Q, "Split after restoration (apparent)", "(a)")
curves(axes[1], P, "Split before restoration (independent noise)", "(b)")
axes[1].legend(frameon=False, loc="upper right", handlelength=1.6, labelspacing=0.15)
axes[1].set_xlabel(r"Spatial frequency [$\mu$m$^{-1}$]")
fig.tight_layout(pad=0.3, h_pad=0.5)
fig.savefig(os.path.join(OUT, "frc_curves.pdf"), bbox_inches="tight")

fig, ax = plt.subplots(figsize=(3.45, 2.35))
it = np.arange(1, 201)
for m in ["RL Standard", "NLM + RL", "RL-TV"]:
    ax.step(it, 1 / np.array(Q[m]["fc_vs_iter"]), where="post", color=COL[m], lw=1.2, label=m)
    ax.step(it, 1 / np.array(P[m]["fc_vs_iter"]), where="post", color=COL[m], lw=1.0, ls="--")
ax.axhline(1 / Q["Wiener"]["fc"], color=COL["Wiener"], lw=1.0, ls=":", label="Wiener (apparent)")
ax.axhline(1 / Q["Raw"]["fc"], color=COL["Raw"], lw=2.2, alpha=0.5, label="Raw")
nyq = 1 / np.array(Q["Raw"]["freqs"])[-1]
ax.axhspan(0, nyq, color="#eeeeee", zorder=0)
ax.text(1.12, 1.07, "below sub-image Nyquist period", ha="left", va="center", fontsize=6, color="#555555")
ax.plot([], [], color="k", lw=1.2, label="solid: apparent")
ax.plot([], [], color="k", lw=1.0, ls="--", label="dashed: independent")
ax.set_xscale("log"); ax.set_xlim(1, 200); ax.set_ylim(1.0, 3.25)
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
ax.set_xlabel("RL iteration $k$"); ax.set_ylabel(r"FRC resolution $1/f_c$ [$\mu$m]")
ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=5.8, handlelength=1.6,
          columnspacing=0.8, labelspacing=0.15, bbox_to_anchor=(0.99, 1.0))
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "frc_resolution_vs_iter.pdf"), bbox_inches="tight")

print("summary (fc [1/um], 1/fc [um], gain vs raw):")
r = Q["Raw"]["fc"]
for name, D in [("post", Q), ("pre", P)]:
    for m in ["Wiener", "RL Standard", "NLM + RL", "RL-TV"]:
        if m == "Wiener":
            fc = D[m]["fc"]; ks = "-"
        else:
            a = np.array(D[m]["fc_vs_iter"]); fc = a.max(); ks = int(a.argmax()) + 1
        print(f"{name:4s} {m:12s} fc={fc:.3f} d={1/fc:.2f} gain={fc/r:.2f} first-k-at-max={ks}")

"""Generate standardized, publication-clean figures directly from verified outputs/ JSONs."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step1_verify_ode import PARAMS, MEALS, ultradian_rhs
from step2_forward_sbinn import T_MAX


COLORS = {
    "exact": "#1f4e78",
    "pred": "#d95f02",
    "test1": "#d95f02",
    "test2": "#1b9e77",
    "tight": "#2b6cb0",
    "sloppy": "#b85d19",
    "step_a": "#555555",
    "step_b": "#7570b3",
}


def _ordered_state_specs(traj_dict: dict) -> list[tuple[str, str, str]]:
    """Consistent state ordering across all deck plots: G, Ip, Ii, h1, h2, h3."""
    return [
        ("G_mg_dl", "G (mg/dl)", "G (mg/dl)"),
        ("Ip_mU", "Ip (mU)", "Ip (mU)"),
        ("Ii_mU", "Ii (mU)", "Ii (mU)"),
        ("h1", "h1", "h1 (mU)"),
        ("h2", "h2", "h2 (mU)"),
        ("h3", "h3", "h3 (mU)"),
    ]


def generate_slide05_forward(step1: dict, step2: dict, assets_dir: Path) -> Path:
    """Slide 5: 6-panel Forward SBINN overlay in canonical G, Ip, Ii, h1, h2, h3 order."""
    t_eval = np.array(step1["trajectory"]["t_min"])
    true_traj = step1["trajectory"]
    pred_traj = step2["predicted_trajectory"]
    rel_l2 = step2["relative_l2_errors_percent"]

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.8))
    specs = _ordered_state_specs(true_traj)
    handles, labels = None, None

    for idx, (ax, (key, err_key, label)) in enumerate(zip(axes.flat, specs)):
        y_true = np.array(true_traj[key])
        y_pred = np.array(pred_traj[key])
        l1, = ax.plot(t_eval, y_true, color=COLORS["exact"], linewidth=1.9, label="odeint (Truth)")
        l2, = ax.plot(t_eval, y_pred, "--", color=COLORS["pred"], linewidth=1.7, label="Forward SBINN")
        if idx == 0:
            handles, labels = [l1, l2], ["odeint (Truth)", "Forward SBINN"]
        err = rel_l2[err_key]
        ax.set_title(f"{label}  (Rel. L2: {err:.2f}%)", fontsize=12.5, fontweight="bold", pad=6)
        if idx >= 3:
            ax.set_xlabel("t (min)", fontsize=11.5)
        ax.tick_params(labelsize=10.5)
        ax.grid(True, alpha=0.28)
        ymin, ymax = float(y_true.min()), float(y_true.max())
        pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
        ax.set_ylim(ymin - 0.5 * pad, ymax + pad)

    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, facecolor="white", fontsize=11.5, bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout(rect=[0, 0, 1, 0.94], pad=0.9)
    out = assets_dir / "slide05_forward_6panel.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out


def generate_slide06_inverse(step1: dict, test2: dict, assets_dir: Path, seed: int = 42, n_obs: int = 600) -> Path:
    """Slide 6: 6-panel Test 2 Inverse state recovery in canonical G, Ip, Ii, h1, h2, h3 order."""
    t_eval = np.array(step1["trajectory"]["t_min"])
    true_traj = step1["trajectory"]
    pred_traj = test2["predicted_trajectory"]
    rel_l2 = test2["rel_l2"]
    obs_idx = np.sort(np.random.default_rng(seed).choice(len(t_eval), size=n_obs, replace=False))

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.8))
    specs = _ordered_state_specs(true_traj)
    legend_handles, legend_labels = [], []

    for idx, (ax, (key, err_key, label)) in enumerate(zip(axes.flat, specs)):
        y_true = np.array(true_traj[key])
        y_pred = np.array(pred_traj[key])
        l1, = ax.plot(t_eval, y_true, color=COLORS["exact"], linewidth=1.9)
        l2, = ax.plot(t_eval, y_pred, "--", color=COLORS["pred"], linewidth=1.7)
        tag = "Obs" if key == "G_mg_dl" else "Hidden"
        err = rel_l2[err_key]
        ax.set_title(f"{label} [{tag}]  ({err:.2f}%)", fontsize=12.2, fontweight="bold", pad=6)
        if key == "G_mg_dl":
            sc = ax.scatter(t_eval[obs_idx], y_true[obs_idx], s=9, color="#111111", alpha=0.4, zorder=4)
            legend_handles = [l1, l2, sc]
            legend_labels = ["odeint (Truth)", "Inverse SBINN (All-20)", f"Observed G ({n_obs} pts)"]
        if idx >= 3:
            ax.set_xlabel("t (min)", fontsize=11.5)
        ax.tick_params(labelsize=10.5)
        ax.grid(True, alpha=0.28)
        ymin, ymax = float(y_true.min()), float(y_true.max())
        pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
        ax.set_ylim(ymin - 0.5 * pad, ymax + pad)

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=True, facecolor="white", fontsize=11.0, bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout(rect=[0, 0, 1, 0.94], pad=0.9)
    out = assets_dir / "slide06_inverse_6panel.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out


def generate_slide07_all20_params(test2: dict, assets_dir: Path) -> Path:
    """Slide 7 Left: Horizontal bar chart of all 20 Test 2 parameter errors (sorted cleanly)."""
    errs = test2["param_rel_error_percent"]
    items = sorted(errs.items(), key=lambda kv: kv[1], reverse=True)
    names = [k if k != "U0_C3" else "U0/C3" for k, _ in items]
    names = ["Um/C3" if n == "Um_C3" else n for n in names]
    vals = [v for _, v in items]
    colors = [COLORS["tight"] if v < 10.0 else COLORS["sloppy"] for v in vals]

    fig, ax = plt.subplots(figsize=(6.7, 5.5))
    y = np.arange(len(names))
    ax.axvspan(0.0, 10.0, color="#e6f0fa", alpha=0.75, zorder=0)
    ax.axvline(10.0, color=COLORS["tight"], linestyle="--", linewidth=1.4, zorder=1, label="10% Tight Threshold")
    ax.barh(y, vals, height=0.66, color=colors, zorder=3)

    for yi, v in zip(y, vals):
        c = COLORS["tight"] if v < 10.0 else COLORS["sloppy"]
        ax.text(
            v + 0.8, yi, f"{v:.1f}%",
            va="center", ha="left", fontsize=9.8, fontweight="bold", color=c,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85, edgecolor="none"),
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.8, fontweight="bold")
    ax.set_xlabel("Relative Parameter Error (%)", fontsize=11.5, fontweight="bold")
    ax.set_title("All-20 Free Parameters: 7/20 Within 10%", fontsize=12.5, fontweight="bold", pad=8)
    ax.set_xlim(0.0, 62.0)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.grid(True, axis="x", alpha=0.28, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=10.0, frameon=True, facecolor="white")

    plt.tight_layout(pad=0.7)
    out = assets_dir / "slide07_all20_params.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out


def generate_slide07_identifiability(test1: dict, test2: dict, fim: dict, assets_dir: Path) -> Path:
    """Slide 7 Right: f2(G) saturation + tight-tolerance 20x20 / 17x17 FIM eigenspectrum."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 5.5), gridspec_kw={"height_ratios": [1.0, 1.15]})

    # Top panel: f2(G) saturation across observed G range
    g_vals = np.linspace(0.0, 20000.0, 400)
    ub, vg = PARAMS["Ub"], PARAMS["Vg"]
    t1_c2, t2_c2 = test1["inferred"]["C2"], test2["inferred"]["C2"]
    ax1.axvspan(8300, 18700, color="#f4e6d8", alpha=0.7, label="Observed G [8.3k, 18.7k] mg")
    ax1.axhline(ub, color="gray", linestyle="--", linewidth=1.1)
    ax1.plot(g_vals, ub * (1.0 - np.exp(-g_vals / (PARAMS["C2"] * vg))), color=COLORS["exact"], linewidth=2.0, label=f"Nominal (C2={PARAMS['C2']:.0f})")
    ax1.plot(g_vals, ub * (1.0 - np.exp(-g_vals / (t2_c2 * vg))), "--", color=COLORS["test2"], linewidth=1.9, label=f"All-20 C2={t2_c2:.1f} (nom. Ub, Vg)")
    ax1.plot(g_vals, ub * (1.0 - np.exp(-g_vals / (t1_c2 * vg))), ":", color=COLORS["test1"], linewidth=2.0, label=f"Vol.-Fixed C2={t1_c2:.1f} (nom. Ub)")
    ax1.set_title("1. f2(G) Sensitivity to C2 (Holding Nominal Ub, Vg)", fontsize=11.3, fontweight="bold", pad=5)
    ax1.set_xlabel("Glucose Mass G (mg)", fontsize=10.5)
    ax1.set_ylabel("f2(G) (mg/min)", fontsize=10.5)
    ax1.set_xlim(0, 20000)
    ax1.set_ylim(0, 82)
    ax1.tick_params(labelsize=9.8)
    ax1.grid(True, alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower right", fontsize=8.4, frameon=True, facecolor="white")

    # Bottom panel: Tight-tolerance FIM eigenvalues
    e_g = np.array(fim["G_only_fim"]["eigenvalues"])
    e_gt1 = np.array(fim["G_plus_T1_fim"]["eigenvalues"])
    e_t1 = np.array(fim["test1_G_only_fim"]["eigenvalues"])

    ax2.semilogy(np.arange(1, 21), np.maximum(e_g, 1e-10), "o-", color=COLORS["sloppy"], linewidth=1.8, markersize=4.5, label="All-20 G-only FIM (600 pts)")
    ax2.semilogy(np.arange(1, 21), np.maximum(e_gt1, 1e-10), "s--", color=COLORS["exact"], linewidth=1.6, markersize=4.2, label="All-20 G + T1 Anchor FIM")
    ax2.semilogy(np.arange(1, 18), np.maximum(e_t1, 1e-10), "^:", color=COLORS["test2"], linewidth=1.8, markersize=4.5, label="Vol.-Fixed G-only (17 free)")

    ax2.annotate("ti mode (1.57e-4 -> 4.07e-2 w/ T1)\nC2 stays 2.09e-4 -> 2.22e-4", xy=(17.0, 4.0e-2), xytext=(3.4, 5.5e-4),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0), fontsize=8.3, fontweight="bold", color="#222222")
    ax2.annotate("2 analytical null modes\n(λ=0; ~1e-8 num. floor)", xy=(19.5, 1.2e-8), xytext=(6.8, 3.0e-8),
                 arrowprops=dict(arrowstyle="->", color="#902020", lw=1.0), fontsize=8.4, fontweight="bold", color="#902020")

    ax2.set_title("2. Numerical FIM Spectrum (rtol=1e-10, atol=1e-12)", fontsize=11.5, fontweight="bold", pad=5)
    ax2.set_xlabel("Eigenmode Index (sorted largest -> smallest)", fontsize=10.5)
    ax2.set_ylabel("FIM Eigenvalue (log)", fontsize=10.5)
    ax2.set_xticks([1, 5, 10, 15, 17, 20])
    ax2.set_ylim(2e-9, 5e5)
    ax2.tick_params(labelsize=9.8)
    ax2.grid(True, which="both", alpha=0.22)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", fontsize=8.6, frameon=True, facecolor="white")

    plt.tight_layout(pad=0.8)
    out = assets_dir / "slide07_identifiability.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out


def generate_slide08_forecast_3panel(step1: dict, test1: dict, test2: dict, assets_dir: Path, seed: int = 42, n_obs: int = 600) -> Path:
    """Slide 8: Clean 3-panel (G, Ip, Ii) Fig. 8 rollout dropping trivial IG."""
    x0_np = np.array([12.0 * PARAMS["Vp"], 4.0 * PARAMS["Vi"], 110.0 * (10.0 * PARAMS["Vg"]), 0.0, 0.0, 0.0])
    t_train = np.array(step1["trajectory"]["t_min"])
    g_train_dl = np.array(step1["trajectory"]["G_mg_dl"])
    obs_idx = np.sort(np.random.default_rng(seed).choice(len(t_train), size=n_obs, replace=False))

    meals_ext = MEALS + [(2000.0, 100.0)]
    t_ext = np.linspace(0.0, 3000.0, 3001)
    true_ext = odeint(ultradian_rhs, x0_np, t_ext, args=(PARAMS, meals_ext), rtol=1e-10, atol=1e-12)
    t1_ext = odeint(ultradian_rhs, x0_np, t_ext, args=(test1["inferred"], meals_ext), rtol=1e-10, atol=1e-12)
    t2_ext = odeint(ultradian_rhs, x0_np, t_ext, args=(test2["inferred"], meals_ext), rtol=1e-10, atol=1e-12)

    scale_g = 10.0 * PARAMS["Vg"]
    panels = [
        (true_ext[:, 2] / scale_g, t2_ext[:, 2] / scale_g, t1_ext[:, 2] / scale_g, "Glucose G (mg/dl @ nom. Vg)", (75, 248)),
        (true_ext[:, 0], t2_ext[:, 0], t1_ext[:, 0], "Plasma Insulin Ip (mU)", (0, 680)),
        (true_ext[:, 1], t2_ext[:, 1], t1_ext[:, 1], "Interstitial Insulin Ii (mU)", (0, 790)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1))
    legend_handles, legend_labels = [], []
    for idx, (ax, (y_ex, y_2, y_1, title, ylim)) in enumerate(zip(axes, panels)):
        l_ex, = ax.plot(t_ext, y_ex, color=COLORS["exact"], linewidth=2.0)
        l_t2, = ax.plot(t_ext, y_2, "--", color=COLORS["test2"], linewidth=2.0)
        l_t1, = ax.plot(t_ext, y_1, ":", color=COLORS["test1"], linewidth=2.0)
        if idx == 0:
            sc = ax.scatter(t_train[obs_idx], g_train_dl[obs_idx], s=8, color="#111111", alpha=0.35, zorder=4)
            legend_handles = [l_ex, l_t2, l_t1, sc]
            legend_labels = ["Exact (odeint)", "All-20 (20 Free)", "Vol.-Fixed (17 Free)", "Train G (600 pts)"]
        ax.axvline(1800.0, color="#666666", linestyle="--", linewidth=1.2)
        ax.set_title(title, fontsize=13.0, fontweight="bold", pad=7)
        ax.set_xlabel("t (min)", fontsize=11.5)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=10.5)
        ax.grid(True, alpha=0.26)

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, frameon=True, facecolor="white", fontsize=10.8, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout(rect=[0, 0, 1, 0.91], pad=0.9)
    out = assets_dir / "slide08_fig8_3panel.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out


def generate_slide09_ablation(step1: dict, ablation: dict, assets_dir: Path) -> Path:
    """Slide 9: Matched 600-pt Step A vs B vs C2 error-vs-time & Stage-2 convergence."""
    t_eval = np.array(step1["trajectory"]["t_min"])
    true_g = np.array(step1["trajectory"]["G_mg_dl"])
    true_ip = np.array(step1["trajectory"]["Ip_mU"])

    s_a = ablation["step_a_naive_12k"]
    s_b = ablation["step_b_zscore_no_s15"]
    s_c2 = ablation["step_c2_zscore_plus_s15"]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))

    # Panel 1: |G_pred - G_true| over time
    err_g_a = np.abs(np.array(s_a["predicted_trajectory"]["G_mg_dl"]) - true_g)
    err_g_b = np.abs(np.array(s_b["predicted_trajectory"]["G_mg_dl"]) - true_g)
    err_g_c = np.abs(np.array(s_c2["predicted_trajectory"]["G_mg_dl"]) - true_g)
    l_a, = axes[0].plot(t_eval, err_g_a, ":", color=COLORS["step_a"], linewidth=1.8)
    l_b, = axes[0].plot(t_eval, err_g_b, "--", color=COLORS["step_b"], linewidth=1.7)
    l_c, = axes[0].plot(t_eval, err_g_c, "-", color=COLORS["pred"], linewidth=1.7)
    axes[0].set_title("Glucose Error |ΔG| (mg/dl)", fontsize=12.5, fontweight="bold", pad=6)
    axes[0].set_xlabel("t (min)", fontsize=11.0)
    axes[0].set_ylabel("Abs. Error (mg/dl)", fontsize=10.5)
    axes[0].grid(True, alpha=0.26)

    # Panel 2: Hidden |Ip_pred - Ip_true| over time
    err_ip_a = np.abs(np.array(s_a["predicted_trajectory"]["Ip_mU"]) - true_ip)
    err_ip_b = np.abs(np.array(s_b["predicted_trajectory"]["Ip_mU"]) - true_ip)
    err_ip_c = np.abs(np.array(s_c2["predicted_trajectory"]["Ip_mU"]) - true_ip)
    axes[1].plot(t_eval, err_ip_a, ":", color=COLORS["step_a"], linewidth=1.8)
    axes[1].plot(t_eval, err_ip_b, "--", color=COLORS["step_b"], linewidth=1.7)
    axes[1].plot(t_eval, err_ip_c, "-", color=COLORS["pred"], linewidth=1.7)
    axes[1].set_title("Hidden Insulin Error |ΔIp| (mU)", fontsize=12.5, fontweight="bold", pad=6)
    axes[1].set_xlabel("t (min)", fontsize=11.0)
    axes[1].set_ylabel("Abs. Error (mU)", fontsize=10.5)
    axes[1].grid(True, alpha=0.26)

    # Panel 3: Deterministic final obs-scale ODE residual on uniform 601-pt grid (apples-to-apples!)
    ode_vals = [
        s_a["final_ode_obs_scale"],
        s_b["final_ode_obs_scale"],
        s_c2["final_ode_obs_scale"],
    ]
    bar_labels = ["Step A\n(19k ep)", "Step B\n(19k ep)", "Step C\n(21.5k ep)"]
    bar_colors = [COLORS["step_a"], COLORS["step_b"], COLORS["pred"]]
    x_pos = np.arange(3)
    bars = axes[2].bar(x_pos, ode_vals, width=0.52, color=bar_colors, alpha=0.90, edgecolor="#222222", linewidth=0.8)
    ratio_b = ode_vals[0] / ode_vals[1]
    ratio_c = ode_vals[0] / ode_vals[2]
    annots = [f"{ode_vals[0]:.4f}", f"{ode_vals[1]:.4f}\n({ratio_b:.1f}× ↓)", f"{ode_vals[2]:.4f}\n({ratio_c:.1f}× ↓)"]
    for bar, txt in zip(bars, annots):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.0018,
            txt,
            ha="center", va="bottom", fontsize=9.8, fontweight="bold", color="#1f2937",
        )
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(bar_labels, fontsize=10.2, fontweight="bold")
    axes[2].set_ylim(0.0, 0.064)
    axes[2].set_ylabel("Obs-Scale MSE", fontsize=10.5)
    axes[2].set_title("Final ODE Residual (601-pt Grid)", fontsize=12.2, fontweight="bold", pad=6)
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.legend(
        [l_a, l_b, l_c],
        [
            f"Step A: Naive 12k-scale (19k ep; ODE {ode_vals[0]:.4f})",
            f"Step B: + Z-score (19k ep; ODE {ode_vals[1]:.4f})",
            f"Step C: + Stage 1.5 (21.5k ep; ODE {ode_vals[2]:.4f})",
        ],
        loc="upper center", ncol=3, frameon=True, facecolor="white", fontsize=9.8, bbox_to_anchor=(0.5, 1.02),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.89], pad=1.0)
    out = assets_dir / "slide09_ablation_and_loss.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out


def main():
    py_dir = Path(__file__).resolve().parent
    out_root = py_dir / "outputs"
    assets_dir = py_dir.parent / "slides" / "assets updated"
    assets_dir.mkdir(parents=True, exist_ok=True)

    step1 = json.loads((out_root / "step 1" / "step1_ultradian_truth.json").read_text(encoding="utf-8"))
    step2 = json.loads((out_root / "step 2" / "step2_forward_results.json").read_text(encoding="utf-8"))
    test1 = json.loads((out_root / "step 3" / "step3_results_test1.json").read_text(encoding="utf-8"))[0]
    test2 = json.loads((out_root / "step 3" / "step3_results_test2.json").read_text(encoding="utf-8"))[0]
    fim = json.loads((out_root / "step 3" / "fim_identifiability.json").read_text(encoding="utf-8"))
    ablation = json.loads((out_root / "step 3" / "ablation_600pt_test2.json").read_text(encoding="utf-8"))

    saved = [
        generate_slide05_forward(step1, step2, assets_dir),
        generate_slide06_inverse(step1, test2, assets_dir),
        generate_slide07_all20_params(test2, assets_dir),
        generate_slide07_identifiability(test1, test2, fim, assets_dir),
        generate_slide08_forecast_3panel(step1, test1, test2, assets_dir),
        generate_slide09_ablation(step1, ablation, assets_dir),
    ]
    print("Generated slide assets:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()

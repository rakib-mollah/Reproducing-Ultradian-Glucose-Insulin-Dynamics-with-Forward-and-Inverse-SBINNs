"""Step 3 (Upgraded): Fast Inverse SBINN with G z-score normalization, curriculum ODE ramp, fwAD, Test 1 & Test 2, and 4-panel Fig. 8 forecast."""
import argparse
from pathlib import Path
import time
import json
import numpy as np
from scipy.integrate import odeint
import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step1_verify_ode import PARAMS, MEALS, meal_intake, ultradian_rhs
from step2_forward_sbinn import (
    T_MAX, STATE_NAMES, FREQS,
    ultradian_rhs_torch, sample_causal_collocation, load_step1_truth,
)

# Priors for hidden states (G mean/std will be computed purely from observed G_obs!)
HIDDEN_MUS = [70.0, 140.0, 0.0, 70.0, 70.0, 70.0]
HIDDEN_STDS = [60.0, 70.0, 1.0, 50.0, 45.0, 40.0]


class InverseSBINN(nn.Module):
    """4x128 SiLU SBINN with z-score output normalization (mu + std * NN) and Hard IC at t=0."""

    def __init__(self, x0: np.ndarray, mus, stds, freqs=FREQS, width: int = 128, depth: int = 4):
        super().__init__()
        self.register_buffer("x0", torch.tensor(x0[None, :], dtype=torch.float32))
        self.register_buffer("mus", torch.tensor(mus, dtype=torch.float32))
        self.register_buffer("stds", torch.tensor(stds, dtype=torch.float32))
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))
        in_dim = 1 + 2 * len(freqs)

        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, 6))
        self.net = nn.Sequential(*layers)

    def features(self, t_scaled: torch.Tensor) -> torch.Tensor:
        arg = t_scaled * self.freqs
        return torch.cat([t_scaled, torch.sin(arg), torch.cos(arg)], dim=1)

    def forward(self, t_scaled: torch.Tensor) -> torch.Tensor:
        # Smooth transition from exact x0 at t=0 to z-score normalized representation across (0, 1800]
        raw_curve = self.mus + self.stds * self.net(self.features(t_scaled))
        gate = 1.0 - torch.exp(-50.0 * t_scaled)  # ~0 at t=0, ~1 by t=50 min
        return (1.0 - gate) * self.x0 + gate * raw_curve


class TrainableParams(nn.Module):
    """Bounded physiological parameters (0.2x .. 1.8x), initialized away from truth."""

    def __init__(self, nominal: dict, fixed_keys: set, seed: int = 42):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.fixed = {k: float(v) for k, v in nominal.items() if k in fixed_keys}
        self.bounds = {}
        self.init_vals = {}
        raw_init = {}

        for k, val in nominal.items():
            if k in fixed_keys:
                continue
            lo, hi = 0.2 * val, 1.8 * val
            guess = float(val * rng.uniform(0.55, 1.45))
            u = np.clip((guess - lo) / (hi - lo), 0.05, 0.95)
            self.bounds[k] = (lo, hi)
            self.init_vals[k] = guess
            raw_init[k] = float(np.log(u / (1.0 - u)))

        self.raw = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v, dtype=torch.float32))
            for k, v in raw_init.items()
        })

    def as_dict(self) -> dict:
        out = dict(self.fixed)
        for k, (lo, hi) in self.bounds.items():
            out[k] = lo + (hi - lo) * torch.sigmoid(self.raw[k])
        return out

    def values_float(self) -> dict:
        with torch.no_grad():
            return {k: float(v.item()) if isinstance(v, torch.Tensor) else v for k, v in self.as_dict().items()}


def plot_inverse_states(t_eval, true_states, pred_states, obs_idx, n_obs, test_mode, seed, img_dir: Path):
    """Save 6-panel inverse hidden-state recovery figure without duplicate suptitle."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8))
    for i, (ax, name) in enumerate(zip(axes.flat, STATE_NAMES)):
        scale = 10.0 * PARAMS["Vg"] if i == 2 else 1.0
        ax.plot(t_eval, true_states[:, i] / scale, label="odeint (Truth)", linewidth=1.9)
        ax.plot(t_eval, pred_states[:, i] / scale, "--", label="Inverse SBINN", linewidth=1.7)
        if i == 2:
            ax.scatter(t_eval[obs_idx], true_states[obs_idx, 2] / scale, s=9, color="black", alpha=0.45, label=f"Observed G ({n_obs} pts)")
        ax.set_title(f"{name}" + (" [Observed]" if i == 2 else " [Hidden]"), fontsize=13.5, fontweight="bold", pad=7)
        ax.set_xlabel("t (min)", fontsize=12.5)
        ax.tick_params(labelsize=11.5)
        ax.grid(True, alpha=0.3)
        if i in (0, 2):
            ax.legend(fontsize=10.5)
    plt.tight_layout(pad=1.0)
    out_states = img_dir / f"step3_inverse_{test_mode}_s{seed}.png"
    plt.savefig(out_states, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_states


def plot_forecast_fig8(inferred: dict, obs_seed: int, n_obs: int, test_mode: str, seed: int, img_dir: Path):
    """Save 4-panel Fig. 8 out-of-sample meal forecast figure without duplicate suptitle."""
    x0_np = np.array([12.0 * PARAMS["Vp"], 4.0 * PARAMS["Vi"], 110.0 * (10.0 * PARAMS["Vg"]), 0.0, 0.0, 0.0])
    t_eval, true_states, _ = load_step1_truth(x0_np)
    rng = np.random.default_rng(obs_seed)
    obs_idx = np.sort(rng.choice(len(t_eval), size=n_obs, replace=False))

    meals_ext = MEALS + [(2000.0, 100.0)]
    t_ext = np.linspace(0.0, 3000.0, 3001)
    true_ext = odeint(ultradian_rhs, x0_np, t_ext, args=(PARAMS, meals_ext))
    pred_ext = odeint(ultradian_rhs, x0_np, t_ext, args=(inferred, meals_ext))

    ig_true = np.array([meal_intake(ti, meals_ext, PARAMS["k"]) for ti in t_ext])
    ig_pred = np.array([meal_intake(ti, meals_ext, inferred["k"]) for ti in t_ext])

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.0))
    panels = [
        (true_ext[:, 0], pred_ext[:, 0], "Plasma Insulin Ip (mU)"),
        (true_ext[:, 1], pred_ext[:, 1], "Interstitial Insulin Ii (mU)"),
        (true_ext[:, 2] / (10.0 * PARAMS["Vg"]), pred_ext[:, 2] / (10.0 * PARAMS["Vg"]), "Glucose G (mg/dl)"),
        (ig_true, ig_pred, "Meal Intake IG (mg/min)"),
    ]
    for idx_p, (ax, (y_t, y_p, title)) in enumerate(zip(axes.flat, panels)):
        ax.plot(t_ext, y_t, label="Exact", linewidth=1.9)
        ax.plot(t_ext, y_p, "--", label="Inferred Forecast", linewidth=1.7)
        if idx_p == 2:
            ax.scatter(t_eval[obs_idx], true_states[obs_idx, 2] / (10.0 * PARAMS["Vg"]), s=9, color="black", alpha=0.35, label="Training Obs")
        ax.axvline(1800.0, color="gray", linestyle=":", linewidth=1.3)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=7)
        ax.set_xlabel("t (min)", fontsize=12.5)
        ax.tick_params(labelsize=11.5)
        ax.grid(True, alpha=0.3)
        if idx_p in (0, 2):
            ax.legend(fontsize=11)
    plt.tight_layout(pad=1.0)
    out_fc = img_dir / f"step3_forecast_{test_mode}_s{seed}.png"
    plt.savefig(out_fc, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    fc_mask = t_ext >= 1800.0
    g_true_dl = true_ext[:, 2] / (10.0 * PARAMS["Vg"])
    g_pred_dl = pred_ext[:, 2] / (10.0 * PARAMS["Vg"])
    forecast_metrics = {
        "forecast_window_min": [1800.0, 3000.0],
        "unseen_meal": {"t_min": 2000.0, "carbs_g": 100.0},
        "rel_l2_percent_post_1800": {
            "Ip (mU)": round(float(np.linalg.norm(pred_ext[fc_mask, 0] - true_ext[fc_mask, 0]) / np.linalg.norm(true_ext[fc_mask, 0]) * 100.0), 4),
            "Ii (mU)": round(float(np.linalg.norm(pred_ext[fc_mask, 1] - true_ext[fc_mask, 1]) / np.linalg.norm(true_ext[fc_mask, 1]) * 100.0), 4),
            "G (mg/dl)": round(float(np.linalg.norm(g_pred_dl[fc_mask] - g_true_dl[fc_mask]) / np.linalg.norm(g_true_dl[fc_mask]) * 100.0), 4),
            "IG (mg/min)": round(float(np.linalg.norm(ig_pred[fc_mask] - ig_true[fc_mask]) / np.linalg.norm(ig_true[fc_mask]) * 100.0), 4),
        },
        "peak_G_post_2000_exact_mg_dl": round(float(g_true_dl[t_ext >= 2000.0].max()), 4),
        "peak_G_post_2000_pred_mg_dl": round(float(g_pred_dl[t_ext >= 2000.0].max()), 4),
        "peak_Ip_post_2000_exact_mU": round(float(true_ext[t_ext >= 2000.0, 0].max()), 4),
        "peak_Ip_post_2000_pred_mU": round(float(pred_ext[t_ext >= 2000.0, 0].max()), 4),
    }
    forecast_trajectory = {
        "t_min": np.round(t_ext, 6).tolist(),
        "Ip_mU_pred": np.round(pred_ext[:, 0], 6).tolist(),
        "Ii_mU_pred": np.round(pred_ext[:, 1], 6).tolist(),
        "G_mg_dl_pred": np.round(g_pred_dl, 6).tolist(),
        "IG_mg_min_pred": np.round(ig_pred, 6).tolist(),
    }
    return out_fc, forecast_metrics, forecast_trajectory


def run_inverse_experiment(
    test_mode: str = "test1",
    seed: int = 42,
    n_obs: int = 600,
    stage2_epochs: int = 16000,
    save_plots: bool = True,
    out_dir: Path | None = None,
    use_zscore: bool = True,
    stage15_epochs: int = 2500,
):
    if torch.get_num_threads() != 2:
        torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    t0_wall = time.time()

    fixed_keys = {"Vp", "Vi", "Vg"} if test_mode == "test1" else set()
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "outputs" / "step 3"

    # 1. Ground truth trajectory over [0, 1800] min
    x0_np = np.array([12.0 * PARAMS["Vp"], 4.0 * PARAMS["Vi"], 110.0 * (10.0 * PARAMS["Vg"]), 0.0, 0.0, 0.0])
    t_eval, true_states, truth_source = load_step1_truth(x0_np)

    # 2. Observe ONLY Glucose G at n_obs random timestamps
    obs_idx = np.sort(rng.choice(len(t_eval), size=n_obs, replace=False))
    t_data = torch.tensor(t_eval[obs_idx, None] / T_MAX, dtype=torch.float32)
    G_obs_np = true_states[obs_idx, 2:3]
    G_obs = torch.tensor(G_obs_np, dtype=torch.float32)

    # Z-score stats for G computed strictly from observed data (no leakage!)
    if use_zscore:
        mu_G = float(G_obs_np.mean())
        std_G = float(G_obs_np.std())
    else:
        mu_G = 0.0
        std_G = 12000.0
    mus = list(HIDDEN_MUS); mus[2] = mu_G
    stds = list(HIDDEN_STDS); stds[2] = std_G
    stds_t = torch.tensor(stds, dtype=torch.float32)
    deriv_scales = stds_t / 60.0

    # Auxiliary endpoint anchor at T1 = 1800 (T0=0 is enforced identically by Hard IC gate)
    t_T1 = torch.tensor([[1.0]], dtype=torch.float32)
    x_T1 = torch.tensor(true_states[-1:, :], dtype=torch.float32)

    model = InverseSBINN(x0_np, mus, stds)
    tparams = TrainableParams(PARAMS, fixed_keys=fixed_keys, seed=seed)
    training_log = {"stage1": [], "stage1_5": [], "stage2": []}

    # --- Stage 1: Supervised pre-fit on G_obs + T1 anchor ---
    opt_s1 = torch.optim.Adam(model.parameters(), lr=2e-3)
    for epoch in range(1, 3001):
        opt_s1.zero_grad()
        loss_data = torch.mean(((model(t_data)[:, 2:3] - G_obs) / std_G) ** 2)
        loss_aux = torch.mean(((model(t_T1) - x_T1) / stds_t) ** 2)
        loss = loss_data + 0.5 * loss_aux
        loss.backward()
        opt_s1.step()
        if epoch in (1, 1500, 3000):
            with torch.no_grad():
                g_pred = model(torch.tensor(t_eval[:, None] / T_MAX, dtype=torch.float32))[:, 2].numpy()
                g_rel = np.linalg.norm(g_pred - true_states[:, 2]) / np.linalg.norm(true_states[:, 2]) * 100.0
            training_log["stage1"].append({
                "epoch": epoch,
                "data_z": round(float(loss_data.item()), 6),
                "G_rel_l2_percent": round(float(g_rel), 4),
            })
            print(f"[{test_mode} seed={seed} | Stage 1 {epoch:4d}] data_z={loss_data.item():.5f} | G_relL2={g_rel:.2f}%")

    # --- Stage 1.5: Wake up hidden states with ODE loss before updating tparams ---
    if stage15_epochs > 0:
        opt_s15 = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1, stage15_epochs + 1):
            opt_s15.zero_grad()
            t_col = sample_causal_collocation(n_chunks=18, pts_per_chunk=12, rng=rng)
            with fwAD.dual_level():
                t_dual = fwAD.make_dual(t_col, torch.ones_like(t_col))
                x_dual = model(t_dual)
                x_col, dx_dtilde = fwAD.unpack_dual(x_dual)
            dx_dt = dx_dtilde / T_MAX
            rhs = ultradian_rhs_torch(x_col, t_col * T_MAX, tparams.as_dict())
            loss_ode = torch.mean(((dx_dt - rhs) / deriv_scales) ** 2)
            loss_data = torch.mean(((model(t_data)[:, 2:3] - G_obs) / std_G) ** 2)
            loss_aux = torch.mean(((model(t_T1) - x_T1) / stds_t) ** 2)
            w_ode_15 = min(1.0, 0.05 + 0.95 * (epoch / 2000.0))
            loss = 50.0 * loss_data + 20.0 * loss_aux + w_ode_15 * loss_ode
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_s15.step()
            if epoch in (1, stage15_epochs):
                training_log["stage1_5"].append({
                    "epoch": epoch,
                    "data_z": round(float(loss_data.item()), 6),
                    "ode": round(float(loss_ode.item()), 6),
                })
                print(f"[{test_mode} seed={seed} | Stage 1.5 {epoch:4d}] data_z={loss_data.item():.5f} | ode={loss_ode.item():.4f}")

    # --- Stage 2: Joint Physics + Data optimization (updates weights AND parameters) ---
    opt_s2 = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1e-3},
        {"params": tparams.parameters(), "lr": 2.5e-3},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=stage2_epochs, eta_min=1e-4)

    for epoch in range(1, stage2_epochs + 1):
        opt_s2.zero_grad()
        t_col = sample_causal_collocation(n_chunks=18, pts_per_chunk=12, rng=rng)

        with fwAD.dual_level():
            t_dual = fwAD.make_dual(t_col, torch.ones_like(t_col))
            x_dual = model(t_dual)
            x_col, dx_dtilde = fwAD.unpack_dual(x_dual)

        dx_dt = dx_dtilde / T_MAX
        rhs = ultradian_rhs_torch(x_col, t_col * T_MAX, tparams.as_dict())

        loss_ode = torch.mean(((dx_dt - rhs) / deriv_scales) ** 2)
        loss_data = torch.mean(((model(t_data)[:, 2:3] - G_obs) / std_G) ** 2)
        loss_aux = torch.mean(((model(t_T1) - x_T1) / stds_t) ** 2)

        loss = 50.0 * loss_data + 20.0 * loss_aux + loss_ode

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt_s2.step()
        sched.step()

        if epoch % 1000 == 0 or epoch == 1:
            training_log["stage2"].append({
                "epoch": epoch,
                "data_z": round(float(loss_data.item()), 6),
                "aux_z": round(float(loss_aux.item()), 6),
                "ode": round(float(loss_ode.item()), 6),
            })
            if epoch % 4000 == 0 or epoch == 1:
                print(f"[{test_mode} seed={seed} | Stage 2 {epoch:5d}] data_z={loss_data.item():.5f} | aux_z={loss_aux.item():.5f} | ode={loss_ode.item():.4f}")

    elapsed = time.time() - t0_wall
    print(f"\n[{test_mode} seed={seed}] Completed in {elapsed:.1f}s")

    # Comparable post-training ODE residual normalized strictly by observed std_G (apples-to-apples across Step A / B / C)
    stds_obs_t = torch.tensor([HIDDEN_STDS[0], HIDDEN_STDS[1], float(G_obs_np.std()), HIDDEN_STDS[3], HIDDEN_STDS[4], HIDDEN_STDS[5]], dtype=torch.float32)
    deriv_obs_scales = stds_obs_t / 60.0
    t_eval_col = torch.tensor(np.linspace(0.0, 1.0, 601)[:, None], dtype=torch.float32)
    with fwAD.dual_level():
        t_dual = fwAD.make_dual(t_eval_col, torch.ones_like(t_eval_col))
        x_dual = model(t_dual)
        x_ev, dx_dtilde_ev = fwAD.unpack_dual(x_dual)
    with torch.no_grad():
        rhs_ev = ultradian_rhs_torch(x_ev, t_eval_col * T_MAX, tparams.as_dict())
        final_ode_obs_scale = float(torch.mean(((dx_dtilde_ev / T_MAX - rhs_ev) / deriv_obs_scales) ** 2).item())

    # 3. Evaluate state recovery
    with torch.no_grad():
        pred_states = model(torch.tensor(t_eval[:, None] / T_MAX, dtype=torch.float32)).numpy()

    rel_l2_dict = {}
    state_metrics = {}
    print(f"\n[{test_mode} seed={seed}] Relative L2 Errors (only G observed):")
    for i, name in enumerate(STATE_NAMES):
        scale = 10.0 * PARAMS["Vg"] if i == 2 else 1.0
        true_i = true_states[:, i]
        pred_i = pred_states[:, i]
        rel_l2 = float(np.linalg.norm(pred_i - true_i) / np.linalg.norm(true_i) * 100.0)
        diff_display = (pred_i - true_i) / scale
        rel_l2_dict[name] = rel_l2
        state_metrics[name] = {
            "role": "observed" if i == 2 else "hidden",
            "rel_l2_percent": round(rel_l2, 4),
            "rmse": round(float(np.sqrt(np.mean(diff_display ** 2))), 4),
            "mae": round(float(np.mean(np.abs(diff_display))), 4),
            "true_min": round(float((true_i / scale).min()), 4),
            "true_max": round(float((true_i / scale).max()), 4),
            "pred_min": round(float((pred_i / scale).min()), 4),
            "pred_max": round(float((pred_i / scale).max()), 4),
        }
        tag = " (OBSERVED)" if i == 2 else " (HIDDEN)"
        print(f"  {name:12s}: {rel_l2:.3f}%{tag}")

    inferred = tparams.values_float()
    param_rel_err = {}
    print(f"\n[{test_mode} seed={seed}] Parameter Inference:")
    print(f"  {'Param':8s} | {'True':>9s} | {'Init':>9s} | {'Inferred':>9s} | {'RelErr%':>8s}")
    print("  " + "-" * 54)
    for k in PARAMS:
        if k in fixed_keys:
            continue
        t_val, i_val, p_val = PARAMS[k], tparams.init_vals[k], inferred[k]
        err = abs(p_val - t_val) / t_val * 100.0
        param_rel_err[k] = round(err, 4)
        print(f"  {k:8s} | {t_val:9.4f} | {i_val:9.4f} | {p_val:9.4f} | {err:7.2f}%")

    forecast_metrics = None
    forecast_trajectory = None
    if save_plots:
        res_dir = Path(__file__).resolve().parent.parent / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            res_dir / f"step3_predictions_{test_mode}_s{seed}.npz",
            t_eval=t_eval,
            true_states=true_states,
            pred_states=pred_states,
            obs_idx=obs_idx,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_states = plot_inverse_states(t_eval, true_states, pred_states, obs_idx, n_obs, test_mode, seed, out_dir)
        out_fc, forecast_metrics, forecast_trajectory = plot_forecast_fig8(inferred, seed, n_obs, test_mode, seed, out_dir)
        print(f"Saved plots:\n  {out_states}\n  {out_fc}")

    return {
        "mode": test_mode,
        "seed": seed,
        "n_obs": n_obs,
        "stage2_epochs": stage2_epochs,
        "truth_source": truth_source,
        "elapsed": elapsed,
        "final_ode_obs_scale": round(final_ode_obs_scale, 6),
        "rel_l2": rel_l2_dict,
        "state_metrics": state_metrics,
        "true_params": PARAMS,
        "init": tparams.init_vals,
        "inferred": inferred,
        "param_rel_error_percent": param_rel_err,
        "training_log": training_log,
        "forecast_fig8_metrics": forecast_metrics,
        "predicted_trajectory": {
            "t_min": np.round(t_eval, 6).tolist(),
            "Ip_mU": np.round(pred_states[:, 0], 6).tolist(),
            "Ii_mU": np.round(pred_states[:, 1], 6).tolist(),
            "G_mg": np.round(pred_states[:, 2], 6).tolist(),
            "G_mg_dl": np.round(pred_states[:, 2] / (10.0 * PARAMS["Vg"]), 6).tolist(),
            "h1": np.round(pred_states[:, 3], 6).tolist(),
            "h2": np.round(pred_states[:, 4], 6).tolist(),
            "h3": np.round(pred_states[:, 5], 6).tolist(),
        },
        "forecast_trajectory": forecast_trajectory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test1", "test2", "both"], default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--n_obs", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=16000)
    parser.add_argument("--plot-forecast-only", action="store_true", help="Regenerate Fig. 8 forecast plot from saved JSON without retraining")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "outputs" / "step 3"
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["test1", "test2"] if args.mode == "both" else [args.mode]

    if args.plot_forecast_only:
        s0 = args.seeds[0]
        for mode in modes:
            out_json = out_dir / f"step3_results_{mode}.json"
            saved = json.loads(out_json.read_text(encoding="utf-8"))
            out_fc, _, _ = plot_forecast_fig8(saved[0]["inferred"], s0, args.n_obs, mode, s0, out_dir)
            print(f"Regenerated forecast plot: {out_fc}")
        return

    for mode in modes:
        out_json = out_dir / f"step3_results_{mode}.json"
        all_res = []
        for i, s in enumerate(args.seeds):
            res = run_inverse_experiment(
                test_mode=mode,
                seed=s,
                n_obs=args.n_obs,
                stage2_epochs=args.epochs,
                save_plots=(i == 0),
                out_dir=out_dir,
            )
            all_res.append(res)

        if len(all_res) > 1:
            print(f"\n=== Multi-Seed Summary ({mode}, seeds={args.seeds}) ===")
            fixed_keys = {"Vp", "Vi", "Vg"} if mode == "test1" else set()
            for k in PARAMS:
                if k in fixed_keys:
                    continue
                vals = np.array([r["inferred"][k] for r in all_res])
                rel_err = abs(vals.mean() - PARAMS[k]) / PARAMS[k] * 100.0
                print(f"  {k:8s}: true={PARAMS[k]:8.4f} | mean={vals.mean():8.4f} +/- {vals.std():7.4f} | mean RelErr={rel_err:6.2f}%")

        out_json.write_text(json.dumps(all_res, indent=2), encoding="utf-8")
        print(f"Saved JSON summary: {out_json}")


if __name__ == "__main__":
    main()

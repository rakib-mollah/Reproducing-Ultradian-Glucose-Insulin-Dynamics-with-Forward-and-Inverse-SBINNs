"""Step 2 (Upgraded): Fast Forward SBINN with Hard IC, non-periodic multi-scale features, causal weighting, and forward-mode AD."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import odeint
import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step1_verify_ode import PARAMS, MEALS, ultradian_rhs

T_MAX = 1800.0
STATE_NAMES = ["Ip (mU)", "Ii (mU)", "G (mg/dl)", "h1", "h2", "h3"]
# Characteristic oscillation amplitudes (matching signal std rather than baseline 12,000)
STATE_STDS = [60.0, 70.0, 2000.0, 50.0, 45.0, 40.0]
# Non-commensurate frequencies (no forced 2*pi periodicity at t=0 vs t=1800)
FREQS = [1.0, 2.0, 3.0, 4.0, 5.0, 8.3, 13.7, 21.1, 31.4, 45.7]


def meal_intake_torch(t: torch.Tensor, events=MEALS, k: float = PARAMS["k"]) -> torch.Tensor:
    """Vectorized causal meal intake I_G(t) in mg/min."""
    total = torch.zeros_like(t)
    for tj, mj_grams in events:
        active = (t >= tj).to(t.dtype)
        total = total + active * (mj_grams * 1000.0) * k * torch.exp(k * (tj - t))
    return total


def ultradian_rhs_torch(x: torch.Tensor, t: torch.Tensor, p: dict = PARAMS) -> torch.Tensor:
    """Differentiable 6-ODE Ultradian RHS matching Eqs. (S6)-(S7)."""
    Ip, Ii, G, h1, h2, h3 = [x[:, i : i + 1] for i in range(6)]
    Ii_safe = torch.clamp(Ii, min=1e-3)

    kappa = (1.0 / p["C4"]) * (1.0 / p["Vi"] + 1.0 / (p["E"] * p["ti"]))
    f1 = p["Rm"] / (1.0 + torch.exp(torch.clamp(-G / (p["Vg"] * p["C1"]) + p["a1"], -50.0, 50.0)))
    f2 = p["Ub"] * (1.0 - torch.exp(torch.clamp(-G / (p["C2"] * p["Vg"]), -50.0, 50.0)))
    f3 = (1.0 / p["Vg"]) * (p["U0_C3"] + p["Um_C3"] / (1.0 + (kappa * Ii_safe) ** (-p["beta"])))
    f4 = p["Rg"] / (1.0 + torch.exp(torch.clamp(p["alpha"] * (h3 / (p["C5"] * p["Vp"]) - 1.0), -50.0, 50.0)))
    ig = meal_intake_torch(t, MEALS, p["k"])

    dIp = f1 - p["E"] * (Ip / p["Vp"] - Ii / p["Vi"]) - Ip / p["tp"]
    dIi = p["E"] * (Ip / p["Vp"] - Ii / p["Vi"]) - Ii / p["ti"]
    dG = f4 + ig - f2 - f3 * G
    dh1 = (Ip - h1) / p["td"]
    dh2 = (h1 - h2) / p["td"]
    dh3 = (h2 - h3) / p["td"]
    return torch.cat([dIp, dIi, dG, dh1, dh2, dh3], dim=1)


class ForwardSBINN(nn.Module):
    """4x128 SiLU SBINN with non-periodic multi-scale sin/cos features and exact Hard IC."""

    def __init__(self, x0: np.ndarray, stds, freqs=FREQS, width: int = 128, depth: int = 4):
        super().__init__()
        self.register_buffer("x0", torch.tensor(x0[None, :], dtype=torch.float32))
        self.register_buffer("stds", torch.tensor(stds, dtype=torch.float32))
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))
        in_dim = 1 + 2 * len(freqs)

        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, 6))
        self.net = nn.Sequential(*layers)

    def features(self, t_scaled: torch.Tensor) -> torch.Tensor:
        arg = t_scaled * self.freqs  # [N, F]
        return torch.cat([t_scaled, torch.sin(arg), torch.cos(arg)], dim=1)

    def forward(self, t_scaled: torch.Tensor) -> torch.Tensor:
        # Exact Hard IC ansatz: x(0) == x0 identically
        return self.x0 + t_scaled * self.stds * self.net(self.features(t_scaled))


def sample_causal_collocation(n_chunks: int = 18, pts_per_chunk: int = 14, rng=None) -> torch.Tensor:
    """Sorted collocation grid across chunks + dense points near meal onsets [tj, tj+40]."""
    if rng is None:
        rng = np.random.default_rng()
    edges = np.linspace(0.0, T_MAX, n_chunks + 1)
    pts = []
    for i in range(n_chunks):
        pts.append(rng.uniform(edges[i], edges[i + 1], size=pts_per_chunk))
    # Extra meal-onset points
    for tj, _ in MEALS:
        pts.append(rng.uniform(tj, min(tj + 45.0, T_MAX), size=20))
    t_all = np.sort(np.concatenate(pts))
    return torch.tensor(t_all[:, None] / T_MAX, dtype=torch.float32)


def load_step1_truth(x0_np: np.ndarray):
    """Load Step 1 ground-truth trajectory JSON if present, else compute with odeint."""
    step1_json = Path(__file__).resolve().parent / "outputs" / "step 1" / "step1_ultradian_truth.json"
    if step1_json.exists():
        raw_traj = json.loads(step1_json.read_text(encoding="utf-8"))["trajectory"]
        t_eval = np.array(raw_traj["t_min"], dtype=float)
        true_states = np.column_stack([
            raw_traj["Ip_mU"],
            raw_traj["Ii_mU"],
            raw_traj["G_mg"],
            raw_traj["h1"],
            raw_traj["h2"],
            raw_traj["h3"],
        ])
        return t_eval, true_states, str(step1_json)
    t_eval = np.linspace(0.0, T_MAX, 1801)
    return t_eval, odeint(ultradian_rhs, x0_np, t_eval), "scipy.integrate.odeint"


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    t0_wall = time.time()

    x0_np = np.array([12.0 * PARAMS["Vp"], 4.0 * PARAMS["Vi"], 110.0 * (10.0 * PARAMS["Vg"]), 0.0, 0.0, 0.0])
    t_eval, true_states, truth_source = load_step1_truth(x0_np)

    stds_t = torch.tensor(STATE_STDS, dtype=torch.float32)
    # Time-scale normalized residual (meal timescale ~ 60 min gives cleaner O(1) balancing)
    deriv_scales = stds_t / 60.0

    model = ForwardSBINN(x0_np, STATE_STDS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    total_epochs = 12000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-4)

    n_bins = 18
    training_log = []
    for epoch in range(1, total_epochs + 1):
        optimizer.zero_grad()
        t_col = sample_causal_collocation(n_chunks=18, pts_per_chunk=12, rng=rng)  # ~276 sorted points

        # Forward-mode AD computes x_hat and dx_hat/d(t_scaled) in ONE pass!
        with fwAD.dual_level():
            t_dual = fwAD.make_dual(t_col, torch.ones_like(t_col))
            x_dual = model(t_dual)
            x_col, dx_dtilde = fwAD.unpack_dual(x_dual)

        dx_dt = dx_dtilde / T_MAX
        rhs = ultradian_rhs_torch(x_col, t_col * T_MAX)
        res_sq = torch.mean(((dx_dt - rhs) / deriv_scales) ** 2, dim=1)  # [N]

        # Causal weighting across sequential time bins (locks early dynamics before later meals)
        t_vals = (t_col[:, 0] * T_MAX).detach()
        bin_losses = []
        for b in range(n_bins):
            lo, hi = b * (T_MAX / n_bins), (b + 1) * (T_MAX / n_bins)
            mask = (t_vals >= lo) & (t_vals <= hi if b == n_bins - 1 else t_vals < hi)
            bin_losses.append(res_sq[mask].mean())
        bin_losses_t = torch.stack(bin_losses)

        # Ramp causal strictness gently so all bins train without freezing tail completely
        eps_causal = 0.05 if epoch < 3000 else 0.25
        with torch.no_grad():
            cum_prev = torch.cat([torch.zeros(1), torch.cumsum(bin_losses_t[:-1], dim=0)])
            causal_w = torch.exp(-eps_causal * cum_prev)
            causal_w = causal_w / causal_w.mean()

        loss = torch.mean(causal_w * bin_losses_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if epoch % 2000 == 0 or epoch == 1:
            w_ode = float(loss.item())
            m_ode = float(bin_losses_t.mean().item())
            min_w = float(causal_w.min().item())
            training_log.append({
                "epoch": epoch,
                "weighted_ode": round(w_ode, 6),
                "mean_ode": round(m_ode, 6),
                "min_w": round(min_w, 4),
            })
            print(f"[Epoch {epoch:5d}] weighted_ode={w_ode:.5f} | mean_ode={m_ode:.5f} | min_w={min_w:.3f}")

    wall_time = time.time() - t0_wall
    print(f"\nCompleted in {wall_time:.1f}s")

    with torch.no_grad():
        t_test = torch.tensor(t_eval[:, None] / T_MAX, dtype=torch.float32)
        pred_states = model(t_test).numpy()

    rel_l2_pct = {}
    state_metrics = {}
    print("\nForward SBINN Relative L2 Errors (vs. Step 1 Truth, zero interior data):")
    for i, name in enumerate(STATE_NAMES):
        scale = 10.0 * PARAMS["Vg"] if i == 2 else 1.0
        true_i = true_states[:, i]
        pred_i = pred_states[:, i]
        rel = float(np.linalg.norm(pred_i - true_i) / np.linalg.norm(true_i) * 100.0)
        diff_display = (pred_i - true_i) / scale
        rel_l2_pct[name] = round(rel, 4)
        state_metrics[name] = {
            "rel_l2_percent": round(rel, 4),
            "rmse": round(float(np.sqrt(np.mean(diff_display ** 2))), 4),
            "mae": round(float(np.mean(np.abs(diff_display))), 4),
            "true_min": round(float((true_i / scale).min()), 4),
            "true_max": round(float((true_i / scale).max()), 4),
            "pred_min": round(float((pred_i / scale).min()), 4),
            "pred_max": round(float((pred_i / scale).max()), 4),
        }
        print(f"  {name:12s}: {rel:.3f}%")

    out_dir = Path(__file__).resolve().parent / "outputs" / "step 2"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(__file__).resolve().parent.parent / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    np.savez(res_dir / "step2_predictions.npz", t_eval=t_eval, true_states=true_states, pred_states=pred_states)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8))
    for i, (ax, name) in enumerate(zip(axes.flat, STATE_NAMES)):
        scale = 10.0 * PARAMS["Vg"] if i == 2 else 1.0
        ax.plot(t_eval, true_states[:, i] / scale, label="odeint (Truth)", linewidth=1.9)
        ax.plot(t_eval, pred_states[:, i] / scale, "--", label="Forward SBINN", linewidth=1.7)
        ax.set_title(name, fontsize=14, fontweight="bold", pad=7)
        ax.set_xlabel("t (min)", fontsize=12.5)
        ax.tick_params(labelsize=11.5)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=11)

    plt.tight_layout(pad=1.0)
    out_path = out_dir / "step2_forward_sbinn.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    json_payload = {
        "truth_source": truth_source,
        "wall_time_s": round(wall_time, 2),
        "epochs": total_epochs,
        "architecture": {
            "depth": 4,
            "width": 128,
            "activation": "SiLU",
            "hard_ic": True,
            "t_max": T_MAX,
            "state_stds": STATE_STDS,
            "freqs": FREQS,
        },
        "training_log": training_log,
        "relative_l2_errors_percent": rel_l2_pct,
        "state_metrics": state_metrics,
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
    }
    json_path = out_dir / "step2_forward_results.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print(f"Saved plot to: {out_path}")
    print(f"Saved JSON to: {json_path}")


if __name__ == "__main__":
    main()

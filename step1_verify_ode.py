"""Step 1: Verify the 6-state Ultradian endocrine ODE simulator against file.pdf Figs. 7-8."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Nominal parameters from Table 5 (file.pdf, p. 14)
PARAMS = {
    "Vp": 3.0, "Vi": 11.0, "Vg": 10.0,
    "E": 0.2, "tp": 6.0, "ti": 100.0, "td": 12.0,
    "k": 0.0083, "Rm": 209.0, "a1": 6.6,
    "C1": 300.0, "C2": 144.0, "C4": 80.0, "C5": 26.0,
    "Ub": 72.0, "U0_C3": 0.04, "Um_C3": 0.9,
    "Rg": 180.0, "alpha": 7.5, "beta": 1.772,
}

# Three meals: (time in min, carbs in grams) from S4 Text
MEALS = [(300.0, 60.0), (650.0, 40.0), (1100.0, 50.0)]


def meal_intake(t: float, events=MEALS, k: float = PARAMS["k"]) -> float:
    """Exogenous glucose delivery I_G(t) in mg/min (causal: active only for t >= tj)."""
    total = 0.0
    for tj, mj_grams in events:
        if t >= tj:
            total += (mj_grams * 1000.0) * k * np.exp(k * (tj - t))
    return total


def ultradian_rhs(x, t: float, p: dict = PARAMS, events=MEALS):
    """6-ODE Ultradian glucose-insulin system (Eqs. S6-S7)."""
    Ip, Ii, G, h1, h2, h3 = x
    kappa = (1.0 / p["C4"]) * (1.0 / p["Vi"] + 1.0 / (p["E"] * p["ti"]))

    f1 = p["Rm"] / (1.0 + np.exp(-G / (p["Vg"] * p["C1"]) + p["a1"]))
    f2 = p["Ub"] * (1.0 - np.exp(-G / (p["C2"] * p["Vg"])))
    f3 = (1.0 / p["Vg"]) * (p["U0_C3"] + p["Um_C3"] / (1.0 + (kappa * Ii) ** (-p["beta"])))
    f4 = p["Rg"] / (1.0 + np.exp(p["alpha"] * (h3 / (p["C5"] * p["Vp"]) - 1.0)))
    ig = meal_intake(t, events, p["k"])

    dIp = f1 - p["E"] * (Ip / p["Vp"] - Ii / p["Vi"]) - Ip / p["tp"]
    dIi = p["E"] * (Ip / p["Vp"] - Ii / p["Vi"]) - Ii / p["ti"]
    dG = f4 + ig - f2 - f3 * G
    dh1 = (Ip - h1) / p["td"]
    dh2 = (h1 - h2) / p["td"]
    dh3 = (h2 - h3) / p["td"]
    return [dIp, dIi, dG, dh1, dh2, dh3]


def main():
    # Convert initial concentrations [12 uU/ml, 4 uU/ml, 110 mg/dl, 0, 0, 0] to amounts
    x0 = [
        12.0 * PARAMS["Vp"],
        4.0 * PARAMS["Vi"],
        110.0 * (10.0 * PARAMS["Vg"]),
        0.0, 0.0, 0.0,
    ]
    t = np.linspace(0.0, 1800.0, 1801)
    sol = odeint(ultradian_rhs, x0, t)

    Ip, Ii, G_mg, h1, h2, h3 = sol.T
    G_mg_dl = G_mg / (10.0 * PARAMS["Vg"])
    ig_curve = np.array([meal_intake(ti) for ti in t])

    print(f"G (mg/dl): min={G_mg_dl.min():.2f}, max={G_mg_dl.max():.2f}")
    print(f"Ip (mU):   min={Ip.min():.2f}, max={Ip.max():.2f}")
    print(f"Ii (mU):   min={Ii.min():.2f}, max={Ii.max():.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8))
    panels = [
        (G_mg_dl, "Glucose G (mg/dl)", "tab:red"),
        (Ip, "Plasma Insulin Ip (mU)", "tab:blue"),
        (Ii, "Interstitial Insulin Ii (mU)", "tab:green"),
        (ig_curve, "Meal Intake IG (mg/min)", "tab:orange"),
        (h1, "Delay State h1 (mU)", "tab:purple"),
        (h3, "Delay State h3 (mU)", "tab:brown"),
    ]
    for ax, (y, title, color) in zip(axes.flat, panels):
        ax.plot(t, y, color=color, linewidth=1.9)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=7)
        ax.set_xlabel("t (min)", fontsize=12.5)
        ax.tick_params(labelsize=11.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=1.0)
    out_dir = Path(__file__).resolve().parent / "outputs" / "step 1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step1_ultradian_truth.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    series_map = {
        "Ip_mU": Ip,
        "Ii_mU": Ii,
        "G_mg": G_mg,
        "G_mg_dl": G_mg_dl,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "IG_mg_min": ig_curve,
    }
    summary = {
        name: {
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "t0": round(float(arr[0]), 4),
            "t1800": round(float(arr[-1]), 4),
        }
        for name, arr in series_map.items()
    }
    window_bounds = [tj for tj, _ in MEALS] + [float(t[-1]) + 1.0]
    meal_peaks = []
    for idx, (tj, mj) in enumerate(MEALS):
        a = int(np.searchsorted(t, tj, side="left"))
        b = int(np.searchsorted(t, window_bounds[idx + 1], side="left"))
        peak_idx = a + int(np.argmax(G_mg_dl[a:b]))
        meal_peaks.append({
            "meal_time_min": float(tj),
            "carbs_g": float(mj),
            "IG_at_onset_mg_min": round(float(ig_curve[a]), 4),
            "peak_G_mg_dl": round(float(G_mg_dl[peak_idx]), 4),
            "peak_G_time_min": round(float(t[peak_idx]), 4),
        })

    json_payload = {
        "params": PARAMS,
        "meals": [{"t_min": float(tj), "carbs_g": float(mj)} for tj, mj in MEALS],
        "x0_amounts": [float(v) for v in x0],
        "t_range_min": [float(t[0]), float(t[-1])],
        "num_points": int(len(t)),
        "summary": summary,
        "meal_peaks": meal_peaks,
        "trajectory": {
            "t_min": np.round(t, 6).tolist(),
            **{name: np.round(arr, 6).tolist() for name, arr in series_map.items()},
        },
    }
    json_path = out_dir / "step1_ultradian_truth.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print(f"Saved plot to: {out_path}")
    print(f"Saved JSON to: {json_path}")


if __name__ == "__main__":
    main()

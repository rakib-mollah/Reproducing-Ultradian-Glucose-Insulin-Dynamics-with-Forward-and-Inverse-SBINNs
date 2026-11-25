"""Phase 1 Diagnostics: Tight-tolerance 20x20 FIM Identifiability Check + Matched 600-pt Step A/B Test 2 Ablations."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import odeint

from step1_verify_ode import PARAMS, MEALS, ultradian_rhs
from step2_forward_sbinn import T_MAX, STATE_NAMES
from step3_inverse_sbinn import HIDDEN_STDS, run_inverse_experiment


def compute_fim_identifiability(n_obs: int = 600, seed: int = 42, eps: float = 1e-4) -> dict:
    """Compute central-difference log-parameter FIM J^T J for G-only and G + T1 anchors."""
    param_names = list(PARAMS.keys())
    t_eval = np.linspace(0.0, T_MAX, 1801)
    x0_base = np.array([12.0 * PARAMS["Vp"], 4.0 * PARAMS["Vi"], 110.0 * (10.0 * PARAMS["Vg"]), 0.0, 0.0, 0.0])

    rng = np.random.default_rng(seed)
    obs_idx = np.sort(rng.choice(len(t_eval), size=n_obs, replace=False))

    sol_nom = odeint(ultradian_rhs, x0_base, t_eval, args=(PARAMS, MEALS), rtol=1e-10, atol=1e-12)
    std_G = float(sol_nom[obs_idx, 2].std())
    anchor_scales = np.array([HIDDEN_STDS[0], HIDDEN_STDS[1], std_G, HIDDEN_STDS[3], HIDDEN_STDS[4], HIDDEN_STDS[5]])

    def solve_for_params(p_dict: dict) -> np.ndarray:
        return odeint(ultradian_rhs, x0_base, t_eval, args=(p_dict, MEALS), rtol=1e-10, atol=1e-12)

    jac_G = np.zeros((n_obs, len(param_names)), dtype=float)
    jac_T1 = np.zeros((6, len(param_names)), dtype=float)

    for j, k in enumerate(param_names):
        p_plus = dict(PARAMS); p_plus[k] = PARAMS[k] * np.exp(eps)
        p_minus = dict(PARAMS); p_minus[k] = PARAMS[k] * np.exp(-eps)
        sol_plus = solve_for_params(p_plus)
        sol_minus = solve_for_params(p_minus)
        jac_G[:, j] = (sol_plus[obs_idx, 2] - sol_minus[obs_idx, 2]) / (2.0 * eps * std_G)
        jac_T1[:, j] = (sol_plus[-1, :] - sol_minus[-1, :]) / (2.0 * eps * anchor_scales)

    def analyze_matrix(J: np.ndarray, names: list[str], label: str) -> dict:
        F = J.T @ J
        eigvals, eigvecs = np.linalg.eigh(F)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        col_norms = {names[idx]: round(float(np.linalg.norm(J[:, idx])), 4) for idx in range(len(names))}
        sloppy_modes = []
        for mode_idx in range(len(names) - 1, max(-1, len(names) - 6), -1):
            vec = eigvecs[:, mode_idx]
            top_idx = np.argsort(np.abs(vec))[::-1][:5]
            sloppy_modes.append({
                "mode_rank_from_top": mode_idx + 1,
                "eigenvalue": float(eigvals[mode_idx]),
                "top_loadings": [
                    {"param": names[idx], "weight": round(float(vec[idx]), 4)}
                    for idx in top_idx
                ],
            })
        return {
            "label": label,
            "condition_number": float(eigvals[0] / max(eigvals[-1], 1e-30)),
            "eigenvalues": [float(v) for v in eigvals],
            "log10_eigenvalues": [round(float(np.log10(max(v, 1e-30))), 4) for v in eigvals],
            "marginal_sensitivity_norms": col_norms,
            "smallest_5_modes": sloppy_modes,
        }

    t1_idx = [i for i, n in enumerate(param_names) if n not in ("Vp", "Vi", "Vg")]
    t1_names = [param_names[i] for i in t1_idx]
    return {
        "n_obs": n_obs,
        "seed": seed,
        "eps": eps,
        "odeint_rtol": 1e-10,
        "odeint_atol": 1e-12,
        "param_names": param_names,
        "G_only_fim": analyze_matrix(jac_G, param_names, "Test 2 (20x20) G-only observation FIM (600 pts)"),
        "G_plus_T1_fim": analyze_matrix(np.vstack([jac_G, jac_T1]), param_names, "Test 2 (20x20) G + T1 (1800 min) anchor FIM (606 pts)"),
        "test1_G_only_fim": analyze_matrix(jac_G[:, t1_idx], t1_names, "Test 1 (17x17, Vp/Vi/Vg fixed) G-only FIM (600 pts)"),
        "test1_G_plus_T1_fim": analyze_matrix(np.vstack([jac_G[:, t1_idx], jac_T1[:, t1_idx]]), t1_names, "Test 1 (17x17, Vp/Vi/Vg fixed) G + T1 FIM (606 pts)"),
    }


def main():
    out_dir = Path(__file__).resolve().parent / "outputs" / "step 3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Running Tight-Tolerance 20x20 & 17x17 FIM Identifiability Check ===")
    fim_res = compute_fim_identifiability(n_obs=600, seed=42, eps=1e-4)
    (out_dir / "fim_identifiability.json").write_text(json.dumps(fim_res, indent=2), encoding="utf-8")
    g_fim = fim_res["G_only_fim"]
    gt1_fim = fim_res["G_plus_T1_fim"]
    print(f"20x20 G-only FIM lambda_max={g_fim['eigenvalues'][0]:.3e}, lambda_min={g_fim['eigenvalues'][-1]:.3e}, cond={g_fim['condition_number']:.3e}")
    print("20x20 G-only smallest 4 modes:")
    for m in g_fim["smallest_5_modes"][:4]:
        print(" ", m)
    print(f"17x17 Test1 G-only lambda_min={fim_res['test1_G_only_fim']['eigenvalues'][-1]:.3e}")
    print("17x17 Test1 G-only smallest 2 modes:")
    for m in fim_res["test1_G_only_fim"]["smallest_5_modes"][:2]:
        print(" ", m)

    print("\n=== 2. Running Matched 600-pt Step A (Test 2, Naive 12k Scale, No Stage 1.5) ===")
    step_a = run_inverse_experiment(
        test_mode="test2",
        seed=42,
        n_obs=600,
        stage2_epochs=16000,
        save_plots=False,
        out_dir=out_dir,
        use_zscore=False,
        stage15_epochs=0,
    )

    print("\n=== 3. Running Matched 600-pt Step B (Test 2, + Z-score, No Stage 1.5) ===")
    step_b = run_inverse_experiment(
        test_mode="test2",
        seed=42,
        n_obs=600,
        stage2_epochs=16000,
        save_plots=False,
        out_dir=out_dir,
        use_zscore=True,
        stage15_epochs=0,
    )

    step_c2 = json.loads((out_dir / "step3_results_test2.json").read_text(encoding="utf-8"))[0]
    ablation_payload = {
        "protocol": {"mode": "test2", "free_params": 20, "n_obs": 600, "seed": 42, "stage1_epochs": 3000, "stage2_epochs": 16000},
        "step_a_naive_12k": step_a,
        "step_b_zscore_no_s15": step_b,
        "step_c2_zscore_plus_s15": step_c2,
    }
    abl_path = out_dir / "ablation_600pt_test2.json"
    abl_path.write_text(json.dumps(ablation_payload, indent=2), encoding="utf-8")
    print(f"\nSaved Phase 1 outputs:\n  {out_dir / 'fim_identifiability.json'}\n  {abl_path}")


if __name__ == "__main__":
    main()

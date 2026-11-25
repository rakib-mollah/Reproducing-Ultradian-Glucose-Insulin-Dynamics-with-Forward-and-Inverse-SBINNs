# Reproducing-Ultradian-Glucose-Insulin-Dynamics-with-Forward-and-Inverse-SBINNs

Pure PyTorch reproduction and structural identifiability analysis of the 6-state **Ultradian glucose–insulin endocrine SBINN** benchmark (Yazdani et al., *PLOS Computational Biology*, 2020), covering forward ODE verification, forward SBINN, inverse parameter/hidden-state recovery (`Vol.-Fixed` 17-param & `All-20` 20-param regimes), Fisher Information Matrix (FIM) identifiability diagnostics, and matched ablations.

## Repository Structure

```text
.
├── step1_verify_ode.py          # 6-ODE exact solver (scipy.integrate.odeint) on [0, 1800] min
├── step2_forward_sbinn.py       # Forward SBINN (Hard IC, Fourier features, causal weights, fwAD)
├── step3_inverse_sbinn.py       # Inverse SBINN (Test 1 Vol.-Fixed, Test 2 All-20, Fig. 8 rollout)
├── run_phase1_diagnostics.py    # 20x20 & 17x17 FIM spectrum + matched 600-pt Step A/B/C ablation
├── generate_slide_visuals.py    # Plots analytical & comparison figures from outputs/ JSONs
├── requirements.txt
└── outputs/
    ├── step 1/                  # Exact ODE trajectory JSON & plot
    ├── step 2/                  # Forward SBINN results JSON & plot
    ├── step 3/                  # Test 1/2 results, FIM JSON, ablation JSON & forecast plots
    └── slide visuals/           # Parameter ranking, FIM spectrum, forecast overlay & ablation PNGs
```

## Setup & Usage

```bash
pip install -r requirements.txt

# 1. Exact 6-state ODE ground truth (1,801 points over [0, 1800] min)
python step1_verify_ode.py

# 2. Forward SBINN (all 20 parameters known)
python step2_forward_sbinn.py

# 3. Inverse SBINN from 600 G-only observations + endpoint anchors (Test 1 & Test 2)
python step3_inverse_sbinn.py --mode both --seeds 42 --n_obs 600 --epochs 16000

# Optional: regenerate Fig. 8 rollout plot ([0, 3000] min) without retraining
python step3_inverse_sbinn.py --mode both --seeds 42 --plot-forecast-only

# 4. FIM identifiability spectrum & matched Step A / B / C ablation (Test 2)
python run_phase1_diagnostics.py

# 5. Generate analytical comparison figures (reads outputs/, writes to ../slides/assets updated/)
python generate_slide_visuals.py
```

---

## Stepwise Results & Outputs

### Step 1 — Exact 6-State ODE Ground Truth (`step1_verify_ode.py`)

![Step 1 Exact Ultradian ODE Truth](outputs/step%201/step1_ultradian_truth.png)

- Solves the 6-state Ultradian system ($I_p, I_i, G, h_1, h_2, h_3$) over $t \in [0, 1800]\text{ min}$ (`1,801` points, `rtol=1e-10, atol=1e-12`) with three meals (`300 min / 60 g`, `650 min / 40 g`, `1100 min / 50 g`).
- Reproduces post-meal glucose spikes (`187.4`, `162.7`, `177.8 mg/dl`) and damped closed-loop insulin/hepatic-delay oscillations (`Ip` up to `369.1 mU`) before settling toward baseline (`G = 106.4 mg/dl` at `1800 min`).
- **Artifacts:** [`outputs/step 1/step1_ultradian_truth.json`](outputs/step%201/step1_ultradian_truth.json), [`step1_ultradian_truth.png`](outputs/step%201/step1_ultradian_truth.png).

---

### Step 2 — Forward SBINN (`step2_forward_sbinn.py`)

![Step 2 Forward SBINN](outputs/step%202/step2_forward_sbinn.png)

- Learns all 6 states given **known ODE parameters** using a `4 × 128` `SiLU` MLP with a Hard-IC ansatz, non-$2\pi$-periodic Fourier features (`freqs = [1..5, 8.3, 13.7, 21.1, 31.4, 45.7]`), 18-bin exponential causal weighting, and forward-mode AD (`fwAD`) over `12,000` Adam epochs (`853.6 s` CPU).
- **Relative $L_2$ error vs. `odeint`:** `G = 0.70%`, `Ip = 1.92%`, `Ii = 1.02%`, `h1 = 1.64%`, `h2 = 1.42%`, `h3 = 1.27%`.
- **Artifacts:** [`outputs/step 2/step2_forward_results.json`](outputs/step%202/step2_forward_results.json), [`step2_forward_sbinn.png`](outputs/step%202/step2_forward_sbinn.png).

---

### Step 3 — Inverse SBINN: Hidden States, Parameters & Forecast (`step3_inverse_sbinn.py`)

From only **`600` scattered glucose $G$ measurements** plus exact 6-state endpoint anchors ($T_0 = 0, T_1 = 1800\text{ min}$), the network jointly infers the 5 unmeasured states ($I_p, I_i, h_1, h_2, h_3$) and unknown ODE parameters under two regimes: **Test 2 (`All-20`, all 20 parameters free)** and **Test 1 (`Vol.-Fixed`, volumes $V_p, V_i, V_g$ fixed, 17 free)**.

#### 3a. 6-State Reconstruction (`All-20` Hero Run)

![Step 3 Inverse States All-20](outputs/step%203/step3_inverse_test2_s42.png)

- **`All-20` Rel. $L_2$:** `G = 0.110%` (observed), `Ip = 12.33%`, `Ii = 8.33%`, `h1 = 11.37%`, `h2 = 10.86%`, `h3 = 10.62%` (`321.3 s` CPU).
- **`Vol.-Fixed` Rel. $L_2$:** `G = 0.115%` (observed), `Ip = 19.34%`, `Ii = 6.50%`, `h1 = 16.86%`, `h2 = 15.45%`, `h3 = 14.63%` (`456.1 s` CPU).
- All hidden oscillations align closely in phase; residual peak undershoot on $I_p$ comes from `50:20:1` data/anchor/ODE loss weighting and $(R_m, t_p)$ / $(C_1, E)$ drift.

<details>
<summary>View Test 1 (<code>Vol.-Fixed</code>, 17 free params) 6-state reconstruction plot</summary>

![Step 3 Inverse States Vol.-Fixed](outputs/step%203/step3_inverse_test1_s42.png)

</details>

#### 3b. Parameter Identification Error (`All-20`)

![All-20 Parameter Recovery Ranking](outputs/slide%20visuals/slide07_all20_params.png)

- **`All-20` achieves `7/20` within `10%`:** `Vp (1.3%)`, `k (3.9%)`, `E (4.2%)`, `C5 (5.0%)`, `td (5.9%)`, `Rg (6.6%)`, `a1 (9.5%)`. Meanwhile `C2 (48.7%)` and `ti (51.2%)` drift freely while holding `G` to `0.110%`.
- **`Vol.-Fixed` achieves `7/17` within `10%`:** sharpens `ti` (`51.2% → 6.9%`) and `Rg` (`6.6% → 1.6%`), along with `a1 (0.5%)`, `beta (3.0%)`, `k (3.2%)`, `C4 (6.3%)`, `td (8.0%)`, while `Rm (52.3%)` and `C2 (34.8%)` still drift.

#### 3c. Out-of-Sample Forecast (`[1800, 3000] min`, Unseen `100 g` Meal at `t = 2000 min`)

![Fig. 8 Forecast Comparison Overlay](outputs/slide%20visuals/slide08_fig8_3panel.png)

- Integrating the inferred parameters forward with `odeint` over `[0, 3000] min` generalizes to the unseen `2000 min` meal on glucose (**`G` post-1800 Rel. $L_2$:** `1.80%` for `All-20`, `2.20%` for `Vol.-Fixed`; exact post-2000 peak `224.0 mg/dl` vs. `224.6` / `221.6 mg/dl`).
- Unmeasured insulin forecasts capture timing accurately (`Ii` `9.12%` in `Vol.-Fixed`, `16.09%` in `All-20`) while underpredicting the unseen $I_p$ peak (`427.0` / `507.8` vs. `617.8 mU` exact).

<details>
<summary>View individual 4-panel rollout plots (with known meal intake <code>I_G</code>) for All-20 and Vol.-Fixed</summary>

![Step 3 Forecast All-20](outputs/step%203/step3_forecast_test2_s42.png)
![Step 3 Forecast Vol.-Fixed](outputs/step%203/step3_forecast_test1_s42.png)

</details>

- **Artifacts:** [`step3_results_test1.json`](outputs/step%203/step3_results_test1.json), [`step3_results_test2.json`](outputs/step%203/step3_results_test2.json), and PNGs in [`outputs/step 3/`](outputs/step%203) & [`outputs/slide visuals/`](outputs/slide%20visuals).

---

### Step 4 — Diagnostics: FIM Identifiability & Matched Ablation (`run_phase1_diagnostics.py`)

#### 4a. Structural & Practical Non-Identifiability (FIM Spectrum)

![FIM Identifiability Spectrum and C2 Saturation](outputs/slide%20visuals/slide07_identifiability.png)

- **Top (`f2(G)` saturation):** Across the observed glucose window (`[8.3k, 18.7k] mg`), $e^{-G/(C_2 V_g)} < 0.004$, leaving $f_2(G) \approx U_b$ flat with near-zero sensitivity to `C2` (FIM eigenvalue `2.09e-4` G-only, `2.22e-4` w/ $T_1$).
- **Bottom (`20×20` vs. `17×17` FIM):** Freeing `(Vp, Vi, Vg)` introduces **two analytical null modes** ($\lambda = 0$; `~1e-8` numerical floor, $\kappa \approx 1.6\times 10^{13}$) unbroken by $T_1$. Fixing the 3 volumes eliminates both null modes ($\lambda_{\min} = 1.61\times 10^{-4}$), though `17×17` remains ill-conditioned ($\kappa = 5.94\times 10^8$).

#### 4b. Matched 600-pt Step A → B → C Ablation (`All-20`, Seed `42`)

![Matched 600-pt Ablation Study](outputs/slide%20visuals/slide09_ablation_and_loss.png)

| Configuration | Epochs | `G` Rel. $L_2$ | `Ip` Rel. $L_2$ | `Ii` Rel. $L_2$ | Mean 20-P Err | `< 10%` Count | 601-pt Obs ODE MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Step A:** Naive `12k` scale | 19k | 0.408% | 19.42% | 16.25% | 26.6% | 3/20 | 0.0527 |
| **Step B:** + Observed Z-score | 19k | 0.193% | 14.03% | 10.49% | 20.3% | 5/20 | 0.0130 (`4.0× ↓`) |
| **Step C:** + Stage 1.5 warmup | 21.5k | **0.110%** | **12.33%** | **8.33%** | **19.2%** | **7/20** | **0.0098** (`5.4× ↓`) |

- **Z-score normalization (`Step A → B`, equal `19k` epochs / `272.9 s`):** Drives the bulk gain, removing a ~`48×` DC-offset gradient suppression on $G$ and cutting the uniform `601-pt` ODE residual `4.0×`.
- **Stage 1.5 (`Step B → C`, `+2.5k` frozen-param ODE ramp / `321.3 s`):** Improves all 6 states monotonically (`G`, `Ip`, `Ii`, and `h1–h3`) and reduces final ODE residual to `0.0098`.
- **Artifacts:** [`fim_identifiability.json`](outputs/step%203/fim_identifiability.json), [`ablation_600pt_test2.json`](outputs/step%203/ablation_600pt_test2.json).

---

## Notes
- `.gitignore` excludes `__pycache__/`, `*.py[cod]`, virtual environments, and OS metadata.
- `step2_forward_sbinn.py` and `step3_inverse_sbinn.py` also save `.npz` array caches to `../results/`, and `generate_slide_visuals.py` writes slide figures to `../slides/assets updated/`.

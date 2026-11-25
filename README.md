# Reproducing-Ultradian-Glucose-Insulin-Dynamics-with-Forward-and-Inverse-SBINNs

Pure PyTorch reproduction and structural identifiability analysis of the 6-state **Ultradian glucose–insulin endocrine SBINN** benchmark (Yazdani et al., *PLOS Computational Biology*, 2020), covering forward ODE simulation, forward SBINN, inverse parameter/hidden-state recovery (`Vol.-Fixed` 17-param & `All-20` 20-param regimes), Fisher Information Matrix (FIM) identifiability diagnostics, and matched ablations.

## Repository Structure

```text
.
├── step1_verify_ode.py          # 6-ODE exact solver (scipy.integrate.odeint) on [0, 1800] min
├── step2_forward_sbinn.py       # Forward SBINN (Hard IC, Fourier features, causal weights, fwAD)
├── step3_inverse_sbinn.py       # Inverse SBINN (Test 1 Vol.-Fixed, Test 2 All-20, Fig. 8 rollout)
├── run_phase1_diagnostics.py    # 20x20 & 17x17 FIM spectrum + matched 600-pt Step A/B/C ablation
├── generate_slide_visuals.py    # Plots figure panels from outputs/ JSONs
├── requirements.txt
└── outputs/
    ├── step 1/                  # Exact ODE trajectory JSON & plot
    ├── step 2/                  # Forward SBINN results JSON & plot
    ├── step 3/                  # Test 1/2 results, FIM JSON, ablation JSON & forecast plots
    └── slide visuals/           # Selected analytical figure PNGs from outputs/step 3 JSONs
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run scripts in pipeline order from the repository root:

```bash
# 1. Generate and verify exact 6-state ODE ground truth (1,801 points)
python step1_verify_ode.py

# 2. Train Forward SBINN (known ODE parameters)
python step2_forward_sbinn.py

# 3. Train Inverse SBINN (Test 1: 17 free params; Test 2: all 20 free params)
python step3_inverse_sbinn.py --mode both --seeds 42 --n_obs 600 --epochs 16000

# Optional: regenerate Fig. 8 forecast plot ([0, 3000] min) from saved JSON without retraining
python step3_inverse_sbinn.py --mode both --seeds 42 --plot-forecast-only

# 4. Run FIM identifiability check & matched Step A / B / C ablation (Test 2)
python run_phase1_diagnostics.py

# 5. Re-generate publication figures (writes to ../slides/assets updated/)
python generate_slide_visuals.py
```

## Notes
- `.gitignore` excludes `__pycache__/`, `*.py[cod]`, virtual environments, and OS metadata.
- Note that `step2_forward_sbinn.py` and `step3_inverse_sbinn.py` also save `.npz` array caches to `../results/`, and `generate_slide_visuals.py` writes slide figures to `../slides/assets updated/` relative to the parent directory.

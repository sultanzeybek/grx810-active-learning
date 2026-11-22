# Hybrid-GPR and cost-aware CEI for L-PBF process-parameter selection (GRX-810)

Code and data for:

S. Zeybek, G. Tarakçı, M. Coşkun, E. Koç, Y. Akyıldız, G. Özer.
*A closed-loop active learning framework for laser powder bed fusion calibration under data scarcity.*
Journal of Intelligent Manufacturing (under review).

The repository contains the eleven-build dataset, the Hybrid-GPR model and the scripts that
produce the Gaussian-process results and the code-generated figures of the paper.

## Repository layout

| Path | Content |
|---|---|
| `builds.csv` | Eleven builds on an EOS M290: seven seed builds (S1–S7) and four follow-up builds (A, B, C, D). Process parameters, Archimedes density, HV1 mean, SD and indent count, and optical area fraction. |
| `hardness_indents.csv` | The individual HV1 readings: five indents per build and ten for build A. `reproduce_main.py` checks the build means and gives the pooled single-indent SD (8.8 HV for the seed builds). |
| `hybrid_gpr.py` | The model of Sect. 2: five inputs (P, v, h, VED, Ċ), physics prior means (Table 2), constant offset β, Matérn-5/2 residual GP with ℓ ≥ 0.5, fixed noise scales, probability of feasibility (Eq. 9) and cost-aware CEI (Eq. 10). |
| `reproduce_main.py` | LOOCV errors (Table 7), sensitivity to the lower density asymptote ρ_min, per-fold hyperparameters (Table A1), feasible region (Table 8), follow-up test and model update (Table 10), sample-size arithmetic and the hardness noise. |
| `paired_errors_prior_centre.py` | Paired LOOCV errors and effect sizes (Fig. 7), and the prior-centre comparison on three datasets (Table A2, Fig. 8). |
| `process_window.py` | Feasible-region comparison on the 10,000-candidate grid and its 151-candidate seed-hull subset (Tables 8 and 9), and Figs. 9, 10 and 12. |
| `active_learning_replay.py` | Active-learning replay on a common test set (Sect. 4.8, Fig. 15) and LOOCV with priors re-derived inside each fold (Table 7). |
| `noise_and_next_build.py` | Sensitivity to the density noise and the prior width (Table A3), the density-correction scenario (Sect. 5.3) and the virtual twelfth build (Table A4, Sect. 5.4). |
| `initial_surrogate.py` | The initial surrogate that selected builds A and C, fitted to the seven seed builds, and its prediction at the four follow-up builds (Sect. 4.7.1). |
| `fig_followup_test.py` | Fig. 13 (predictions of both GP models at the four follow-up builds). |
| `fig_framework.py` | Fig. 1 (vector drawing). |
| `results/` | All outputs of the scripts: tables as CSV, console summaries as TXT, figures as PDF, SVG and PNG. |
| `run_all.sh` | Runs every script and writes the outputs to `results/`. |

## Model in brief

    y_i = m(x_i) + β + δ(x_i) + ε_i,   δ ~ GP(0, k_Matérn5/2),   ε_i ~ N(0, s_i²) with s_i fixed

The noise scales are fixed before fitting. Density uses an effective build-level SD of
0.08 g/cm³, supported by the replicate pair S2/B and by the spread of the eleven builds.
Hardness uses 8.8/√n HV for an n-indent mean, from the pooled single-indent SD of the seed
builds. Optical porosity uses 0.27 decades for a single field, from the paired fields of the
follow-up builds and the S2/B replicate. With seven builds the marginal likelihood cannot separate signal from
noise. The initial surrogate learned a single noise variance close to zero, and at the
replicated setting (build B) the measured density fell outside its 95% predictive interval.
Density is the objective of the acquisition; its threshold sets the lower limit of the
improvement integral, and the hardness constraint enters as a probability.

## Run

    pip install -r requirements.txt
    bash run_all.sh               # Gaussian-process and prior-only results
    bash run_all.sh --with-trees  # also refits the random-forest and XGBoost reference rows of Table 8

Random seeds are fixed (candidate grid seed 0, optimiser restarts seed 0). A full run takes
under a minute on a laptop CPU. The files in `results/` were generated with Python 3.13,
NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, Matplotlib 3.11.2, scikit-learn 1.9.1 and
XGBoost 3.0.2. Last-digit differences in fitted GP parameters can occur across numerical
environments.

## Not included

The micrographs (Figs. 3, 5, 6 and 14), the micro-CT indication lists and the CT reports
are available from the corresponding author on request. The tree-ensemble and SVR
baselines of the LOOCV comparison in Table 7 are reported values and are not refitted here.

## Initial surrogate

The initial surrogate that selected builds A and C uses six inputs, learns its noise from
the data and applies a hard feasibility mask. `initial_surrogate.py` contains its model
code unchanged and prints its predictions at A, B, C and D. The original seven-build
version of the repository is kept in the commit history.

## Citation

See `CITATION.cff`.

## License

MIT (see `LICENSE`).

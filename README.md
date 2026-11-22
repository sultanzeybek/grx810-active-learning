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
| `builds.csv` | Eleven builds on an EOS M290: seven seed builds (S1–S7) and four follow-up builds (A, B, C, D). Process parameters, Archimedes density, HV1 mean, SD and indent count, optical area fraction, and the laboratory label of each build. |
| `hybrid_gpr.py` | The model of Sect. 2: five inputs (P, v, h, VED, Ċ), physics prior means (Table 2), constant offset β, Matérn-5/2 residual GP with ℓ ≥ 0.5, fixed noise scales, probability of feasibility (Eq. 9) and cost-aware CEI (Eq. 10). |
| `reproduce_main.py` | LOOCV errors (Table 7), sensitivity to the lower density asymptote ρ_min, per-fold hyperparameters (Table A1), feasible region (Table 8), follow-up test and model update (Table 10), sample-size arithmetic and the value of a twelfth build. |
| `paired_errors_prior_centre.py` | Paired LOOCV errors and effect sizes (Fig. 7), and the prior-centre comparison on three datasets (Table A2, Fig. 8). |
| `process_window.py` | Feasible-region comparison on the 10,000-candidate grid and its 151-candidate seed-hull subset (Tables 8 and 9), and Figs. 9, 10 and 12. |
| `active_learning_replay.py` | Active-learning replay on a common test set (Sect. 4.8, Fig. 15) and LOOCV with priors re-derived inside each fold (Table 7). |
| `fig_framework.py` | Fig. 1 (vector drawing). |
| `results/` | All outputs of the scripts: tables as CSV, console summaries as TXT, figures as PDF, SVG and PNG. |
| `run_all.sh` | Runs every script and writes the outputs to `results/`. |

## Model in brief

    y_i = m(x_i) + β + δ(x_i) + ε_i,   δ ~ GP(0, k_Matérn5/2),   ε_i ~ N(0, s_i²) with s_i fixed

The noise scales are fixed from replicate measurements: density uses an effective build-level
SD of 0.08 g/cm³, hardness 8.8/√n HV for an n-indent mean, and optical porosity 0.27 decades
for a single field. With seven builds the marginal likelihood cannot separate signal from
noise. The initial surrogate learned a single noise variance close to zero, and at the
replicated setting (build B) the measured density fell outside its 95% predictive interval.
Density is the objective of the acquisition; its threshold sets the lower limit of the
improvement integral, and the hardness constraint enters as a probability.

## Run

    pip install -r requirements.txt
    ./run_all.sh               # Gaussian-process and prior-only results
    ./run_all.sh --with-trees  # also refits the random-forest and XGBoost reference rows of Table 8

Random seeds are fixed (candidate grid seed 0, optimiser restarts seed 0). A full run takes
under a minute on a laptop CPU. The files in `results/` were generated with Python 3.13,
NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, Matplotlib 3.11.2, scikit-learn 1.9.1 and
XGBoost 3.0.2. Last-digit differences in fitted GP parameters can occur across numerical
environments.

## Laboratory labels

The laboratory files use their own names. Build A appears as 8 and build D appears as A,
so file names must not be read as paper labels. The column `lab_label` in `builds.csv`
gives the laboratory name of each build.

| Paper | Hardness and density sheet | Optical files (ZEN) | Micro-CT report | Tensile folder |
|---|---|---|---|---|
| S1–S7 | 1–7 (S2 also as 2=B) | 1–7 | Scan 1–7 | 1–7 |
| A | 8 (Validasyon) | not in the shared files | Report - 8 | 8 |
| B | 2=B | GRX B-1, B-2 | Report - B | b |
| C | C | GRX C-1, C-2 | Report - C | c |
| D | A (E in a later summary sheet) | GRX A-1, A-2 | Report - A | a |

For the follow-up builds, field 1 is the upper region and field 2 the lower region of the
polished section. The tensile results are not used in the paper.

## Not included

The micrographs (Figs. 3, 5, 6 and 14), the micro-CT indication lists and the CT reports
are available from the corresponding author on request. The tree-ensemble and SVR
baselines of the LOOCV comparison in Table 7 are reported values and are not refitted here.

## Initial surrogate

The initial surrogate that selected builds A and C uses six inputs, learns its noise from
the data and applies a hard feasibility mask. Its implementation is kept at
[commit 50172b8](https://github.com/sultanzeybek/grx810-active-learning/tree/50172b8121545f7be053f4910ed8280bdf178b0c).

## Citation

See `CITATION.cff`.

## License

MIT (see `LICENSE`).

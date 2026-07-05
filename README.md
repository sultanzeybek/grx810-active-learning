# A closed-loop active learning framework for laser powder bed fusion calibration under data scarcity
Physics-informed Hybrid Gaussian Process Regression (Hybrid-GPR) with Constrained
Expected Improvement (CEI) for data-efficient laser powder bed fusion (L-PBF)
process calibration under severe data scarcity, demonstrated on NASA's GRX-810
oxide-dispersion-strengthened superalloy.

This repository accompanies the paper:

> S. Zeybek, G. Tarakçı, M. Coşkun, E. Koç, Y. Akyildiz, G. Özer,
> "A closed-loop active learning framework for laser powder bed fusion calibration under data scarcity",
> *Journal of Intelligent Manufacturing* (under review).

## What is here

```
.
├── data/
│   └── builds.csv              # the seven-build seed dataset (EOS M290)
├── src/
│   ├── grx810_framework.py     # core library: Hybrid-GPR, priors, CEI, Pareto
│   ├── reproduce_analyses.py   # reproduces all numerical results in the paper
│   └── make_figures.py         # regenerates the analysis figures (300 dpi)
├── figures/                    # generated figures
├── requirements.txt
└── LICENSE
```

## Method in brief

The surrogate decomposes each target as a fixed physics-informed prior mean plus a
zero-mean residual Gaussian process (Kennedy-O'Hagan):

    f(x) = mu_prior(x) + delta(x),   delta ~ GP(0, k)

with a Matern-5/2 kernel. The prior mean encodes the established process-property
physics of GRX-810 (parabolic density-vs-VED, Hall-Petch hardness, parabolic
porosity). Under severe data scarcity (N = 7) the prior mean, not the input
features, is the dominant design lever: it lets the surrogate locate a feasible
process window where a zero-mean GP finds none. The decision layer combines
empirical Pareto-front extraction with Cost-Aware CEI to recommend the next builds.

## Reproducing the results

```bash
pip install -r requirements.txt
python src/reproduce_analyses.py     # prints the tables and key numbers
python src/make_figures.py           # writes figures/ at 300 dpi
```

`reproduce_analyses.py` reproduces the LOOCV tournament, the feature ablation
(<1% across all targets), the Hybrid-vs-vanilla comparison, the VED-porosity
correlation, the cross-laboratory density check (0.204% relative error), the
cost-coefficient sensitivity, and the desirability single-point recommendation.

## Dataset

`data/builds.csv` contains the seven GRX-810 builds produced on an EOS M290 at the
ALUTEAM facility: laser power, scan speed, hatch spacing, layer thickness,
Archimedes density, Vickers microhardness (mean and standard deviation), and
ZEISS cross-sectional porosity. Physics-derived features (VED, cooling-rate index,
thermal-input index) are computed in `load_data`.

## Citation

If you use this code or data, please cite the paper above. A BibTeX entry will be
added on publication.

## License

MIT License (see `LICENSE`).

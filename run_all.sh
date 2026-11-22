#!/usr/bin/env bash
# Reproduce all numerical results and code-generated figures into results/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
cd results
python ../reproduce_main.py              > reproduce_output.txt
python ../paired_errors_prior_centre.py  > paired_errors_output.txt
python ../process_window.py "$@"         > window_output.txt
python ../active_learning_replay.py      > replay_output.txt
python ../initial_surrogate.py           > initial_surrogate_output.txt
python ../noise_and_next_build.py        > noise_and_next_build_output.txt
python ../fig_followup_test.py
python ../fig_framework.py
echo "Done. Outputs are in results/."

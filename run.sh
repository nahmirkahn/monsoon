#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/miso/Documents/WINDOWS/monsoon/final_solution"
SCRIPTS="$ROOT/scripts"
cd "$SCRIPTS"

# Final minimal flow using scripts and artifacts directories
python3 build_and_train.py | cat
python3 plot_all.py | cat
python3 shap_analysis.py | cat

echo "[final_solution] Completed. Check outputs in $ROOT"

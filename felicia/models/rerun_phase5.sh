#!/usr/bin/env bash
set -euo pipefail

PY="/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python"
export PYTHONWARNINGS=ignore

$PY felicia/models/models.py --method method1 --model logistic --output-json felicia/results/evaluation_metrics/method1_logistic.json
$PY felicia/models/models.py --method method1 --model rf --output-json felicia/results/evaluation_metrics/method1_rf.json
$PY felicia/models/models.py --method method2 --model logistic --output-json felicia/results/evaluation_metrics/method2_logistic.json
$PY felicia/models/models.py --method method2 --model rf --output-json felicia/results/evaluation_metrics/method2_rf.json
$PY felicia/models/models.py --method method3_2 --model logistic --output-json felicia/results/evaluation_metrics/method3_2_logistic.json
$PY felicia/models/models.py --method method3_2 --model rf --output-json felicia/results/evaluation_metrics/method3_2_rf.json
$PY felicia/models/models.py --method method4 --model logistic --resampler smote --output-json felicia/results/evaluation_metrics/method4_logistic_smote.json
$PY felicia/models/models.py --method method4 --model logistic --resampler borderline_smote --output-json felicia/results/evaluation_metrics/method4_logistic_borderline_smote.json
$PY felicia/models/models.py --method method4 --model logistic --resampler adasyn --output-json felicia/results/evaluation_metrics/method4_logistic_adasyn.json
$PY felicia/models/models.py --method method4 --model rf --resampler smote --output-json felicia/results/evaluation_metrics/method4_rf_smote.json
$PY felicia/models/models.py --method method4 --model rf --resampler borderline_smote --output-json felicia/results/evaluation_metrics/method4_rf_borderline_smote.json
$PY felicia/models/models.py --method method4 --model rf --resampler adasyn --output-json felicia/results/evaluation_metrics/method4_rf_adasyn.json

printf "All reruns completed.\n"

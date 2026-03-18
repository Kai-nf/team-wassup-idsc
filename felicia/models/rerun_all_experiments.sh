#!/bin/bash
# Re-run all 13 experiments with latest Phase 5 metrics + per-fold sample counts

cd /Users/a1357/Documents/GitHub/team-wassup-idsc

export PYTHONWARNINGS=ignore
PYTHON="/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python"

echo "=== Running All Phase 5 Experiments ==="
echo "Timestamp: $(date)"
echo ""

# Method 1
echo "[1/13] method1 + logistic..."
$PYTHON felicia/models/models.py --method method1 --model logistic --output-json felicia/results/evaluation_metrics/method1_logistic.json 2>&1 | tail -1

echo "[2/13] method1 + rf..."
$PYTHON felicia/models/models.py --method method1 --model rf --output-json felicia/results/evaluation_metrics/method1_rf.json 2>&1 | tail -1

# Method 2
echo "[3/13] method2 + logistic..."
$PYTHON felicia/models/models.py --method method2 --model logistic --output-json felicia/results/evaluation_metrics/method2_logistic.json 2>&1 | tail -1

echo "[4/13] method2 + rf..."
$PYTHON felicia/models/models.py --method method2 --model rf --output-json felicia/results/evaluation_metrics/method2_rf.json 2>&1 | tail -1

# Method 3.2
echo "[5/13] method3_2 + logistic..."
$PYTHON felicia/models/models.py --method method3_2 --model logistic --output-json felicia/results/evaluation_metrics/method3_2_logistic.json 2>&1 | tail -1

echo "[6/13] method3_2 + rf..."
$PYTHON felicia/models/models.py --method method3_2 --model rf --output-json felicia/results/evaluation_metrics/method3_2_rf.json 2>&1 | tail -1

# Method 4 + SMOTE
echo "[7/13] method4 + logistic + smote..."
$PYTHON felicia/models/models.py --method method4 --model logistic --resampler smote --output-json felicia/results/evaluation_metrics/method4_logistic_smote.json 2>&1 | tail -1

echo "[8/13] method4 + logistic + borderline_smote..."
$PYTHON felicia/models/models.py --method method4 --model logistic --resampler borderline_smote --output-json felicia/results/evaluation_metrics/method4_logistic_borderline_smote.json 2>&1 | tail -1

echo "[9/13] method4 + logistic + adasyn..."
$PYTHON felicia/models/models.py --method method4 --model logistic --resampler adasyn --output-json felicia/results/evaluation_metrics/method4_logistic_adasyn.json 2>&1 | tail -1

echo "[10/13] method4 + rf + smote..."
$PYTHON felicia/models/models.py --method method4 --model rf --resampler smote --output-json felicia/results/evaluation_metrics/method4_rf_smote.json 2>&1 | tail -1

echo "[11/13] method4 + rf + borderline_smote..."
$PYTHON felicia/models/models.py --method method4 --model rf --resampler borderline_smote --output-json felicia/results/evaluation_metrics/method4_rf_borderline_smote.json 2>&1 | tail -1

echo "[12/13] method4 + rf + adasyn..."
$PYTHON felicia/models/models.py --method method4 --model rf --resampler adasyn --output-json felicia/results/evaluation_metrics/method4_rf_adasyn.json 2>&1 | tail -1

# Bonus: method1 + logistic test
echo "[13/13] Building comparison tables..."
$PYTHON felicia/models/build_comparison_table.py 2>&1 | grep "✓"

echo ""
echo "=== All experiments complete! ==="
echo "Timestamp: $(date)"

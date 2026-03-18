# Unified Execution Plan (Felicia Scope)

This workflow integrates Phase 3 augmentation/resampling into Phase 4 model training.

## Hard Rules

1. Use only `master_folds_drop14.json` for outer 5-fold evaluation splits.
2. Never augment or resample test folds.
3. Domain augmentation is only for Methods 1-3 training positives.
4. SMOTE / Borderline-SMOTE / ADASYN is only for Method 4 training features.

## Step-by-Step

1. Verify fold source file exists at workspace root: `master_folds_drop14.json`.
2. Run Method 2 first (priority P0) with logistic and random forest.
3. Run Method 4 next (priority P1) with logistic and random forest across 3 resamplers.
4. Run Method 3 family (priority P2), then Method 1 sanity run (priority P3).
5. Compare `summary` metrics across runs.

## Commands

Run from workspace root.

### P0: Method 2

```bash
/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python felicia/models/models.py \
  --method method2 \
  --model logistic \
  --fold-file master_folds_drop14.json \
  --output-json felicia/results/evaluation_metrics/method2_logistic.json

/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python felicia/models/models.py \
  --method method2 \
  --model rf \
  --fold-file master_folds_drop14.json \
  --output-json felicia/results/evaluation_metrics/method2_rf.json
```

### P1: Method 4

```bash
/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python felicia/models/models.py \
  --method method4 --model logistic --resampler smote \
  --fold-file master_folds_drop14.json \
  --method4-csv Steve/dataset_v4_features_drop14.csv \
  --output-json felicia/results/evaluation_metrics/method4_logistic_smote.json

/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python felicia/models/models.py \
  --method method4 --model logistic --resampler borderline_smote \
  --fold-file master_folds_drop14.json \
  --method4-csv Steve/dataset_v4_features_drop14.csv \
  --output-json felicia/results/evaluation_metrics/method4_logistic_borderline_smote.json

/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python felicia/models/models.py \
  --method method4 --model logistic --resampler adasyn \
  --fold-file master_folds_drop14.json \
  --method4-csv Steve/dataset_v4_features_drop14.csv \
  --output-json felicia/results/evaluation_metrics/method4_logistic_adasyn.json

/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python felicia/models/models.py \
  --method method4 --model rf --resampler smote \
  --fold-file master_folds_drop14.json \
  --method4-csv Steve/dataset_v4_features_drop14.csv \
  --output-json felicia/results/evaluation_metrics/method4_rf_smote.json

/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python felicia/models/models.py \
  --method method4 --model rf --resampler borderline_smote \
  --fold-file master_folds_drop14.json \
  --method4-csv Steve/dataset_v4_features_drop14.csv \
  --output-json felicia/results/evaluation_metrics/method4_rf_borderline_smote.json

/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python felicia/models/models.py \
  --method method4 --model rf --resampler adasyn \
  --fold-file master_folds_drop14.json \
  --method4-csv Steve/dataset_v4_features_drop14.csv \
  --output-json felicia/results/evaluation_metrics/method4_rf_adasyn.json
```

### P2: Method 3 Family

```bash
/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python felicia/models/models.py \
  --method method3_2 --model logistic \
  --fold-file master_folds_drop14.json \
  --output-json felicia/results/evaluation_metrics/method3_2_logistic.json

/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python felicia/models/models.py \
  --method method3_2 --model rf \
  --fold-file master_folds_drop14.json \
  --output-json felicia/results/evaluation_metrics/method3_2_rf.json
```

### P3: Method 1 Sanity

```bash
/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python felicia/models/models.py \
  --method method1 --model logistic \
  --fold-file master_folds_drop14.json \
  --output-json felicia/results/evaluation_metrics/method1_logistic.json

/Users/a1357/Documents/GitHub/team-wassup-idsc/.venv/bin/python felicia/models/models.py \
  --method method1 --model rf \
  --fold-file master_folds_drop14.json \
  --output-json felicia/results/evaluation_metrics/method1_rf.json
```

## Output Contract

Each JSON output includes:

1. `fold_source` (must be `master_folds_drop14.json`)
2. `fold_source_sha256`
3. `n_splits`, `random_state`
4. `fold_results` (metrics + best params per fold)
5. `summary` (mean/std over outer folds)
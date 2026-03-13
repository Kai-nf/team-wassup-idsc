# Data Integrity Check Report

## Dataset: `metadata.csv`

### Executive summary

- Total rows (tuples): 363
- Total columns (variables): 4
- Overall Status: 

## 1. ✅ Duplicate IDs Check

Status: PASSED

- No duplicated `patient_id` was found
- `patient_id` sequence is uncontinuous but increasing (188,981 to 3,139,196)

## 2. ✅ Missing Value Check

Status: PASSED

- No missing value for all 4 variables (`patient_id`, `basal_pattern`, `sudden_death`, and `brugada`) for all tuples

## 3. ✅ Value Range Check

Status: PASSED

- All variables are categorical
- Outliers undetected as of the presence of numerical columns

### Categorical colums

- `basal_patern`: [0 to 1]
- `sudden_death`: [0 to 1]
- `brugada`: [0 to 2]

## 4. ⚠️ Illogical Combinations Check

Status: ISSUE FOUND

### Variable combination table

| `basal_pattern` | `sudden_death` | `brugada` | Percentage | Explanation |
| - | - | - | - | - |
| 0 | 0 | 0 | 71.34% | **Logical**: Healthy control subject with no pattern and symptom. |
| 0 | 1 | 0 | 1.38% | **Questionable**: A healthy subject who died suddenly out of no suspected reason, but it remains logically sound |
| 0 | 0/1 | 1 | 13.77% | **Logical**: A "concealed" Brugada case not pre-diagnosed whether or not the subject has gone through the fatal event |
| 0 | 0/1 | 2 | 0.83% | **Logical**: Patient diagnosed with other disease |
| 1 | 0/1 | 0 | 6.33% | ⚠️**Illogical**: The subject shows Brugada-like baseline pattern but is labeled as "Normal" |
| 1 | 0/1 | 1 | 5.23% | **Logical**: The subject shows Brugada-like baseline and is diagnosed with Brugada syndrome, no matter whether he has experienced the fatal event
| 1 | 0/1 | 2 | 1.10% | ⚠️**Illogical**: It's illogical to diagnose the patient as normal after showing a Brugada-like baseline |

### Basal pattern-brugada diagnosis mismatch

**Analysis**:
- A baseline pattern of 1 normally indicates that the observed subject exhibits a Brugada-positive pattern in their ECG waveform.
- It's a mistaken logical reasoning or a data mismatching because a baseline pattern of 1 certainly point to a brugada diagnosis of 1.
- 6.33% (23 subjects) in the dataset show the mismatched diagnosis to "Normal" case for patient with known Brugada-like baseline pattern.
- 1.10% (4 subjects) in the dataset show the mismatched diagnosis to "Others" case for patient with known Brugada-like baseline pattern.

**Logical Check Score**: 92.56% (Logical)

## 5. 📄 Data Quality Score
- **Duplicate IDs**: ✅ 100% (No duplicated `patient_id`)
- **Missing Values**: ✅ 100% (No missing value)
- **Value Ranges**: ✅ 100% (no outliers as for all categorical variables)
- **Logical Consistency**: ⚠️ 92.56% (27 potentially illogical combinations)

**Overall Data Quality**: **Good** with minor issues that should be addressed before modeling.
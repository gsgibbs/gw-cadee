# GW-CADEE: Distance Correlation-Based Mutual Information Estimation

**Reproducible Code for Dissertation Analysis**

This repository contains all code, data, and analysis scripts to reproduce the statistical comparison of CADEE, GW-CADEE, and Normalized MI across three datasets.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Datasets](#datasets)
5. [Reproduction Steps](#reproduction-steps)
6. [Project Structure](#project-structure)
7. [Key Results](#key-results)
8. [Citation](#citation)

---

## Overview

This repository implements and compares three mutual information estimation algorithms:

1. **CADEE** - Copula-based Adaptive Differential Entropy Estimator (Spearman correlation)
2. **GW-CADEE** - Our contribution using distance correlation instead of Spearman
3. **NMI** - Normalized Mutual Information using k-NN (KSG estimator)

**Main Finding**: GW-CADEE detected 98.0% of dependencies vs CADEE's 66.7% across 51 feature pairs (p<0.001, Cohen's h=0.950, large effect).

---

## Requirements

### Python Version
- Python 3.8 or higher

### Required Packages
```bash
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```


---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/gw-cadee.git
cd gw-cadee
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Datasets

### Dataset 1: Ariel Synthetic (11 pairs)
**Source**: Generated programmatically  or ariel_synthetic_dataset.csv
**Purpose**: Controlled validation with known ground truth  
**Generation**: Run `generate_ariel_data()` function in `REPRODUCIBLE_CODE.py`  
**Composition**:
- 5 monotonic patterns (linear, exponential, logarithmic, cubic)
- 6 non-monotonic patterns (quadratic, sine, circle, X-shape, step, parabolic)
- 1 independent pattern (null control)

### Dataset 2: Wisconsin Breast Cancer (30 pairs)
**Source**: UCI Machine Learning Repository  
**File**: `wisconsin_breast_cancer.csv`  
**Download**: [UCI ML Repo - Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
**Purpose**: Real biomedical data  
**Samples**: 569 patients  
**Features**: 30 tumor measurements  
**Selected Pairs**: 15 monotonic + 15 non-monotonic (balanced)

**To download:**
```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data -O dataset2_wisconsin_breast_cancer.csv
```

### Dataset 3: White Wine Quality (10 pairs)
**Source**: UCI Machine Learning Repository  
**File**: Pre-computed results in `winequality-white copy.csv`  
**Download**: [UCI ML Repo - Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
**Purpose**: Cross-modal non-linear relationships  
**Samples**: 4,898 wines  
**Features**: 11 chemical properties + quality rating  
**Selected Pairs**: 10 quality-predictive feature pairs

**To download:**
```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
```

---

## Reproduction Steps

### Quick Start (All-in-One Script)

```bash
python REPRODUCIBLE_CODE.py
```

This will:
1. Generate Ariel synthetic data
2. Load and process Wisconsin Breast Cancer data
3. Load White Wine pre-computed results
4. Run all three algorithms (CADEE, GW-CADEE, NMI)
5. Perform statistical analysis
6. Generate figures
7. Save all results to CSV files

**Expected Runtime**: ~10-15 minutes on a standard laptop

---

### Step-by-Step Reproduction

#### Step 1: Generate Ariel Synthetic Data
```python
from REPRODUCIBLE_CODE import generate_ariel_synthetic

patterns = generate_ariel_synthetic(n=1000)
# Returns 11 patterns with known ground truth
```

**Output**: 11 (name, x, y, type) tuples

#### Step 2: Run Algorithms on Dataset
```python
from REPRODUCIBLE_CODE import compute_mi_all_methods

for name, x, y, rel_type in patterns:
    results = compute_mi_all_methods(x, y)
    # results = {'CADEE': float, 'GW-CADEE': float, 'NMI': float}
```

**Output**: Dictionary with MI values for all three methods

#### Step 3: Load Breast Cancer Data
```python
import pandas as pd

cancer_df = pd.read_csv('dataset2_wisconsin_breast_cancer.csv')
# Process as shown in REPRODUCIBLE_CODE.py lines 400-450
```

**Output**: 30 selected pairs (15 monotonic + 15 non-monotonic)

#### Step 4: Run Statistical Analysis
```python
from REPRODUCIBLE_CODE import mcnemar_test, cohens_h

# Compare GW-CADEE vs CADEE
p_value, b, c = mcnemar_test(gwcadee_detected, cadee_detected)
effect_size = cohens_h(gwcadee_rate, cadee_rate)
```

**Output**: p-values and effect sizes

#### Step 5: Generate Figures
```bash
python statistical_comparison.py
```

**Output**: 4 PDF figures in current directory

#### Step 6: Generate LaTeX Tables
```bash
python generate_latex_comparison_tables.py
```

**Output**: `statistical_comparison_tables.tex`

---

## Project Structure

```
gw-cadee/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── REPRODUCIBLE_CODE.py                   # Main analysis script
├── statistical_comparison.py              # Statistical tests + figures
├── generate_latex_comparison_tables.py    # LaTeX table generator
│
├── pipeline_cadee.py                      # Standalone CADEE pipeline
├── pipeline_gwcadee.py                    # Standalone GW-CADEE pipeline
├── pipeline_nmi.py                        # Standalone NMI pipeline
├── run_all_pipelines.py                   # Master pipeline runner
│
├── data/
│   ├── dataset2_wisconsin_breast_cancer.csv  # Breast cancer data
│   ├── final_wine_results.csv                # Pre-computed wine results
│   └── README_DATASETS.txt                   # Dataset documentation
│
├── results/
│   ├── full_ariel_results.csv                # Ariel results (11 pairs)
│   ├── full_cancer_results.csv               # Cancer results (30 pairs)
│   ├── full_wine_results.csv                 # Wine results (10 pairs)
│   ├── statistical_comparison_performance.csv
│   ├── statistical_comparison_significance.csv
│   └── statistical_comparison_tables.tex
│
├── figures/
│   ├── comparison_barchart.pdf
│   ├── comparison_heatmap.pdf
│   ├── comparison_improvement.pdf
│   └── comparison_significance_matrix.pdf
│
└── docs/
    ├── COMPREHENSIVE_STATISTICAL_GUIDE.md    # Complete analysis guide
    ├── EXECUTIVE_SUMMARY.md                  # Key takeaways
    └── BREAST_CANCER_EXPLANATION.md          # Why full analysis matters
```

---

## Key Results

### Overall Performance (51 pairs)

| Algorithm | Detected | Rate  | vs CADEE Improvement |
|-----------|----------|-------|---------------------|
| CADEE     | 34/51    | 66.7% | baseline            |
| **GW-CADEE**  | **50/51**    | **98.0%** | **+16 (+31.4pp, +47.1%)** |
| NMI       | 27/51    | 52.9% | -7 (-13.7pp)        |

**Statistical Significance**: p < 0.001 (***), Cohen's h = 0.950 (large effect)

### By Dataset

| Dataset | n | CADEE | GW-CADEE | Improvement | p-value | Effect |
|---------|---|-------|----------|-------------|---------|--------|
| Ariel   | 11 | 63.6% | 90.9% | +27.3pp (+42.9%) | 0.250 | medium |
| Cancer  | 30 | 80.0% | 100.0% | +20.0pp (+25.0%) | 0.031* | large |
| Wine    | 10 | 30.0% | 100.0% | +70.0pp (+233%) | 0.016* | very large |

### Critical Finding: Non-Monotonic Detection

On Ariel's non-monotonic patterns:
- **CADEE**: 2/6 (33.3%) - Missed quadratic, circle, X-shape, step
- **GW-CADEE**: 6/6 (100.0%) - Detected all patterns correctly

---

## Detailed Code Explanation

### Core Algorithms

#### 1. CADEE (Spearman-based)
```python
def cadee_recursive(u, v, depth=0):
    # Step 4: Spearman independence test
    rho, p_value = spearmanr(u, v)
    if p_value > threshold:
        return 0.0
    
    # Step 6: Split on max |Spearman|
    rho_u = abs(spearmanr(u, v)[0])
    split_var = u if rho_u > rho_v else v
    
    # Steps 7-10: Partition, rescale, recurse, aggregate
    # ... (identical to GW-CADEE)
```

#### 2. GW-CADEE (Distance correlation-based)
```python
def gwcadee_recursive(u, v, depth=0):
    # Step 4: Distance correlation test (DIFFERENT)
    dcor = distance_correlation(u, v)
    if dcor < adaptive_threshold(n):
        return 0.0
    
    # Step 6: Split on max dCor (DIFFERENT)
    dcor_u = distance_correlation(u, v)
    split_var = u if dcor_u > dcor_v else v
    
    # Steps 7-10: Partition, rescale, recurse, aggregate
    # ... (identical to CADEE)
```

**Key Difference**: Only Steps 4 and 6 differ! Everything else is identical.

#### 3. Normalized MI (k-NN)
```python
def compute_nmi(x, y):
    # Direct k-NN estimation (no recursion)
    mi = mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=3)
    
    # Normalize to [0,1]
    h_x = entropy(x)
    h_y = entropy(y)
    nmi = mi / min(h_x, h_y)
    
    return nmi
```

**Completely different approach**: No copula, no recursion, k-NN based.

---

## How to Verify Results

### Check 1: Ariel Synthetic Ground Truth
```python
# Run Ariel analysis
python REPRODUCIBLE_CODE.py

# Expected: GW-CADEE detects 10/11, CADEE detects 7/11
# Check full_ariel_results.csv
```

**Verification**: 
- Quadratic: CADEE MI ≈ 0.00, GW-CADEE MI ≈ 1.15 ✓
- Circle: CADEE MI ≈ 0.00, GW-CADEE MI ≈ 2.25 ✓
- X-Shape: CADEE MI ≈ 0.00, GW-CADEE MI ≈ 1.69 ✓

### Check 2: Statistical Significance
```python
# Run statistical analysis
python statistical_comparison.py

# Check statistical_comparison_significance.csv
# Expected: Overall p < 0.001, h ≈ 0.950
```

### Check 3: Reproducibility
```bash
# Run twice and compare results
python REPRODUCIBLE_CODE.py > run1.txt
python REPRODUCIBLE_CODE.py > run2.txt
diff run1.txt run2.txt  # Should be identical (random seed = 42)
```

---

## Troubleshooting

### Issue: "FileNotFoundError: dataset2_wisconsin_breast_cancer.csv"
**Solution**: Download from UCI repository (see Datasets section)

### Issue: "ModuleNotFoundError: No module named 'scipy'"
**Solution**: Install requirements: `pip install -r requirements.txt`

### Issue: Results don't match exactly
**Possible causes**:
1. Different package versions (check `requirements.txt`)
2. Different random seed (should be 42)
3. Different dataset preprocessing

**Debug**:
```python
import numpy as np
print(np.random.get_state()[1][0])  # Should be 42 after seed setting
```

### Issue: Code runs very slowly
**Expected runtimes**:
- Ariel: ~30 seconds
- Cancer (30 pairs): ~5 minutes
- Full analysis: ~10-15 minutes

**Optimization**: Reduce pairs or use smaller sample size for testing

---

## Extending the Code

### Add New Dataset
```python
# In REPRODUCIBLE_CODE.py, add:

# Step X: Load your dataset
your_df = pd.read_csv('your_data.csv')

# Select pairs
your_pairs = [(col1, col2) for ...]

# Run algorithms
your_results = []
for col1, col2 in your_pairs:
    x = your_df[col1].values
    y = your_df[col2].values
    mi_values = compute_mi_all_methods(x, y)
    your_results.append({...})
```

### Modify Algorithm Parameters
```python
# Change recursion depth
cadee_recursive(u, v, max_depth=6)  # Default: 4

# Change minimum samples
gwcadee_recursive(u, v, min_samples=50)  # Default: 30

# Change k for NMI
compute_nmi(x, y, k=5)  # Default: 3
```

### Add New Visualization
```python
# In statistical_comparison.py, add:

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
# Your plotting code here
plt.savefig('your_figure.pdf', dpi=300)
```

---

## Performance Notes

### Computational Complexity

| Algorithm | Complexity per pair | Notes |
|-----------|---------------------|-------|
| CADEE | O(n log n) | Dominated by sorting |
| GW-CADEE | O(n² log n) | dCor computation is O(n²) |
| NMI | O(n²) | k-NN search |

### Recommended Sample Sizes

| Algorithm | Minimum | Optimal | Maximum |
|-----------|---------|---------|---------|
| CADEE | 100 | 1000+ | No limit |
| GW-CADEE | 100 | 500-1000 | 5000 (slow beyond) |
| NMI | 50 | 500 | 2000 |

### Memory Requirements

- Ariel (n=1000): ~10 MB
- Cancer (n=569, 30 pairs): ~50 MB
- Full analysis: ~100 MB peak

---

## Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_algorithms.py -v
```

### Integration Tests
```bash
# Test full pipeline
python test_reproducibility.py
```

Expected output: "All tests passed ✓"

---

## Citation

If you use this code in your research, please cite:

```bibtex
@gsgibbs{ggibbs2026gwcadee,
  title={GW-CADEE: Distance Correlation-Based Mutual Information Estimation},
  author={Gregory Gibbs},
  year={2026},
  school={Meharry Medical College},
  note={GitHub: https://github.com/gsgibbs/gw-cadee}
}
```

### References

1. Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007). Measuring and testing dependence by correlation of distances. *The Annals of Statistics*, 35(6), 2769-2794.

2. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.

3. Ariel G and Louzoun Y. Estimating Differential Entropy using Recursive Copula Splitting. Entropy
2020;22:236.

---

## License

MIT License - See LICENSE file for details

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: ggibbs23@mmc.edu

---

## Changelog

### Version 1.0.0 (March 2026)
- Initial release
- Full analysis of 3 datasets (51 pairs)
- Statistical comparison with significance testing
- Publication-ready figures and tables

---

## Acknowledgments

- UCI Machine Learning Repository for datasets
- Original CADEE authors
- Distance correlation theory by Székely et al.

---

**Last Updated**: March 29, 2026
 ## Datasets

This repository includes:

1. **Ariel Synthetic Dataset** (included)
   - 1,000 samples across 11 dependency patterns
   - File: `data/ariel_synthetic_dataset.csv`
   - See: `data/ARIEL_DATASET_README.md` for details

2. **Wisconsin Breast Cancer** (download required)
   - Download from UCI Machine Learning Repository
   - See: `data/README_DATASETS.txt` for instructions

3. **White Wine Quality** (included)
   - Pre-computed results in `data/winequality-white copy.csv`

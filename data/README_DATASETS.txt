# DATASET INFORMATION

This directory contains datasets and data-related documentation for the GW-CADEE analysis.

---

## Included Files

### 1. ariel_synthetic_dataset.csv ✓ INCLUDED
**Size**: ~500 KB  
**Source**: Generated programmatically (seed=42)  
**Purpose**: Controlled validation with known ground truth  
**Samples**: 1,000  
**Patterns**: 11 different dependency types

**Columns**: 23 total
- 1 sample_id column
- 22 data columns (11 patterns × 2 variables each)

**Pattern pairs**:
- `linear_positive_x` & `linear_positive_y`
- `linear_negative_x` & `linear_negative_y`
- `quadratic_x` & `quadratic_y`
- `cubic_x` & `cubic_y`
- `sine_x` & `sine_y`
- `circle_x` & `circle_y`
- `xshape_x` & `xshape_y`
- `step_x` & `step_y`
- `exponential_x` & `exponential_y`
- `logarithmic_x` & `logarithmic_y`
- `independent_x` & `independent_y`

**See**: `ARIEL_DATASET_README.md` for full details

**Usage**:
```python
import pandas as pd
df = pd.read_csv('data/ariel_synthetic_dataset.csv')

# Extract one pattern
x = df['quadratic_x'].values
y = df['quadratic_y'].values
```

---

### 2. final_wine_results.csv ✓ INCLUDED
**Size**: ~1 KB  
**Source**: Pre-computed results  
**Purpose**: White Wine Quality analysis results  
**Pairs**: 10 feature pairs  

**Contains**: Pre-computed MI values for:
- CADEE
- GW-CADEE (labeled as NL-CADEE)
- NMI

**Usage**: Load directly for analysis
```python
df = pd.read_csv('data/final_wine_results.csv')
```

---

### 3. dataset2_wisconsin_breast_cancer.csv ⚠️ DOWNLOAD REQUIRED

**Size**: ~100 KB  
**Source**: UCI Machine Learning Repository  
**Purpose**: Real biomedical data for validation  
**Samples**: 569 patients  
**Features**: 30 tumor measurements  

**Download Instructions**:

```bash
# Option 1: wget (Linux/Mac)
wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

# Option 2: curl (Mac/Linux)
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

# Rename the file
mv wdbc.data dataset2_wisconsin_breast_cancer.csv
```

**Or download manually**:
1. Go to: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
2. Click "Data Folder"
3. Download `wdbc.data`
4. Rename to `dataset2_wisconsin_breast_cancer.csv`
5. Place in this `data/` directory

**Why not included**: 
- Dataset is publicly available from UCI
- Keeps repository size small
- Ensures users get most recent version

**Used for**: 30 balanced feature pairs (15 monotonic + 15 non-monotonic)

---

## Dataset Summary Table

| Dataset | File | Included? | Size | Samples | Features | Pairs Analyzed |
|---------|------|-----------|------|---------|----------|----------------|
| **Ariel Synthetic** | ariel_synthetic_dataset.csv | ✓ Yes | 500 KB | 1,000 | 22 | 11 |
| **Breast Cancer** | dataset2_wisconsin_breast_cancer.csv | ✗ Download | 100 KB | 569 | 30 | 30 |
| **White Wine** | final_wine_results.csv | ✓ Yes | 1 KB | 4,898* | 11* | 10 |

*Wine: Pre-computed results only (original dataset not needed)

---

## Quick Start

### If You Have All Datasets:

```python
import pandas as pd

# Load Ariel
ariel = pd.read_csv('data/ariel_synthetic_dataset.csv')

# Load Breast Cancer
cancer = pd.read_csv('data/dataset2_wisconsin_breast_cancer.csv')

# Load Wine results
wine = pd.read_csv('data/final_wine_results.csv')

# Run analysis
python src/REPRODUCIBLE_CODE.py
```

### If You're Missing Breast Cancer Dataset:

1. Download as described above
2. Place in `data/` directory
3. Run analysis

The code will tell you if the file is missing.

---

## Ground Truth Summary

### Ariel Synthetic
- **Known dependencies**: All patterns (except independent)
- **Ground truth**: 10/11 pairs should be detected
- **CADEE expected**: ~7/11 (misses non-monotonic)
- **GW-CADEE expected**: ~10/11 (detects all)

### Breast Cancer
- **Real dependencies**: Unknown (real data)
- **Selected pairs**: 15 monotonic + 15 non-monotonic
- **CADEE expected**: ~24/30 (80%, misses some non-monotonic)
- **GW-CADEE expected**: 30/30 (100%, detects all)

### White Wine
- **Real dependencies**: Chemical → quality relationships
- **Known from domain**: pH, sulphates, acidity affect quality
- **CADEE expected**: ~3/10 (30%, misses most)
- **GW-CADEE expected**: 10/10 (100%, detects all)

---

## Data Integrity

All datasets use:
- **Encoding**: UTF-8
- **Separator**: Comma (,)
- **Missing values**: None in Ariel, handled in code for others
- **Header**: Yes (first row contains column names)

---

## Reproducibility

### Ariel Dataset
To regenerate exactly the same data:
```python
import numpy as np
np.random.seed(42)
# Then generate patterns as in REPRODUCIBLE_CODE.py
```

### Breast Cancer
UCI dataset is static (no randomness)

### Wine Wine
Results are pre-computed (deterministic given seed=42)

---

## Citations

### Ariel Synthetic Dataset
```bibtex
@dataset{ariel_synthetic_2026,
  title={Ariel Synthetic Dataset},
  author={[Your Name]},
  year={2026},
  note={Available at https://github.com/YOUR-USERNAME/gw-cadee}
}
```

### Wisconsin Breast Cancer Dataset
```bibtex
@misc{Dua:2019,
  author={Dua, Dheeru and Graff, Casey},
  year={2017},
  title={{UCI} Machine Learning Repository},
  url={http://archive.ics.uci.edu/ml},
  institution={University of California, Irvine, School of Information and Computer Sciences}
}
```

### White Wine Quality Dataset
```bibtex
@article{cortez2009modeling,
  title={Modeling wine preferences by data mining from physicochemical properties},
  author={Cortez, Paulo and Cerdeira, Ant{\'o}nio and Almeida, Fernando and Matos, Telmo and Reis, Jos{\'e}},
  journal={Decision Support Systems},
  volume={47},
  number={4},
  pages={547--553},
  year={2009},
  publisher={Elsevier}
}
```

---

## File Structure After Setup

```
data/
├── README_DATASETS.txt                     ← This file
├── ARIEL_DATASET_README.md                 ← Ariel documentation
├── ariel_synthetic_dataset.csv             ← Included (500 KB)
├── final_wine_results.csv                  ← Included (1 KB)
└── dataset2_wisconsin_breast_cancer.csv    ← Download required (100 KB)
```

---

## Troubleshooting

### "FileNotFoundError: dataset2_wisconsin_breast_cancer.csv"
→ Download from UCI (see instructions above)

### "Ariel results don't match published values"
→ Check random seed is set to 42

### "Wine file is corrupt"
→ Re-download from repository

### "Breast cancer file won't load"
→ Ensure file is named exactly: `dataset2_wisconsin_breast_cancer.csv`  
→ Check it's in the `data/` directory

---

## Contact

For questions about datasets:
- Open an issue on GitHub
- Email: [your.email@university.edu]

---

**Last Updated**: March 29, 2026

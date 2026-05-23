# Macrocyclic Peptide Membrane Permeability Prediction

Classical machine learning baselines (Random Forest and SVR) for predicting the PAMPA membrane permeability of cyclic peptides, evaluated under a rigorous **Murcko scaffold split** to test generalisation to entirely new molecular scaffolds.

---

## Table of Contents

- [Biological Background](#biological-background)
- [Why Predictability Matters](#why-predictability-matters)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Evaluation Strategy — Scaffold Split](#evaluation-strategy--scaffold-split)
- [Models](#models)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

---

## Biological Background

**Macrocyclic peptides** are ring-shaped molecules formed by head-to-tail (or side-chain) cyclisation of amino acid chains. Unlike linear peptides, the cyclic backbone restricts conformational flexibility, which confers several drug-relevant advantages:

- **Metabolic stability** — reduced degradation by proteases
- **Target selectivity** — pre-organised binding conformation
- **Cell penetration** — ability to cross membranes despite relatively large size

Many approved drugs are macrocyclic peptides: cyclosporine A (immunosuppressant), vancomycin (antibiotic), and somatostatin analogues used in oncology.

### The Membrane Permeability Problem

For a drug candidate to be orally bioavailable it must cross the intestinal epithelium. For most small molecules, Lipinski's "Rule of Five" predicts this reasonably well. Macrocycles routinely violate these rules (MW > 500 Da, multiple H-bond donors/acceptors) yet some — like cyclosporine A — still achieve good oral absorption.

This paradox arises because macrocycles can dynamically change conformation: a **polar, extended conformation** in water and a **compact, apolar conformation** in the membrane lipid bilayer, shielding H-bond donors through intramolecular hydrogen bonding. This chameleonic behaviour makes permeability extremely difficult to predict by simple rules and motivates the use of machine learning.

### PAMPA Assay

The **Parallel Artificial Membrane Permeability Assay (PAMPA)** quantifies passive transcellular permeability. A donor chamber contains the compound dissolved in buffer; it is separated from a receiver chamber by an artificial lipid membrane. After incubation, compound concentration in both chambers is measured. The effective permeability $P_e$ (units: cm/s, typically reported as log $P_e$) is calculated from the flux across the membrane.

PAMPA is high-throughput, reproducible, and closely models the transcellular permeability component of intestinal absorption, making it the standard *in vitro* assay for early ADME screening.

---

## Why Predictability Matters

### Accelerating drug discovery

Synthesising and testing macrocyclic peptides is expensive and slow. A reliable computational model that predicts PAMPA from structure alone enables:

1. **Virtual screening** — rank thousands of candidates before any synthesis
2. **Lead optimisation** — understand which structural changes improve permeability
3. **Scaffold hopping** — identify structurally distinct molecules with similar permeability profiles

### The generalisation challenge

A model that only interpolates between known scaffolds is of limited use. In real drug discovery, medicinal chemists explore **novel chemical space** — new ring systems, unusual amino acid combinations, non-standard macrolactam/macrolactone backbones. A robust model must generalise to these new scaffolds without having seen them during training.

This is why the **Murcko scaffold split** (see below) is essential for honest evaluation, and why classical ML baselines like Random Forest and SVR often underperform attention-based models on this stricter metric.

---

## Dataset

**CycPeptMPDB** — A Comprehensive Database of Membrane Permeability of Cyclic Peptides

| Property | Value |
|---|---|
| Source | Chen et al., J. Chem. Inf. Model. (2023) |
| DOI | [10.1021/acs.jcim.3c00054](https://doi.org/10.1021/acs.jcim.3c00054) |
| File | `CycPeptMPDB_Peptide_Assay_PAMPA.csv` |
| Contents | Cyclic peptide SMILES, PAMPA values, pre-computed RDKit 2D descriptors |
| Assay type | PAMPA (log $P_e$, cm/s) |

The CSV file is provided in this repository. Each row is a unique cyclic peptide with:
- `SMILES` — canonical SMILES string
- `PAMPA` — experimental log permeability (the regression target)
- Dozens of pre-computed physicochemical descriptors (MW, LogP, TPSA, H-bond counts, ring descriptors, etc.)

---

## Feature Engineering

Each molecule is represented by a **concatenated feature vector** of two complementary parts:

### Part A — RDKit 2D Physicochemical Descriptors

These are pre-computed in the CycPeptMPDB CSV and capture **global** drug-likeness properties:
- Molecular weight, LogP, TPSA
- H-bond donors and acceptors
- Rotatable bond count
- Ring system descriptors
- Charge descriptors

These descriptors directly encode properties known to influence passive membrane diffusion.

### Part B — Morgan Fingerprints (ECFP4, 2048 bits)

Extended Connectivity Fingerprints encode the **local structural environment** around every heavy atom:
- **Radius = 2** (ECFP4): each bit represents a unique circular subgraph extending up to 4 bonds from a central atom
- **2048 bits**: chosen to minimise bit collisions for large macrocyclic ring systems (shorter vectors collapse distinct substructures onto the same bit)

Morgan fingerprints capture substructure-level information (specific amino acid types, side-chain patterns, ring junction chemistry) that complements the global descriptors in Part A.

> **Featurisation reference:** This combined descriptor + fingerprint approach was used as the baseline feature set in the Molecular Attention Transformer (MAT) benchmarking study (Maziarka et al., 2020), which demonstrated that attention-based molecular representations outperform fixed fingerprints for scaffold generalisation.

---

## Evaluation Strategy — Scaffold Split

### The problem with random splits

A naive 80/20 random split places structurally similar molecules in both train and test sets. Because many macrocyclic peptide databases contain multiple analogues of the same scaffold, the model effectively interpolates between known chemical neighbours — this inflates performance metrics and does not reflect real predictive power.

### Murcko scaffold split (used in this work)

The **Bemis-Murcko scaffold** of a molecule is obtained by stripping all side chains, retaining only the ring systems and the atoms linking them. Two molecules sharing the same Murcko scaffold have the same core ring topology.

The splitting algorithm:
1. Assign a canonical Murcko scaffold to every molecule
2. Group all molecules by their scaffold
3. Sort scaffold groups largest → smallest
4. Greedily assign whole scaffold groups to train until ~80% is reached; all remaining groups go entirely to test

**Result:** The test set contains molecules with scaffolds that were **never seen during training**. This is a strict, drug-discovery-realistic evaluation of generalisation to new chemical space.

```
Example:
  Train: cyclosporine analogues (scaffold A) → model learns scaffold A
  Test:  FK506 analogues (scaffold B)        → model must generalise to scaffold B
  ✓  No scaffold B molecules leak into training
```

---

## Models

### 1. Random Forest Regressor (`rfr_scaffold_split.py`)

An ensemble of decision trees trained on bootstrapped subsets of the data. Final prediction = average across all trees. Does not require feature scaling.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 100 | Balances accuracy vs. training time |
| `max_depth` | 20 | Prevents memorisation of training scaffolds |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Uses all CPU cores |

### 2. Support Vector Regressor (`svm_scaffold_split.py`)

Finds a regression function within an ε-insensitive tube, using the RBF kernel to model non-linear relationships. **Requires StandardScaler** (fitted on training data only — no leakage).

| Hyperparameter | Value | Rationale |
|---|---|---|
| `kernel` | `rbf` | Standard for complex, non-linear chemical data |
| `C` | 1.0 | Regularisation strength |
| `epsilon` | 0.1 | ε-tube half-width |

**Scaling note:** The `StandardScaler` is fitted **only on the training set**. Fitting it on the full dataset before splitting would allow test-set statistics (mean, variance) to influence training — a form of data leakage. The scaler learned from the training set is then applied (transform only) to the test set.

---

## Repository Structure

```
macrocyclic-permeability/
│
├── macrocyclic_permeability_prediction.ipynb   # Master notebook (both models)
├── rfr_scaffold_split.py                       # Standalone Random Forest script
├── svm_scaffold_split.py                       # Standalone SVR script
├── CycPeptMPDB_Peptide_Assay_PAMPA.csv         # Dataset (uploaded separately)
└── README.md                                   # This file
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/macrocyclic-permeability.git
cd macrocyclic-permeability

# Install dependencies
pip install rdkit scikit-learn pandas numpy
```

**Google Colab users:** Run `!pip install rdkit` in the first cell — all other dependencies come pre-installed.

---

## Usage

### Jupyter Notebook (recommended)

Open `macrocyclic_permeability_prediction.ipynb`. The notebook walks through all steps from data loading to model comparison with explanatory text in each section.

### Standalone scripts

```bash
# Random Forest (default CSV in working directory)
python rfr_scaffold_split.py

# SVR
python svm_scaffold_split.py

# Custom CSV path
python rfr_scaffold_split.py --csv /path/to/CycPeptMPDB_Peptide_Assay_PAMPA.csv
python svm_scaffold_split.py --csv /path/to/CycPeptMPDB_Peptide_Assay_PAMPA.csv
```

---

## Results

Both models are evaluated on the scaffold-held-out test set (molecules with unseen core ring systems).

| Model | Split | R² | MAE | RMSE |
|---|---|---|---|---|
| Random Forest | Scaffold | *(run to obtain)* | *(run to obtain)* | *(run to obtain)* |
| SVR (RBF) | Scaffold | *(run to obtain)* | *(run to obtain)* | *(run to obtain)* |

> Update this table with your actual results after running the scripts.

### Interpreting the metrics

- **R²**: Fraction of variance in PAMPA explained by the model. R² = 1.0 is perfect; R² = 0.0 means the model is no better than predicting the mean.
- **MAE**: Mean Absolute Error in log permeability units (cm/s). Lower is better.
- **RMSE**: Root Mean Squared Error. More sensitive to large errors than MAE.

Under scaffold split, R² values for classical ML models typically range from 0.3–0.6, compared to 0.6–0.8 for graph neural network approaches. The gap reflects the inherent limitation of fixed fingerprints vs. learned molecular representations for scaffold generalisation.

---

## References

1. **Dataset:**
   Chen, Z., Jiang, D., Chen, G., Hao, G., Yang, G. et al.
   *CycPeptMPDB: A Comprehensive Database of Membrane Permeability of Cyclic Peptides.*
   J. Chem. Inf. Model. **63**, 3026–3036 (2023).
   DOI: [10.1021/acs.jcim.3c00054](https://doi.org/10.1021/acs.jcim.3c00054)

2. **Molecular Attention Transformer (featurisation / benchmarking reference):**
   Maziarka, Ł., Danel, T., Mucha, S., Rataj, K., Tabor, J. & Jastrzębski, S.
   *Molecule Attention Transformer.*
   arXiv:2002.08264 (2020).
   URL: [https://arxiv.org/abs/2002.08264](https://arxiv.org/abs/2002.08264)

3. **Murcko scaffold splitting:**
   Bemis, G. W. & Murcko, M. A.
   *The Properties of Known Drugs. 1. Molecular Frameworks.*
   J. Med. Chem. **39**, 2887–2893 (1996).
   DOI: [10.1021/jm9602928](https://doi.org/10.1021/jm9602928)

4. **Morgan / ECFP4 fingerprints:**
   Rogers, D. & Hahn, M.
   *Extended-Connectivity Fingerprints.*
   J. Chem. Inf. Model. **50**, 742–754 (2010).
   DOI: [10.1021/ci100050t](https://doi.org/10.1021/ci100050t)

5. **Cyclic peptide oral bioavailability (biological motivation):**
   Bockus, A. T. & Lokey, R. S.
   *From drugs to probes: the (re)purpose of cyclosporin A as a tool compound.*
   Curr. Opin. Chem. Biol. **17**, 916–922 (2013).

---

## License

This project is released for academic and research use. The CycPeptMPDB dataset is subject to its own license terms — please refer to the original publication (Chen et al., 2023) for details.

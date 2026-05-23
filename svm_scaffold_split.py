"""
svm_scaffold_split.py
=====================
Support Vector Regressor (SVR) for macrocyclic peptide PAMPA permeability prediction.

Evaluation strategy: Murcko Scaffold Split
  - All molecules sharing a core ring scaffold are kept entirely in either
    train or test (never split across both).
  - This tests generalisation to chemically novel scaffolds — the most
    realistic scenario in drug discovery.

Dataset: CycPeptMPDB_Peptide_Assay_PAMPA.csv
  Source: Chen et al. J. Chem. Inf. Model. 63, 3026-3036 (2023).
  DOI:    10.1021/acs.jcim.3c00054

Feature strategy:
  - Part A: Pre-computed RDKit 2D physicochemical descriptors (from the CSV)
  - Part B: 2048-bit Morgan fingerprints, radius=2 (ECFP4)
  Both parts were used as benchmark features in the Molecular Attention
  Transformer study (Maziarka et al., arXiv:2002.08264, 2020).

IMPORTANT — Feature Scaling:
  SVR is a distance-based kernel method. All features MUST be standardised
  to zero mean and unit variance before training. The StandardScaler is fitted
  ONLY on the training data to prevent data leakage.

Hyperparameters:
  kernel  = 'rbf'   (Radial Basis Function — standard for non-linear chem data)
  C       = 1.0     (regularisation; higher C → tighter fit to training data)
  epsilon = 0.1     (half-width of the ε-insensitive tube)

Usage:
  python svm_scaffold_split.py
  python svm_scaffold_split.py --csv /path/to/your/data.csv

Requirements:
  pip install rdkit scikit-learn pandas numpy
"""

# ── Standard library ─────────────────────────────────────────────────────────────
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ════════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit defaults here or pass CLI arguments
# ════════════════════════════════════════════════════════════════════════════════
DEFAULT_CSV    = "CycPeptMPDB_Peptide_Assay_PAMPA.csv"
TRAIN_FRACTION = 0.80   # Target fraction of molecules for the training set
MFP_RADIUS     = 2      # Morgan fingerprint radius (2 = ECFP4)
MFP_NBITS      = 2048   # Fingerprint bit-vector length
SVR_KERNEL     = "rbf"  # Kernel function for SVR
SVR_C          = 1.0    # Regularisation parameter
SVR_EPSILON    = 0.1    # ε-insensitive tube half-width

# Columns to exclude before selecting descriptor features
EXCLUDE_COLS = [
    'ID', 'Source', 'Year', 'Version',
    'Original_Name_in_Source_Literature',
    'Structurally_Unique_ID',
    'Same_Peptides_ID', 'Same_Peptides_Source',
    'Same_Peptides_Permeability', 'Same_Peptides_Assay',
    'SMILES', 'HELM', 'HELM_URL', 'Sequence',
    'Sequence_LogP', 'Sequence_TPSA',
    'Monomer_Length', 'Monomer_Length_in_Main_Chain', 'Molecule_Shape',
    'Permeability', 'PAMPA', 'Caco2', 'MDCK', 'RRCK',
    'Detection_Limit_1', 'Detection_Limit_2',
    'R_PAMAP', 'R_Caco2', 'R_MDCK', 'R_RRCK',
]


# ════════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def get_murcko_scaffold(smiles: str) -> str:
    """
    Return the canonical SMILES of the Bemis-Murcko scaffold for a molecule.
    Returns "" for invalid SMILES or extraction failures.
    The Murcko scaffold strips all side chains, leaving only the ring system
    and linkers — used to group structurally related molecules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return ""


def scaffold_split(df: pd.DataFrame, train_fraction: float = 0.80):
    """
    Split a DataFrame into train/test index lists using Murcko scaffold grouping.

    Algorithm:
      1. Assign a scaffold to every molecule.
      2. Group molecule indices by scaffold.
      3. Sort groups largest → smallest (greedy packing).
      4. Assign whole groups to train until train_fraction is reached;
         remaining groups go entirely to test.

    This guarantees zero scaffold overlap between train and test.

    Returns:
        train_idx (list): DataFrame indices assigned to training set.
        test_idx  (list): DataFrame indices assigned to test set.
    """
    df = df.copy()
    df["_scaffold"] = df["SMILES"].apply(get_murcko_scaffold)

    groups = df.groupby("_scaffold").groups
    sorted_scaffolds = sorted(groups.keys(), key=lambda s: len(groups[s]), reverse=True)

    train_idx, test_idx = [], []
    target_train_size = int(train_fraction * len(df))

    for scaffold in sorted_scaffolds:
        idxs = groups[scaffold].tolist()
        if len(train_idx) + len(idxs) <= target_train_size:
            train_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    return train_idx, test_idx


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the combined feature matrix:
      Part A — pre-computed RDKit descriptors from the CSV
      Part B — 2048-bit ECFP4 Morgan fingerprints computed from SMILES

    Returns a cleaned float32 DataFrame with NaN / ±Inf replaced by 0.
    Note: StandardScaler is applied AFTER this function, outside of it,
    to preserve the train-only fitting requirement.
    """
    # ── Part A: select numeric descriptor columns from the CSV ──────────────────
    descriptor_cols = [
        col for col in df.columns
        if col not in EXCLUDE_COLS and df[col].dtype in [np.float64, np.int64]
    ]
    X_desc = df[descriptor_cols].copy()

    # ── Part B: generate Morgan fingerprints from SMILES ────────────────────────
    fps = []
    for smi in df["SMILES"].tolist():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MFP_RADIUS, nBits=MFP_NBITS)
            fps.append(np.array(fp))
        else:
            fps.append(np.zeros(MFP_NBITS))  # Zero vector for unparseable SMILES

    X_fp = pd.DataFrame(fps, columns=[f"MFP_{i}" for i in range(MFP_NBITS)])

    # ── Concatenate and clean ────────────────────────────────────────────────────
    X = pd.concat([X_desc.reset_index(drop=True), X_fp.reset_index(drop=True)], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")  # Force numeric; strings → NaN
    X = X.astype(np.float32)                      # Downcast: overflow → ±inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)  # Replace inf/NaN with 0

    print(f"  Feature matrix: {X.shape[0]} molecules × {X.shape[1]} features")
    print(f"    ({len(descriptor_cols)} descriptors + {MFP_NBITS} fingerprint bits)")
    return X


def print_metrics(y_true, y_pred, label: str = "") -> dict:
    """Print a formatted table of regression metrics and return them as a dict."""
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    header = f"  SVR Results — {label}" if label else "  SVR Results"
    print("\n" + "=" * 50)
    print(header)
    print("=" * 50)
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print("=" * 50)
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MSE": mse}


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════════

def main(csv_path: str = DEFAULT_CSV):
    print("=" * 60)
    print("  Macrocyclic Peptide Permeability — SVR Model")
    print("  Evaluation: Murcko Scaffold Split")
    print("=" * 60)

    # ── Step 1: Load data ────────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {df.shape[0]} molecules from '{csv_path}'")
    print(f"  PAMPA range: {df['PAMPA'].min():.3f} — {df['PAMPA'].max():.3f}")

    # ── Step 2: Feature engineering ──────────────────────────────────────────────
    print("\n[2/6] Engineering features...")
    X = build_feature_matrix(df)
    y = df["PAMPA"].values

    # ── Step 3: Murcko scaffold split ────────────────────────────────────────────
    print("\n[3/6] Performing Murcko scaffold split...")
    train_idx, test_idx = scaffold_split(df, train_fraction=TRAIN_FRACTION)
    X_train_raw = X.iloc[train_idx]
    X_test_raw  = X.iloc[test_idx]
    y_train      = y[train_idx]
    y_test       = y[test_idx]
    print(f"  Train: {len(train_idx):,} ({len(train_idx)/len(df)*100:.1f}%)")
    print(f"  Test : {len(test_idx):,}  ({len(test_idx)/len(df)*100:.1f}%)")
    print("  Note: Test scaffolds are entirely unseen during training.")

    # ── Step 4: Feature scaling ──────────────────────────────────────────────────
    # CRITICAL: SVR relies on Euclidean distances in feature space.
    # Without scaling, features with large absolute values dominate the kernel.
    # The scaler MUST be fitted only on the training data — applying it to the
    # full dataset before splitting would constitute data leakage.
    print("\n[4/6] Scaling features (fit on train only — no data leakage)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)  # Learn mean/std from train; apply
    X_test  = scaler.transform(X_test_raw)        # Apply train statistics to test
    print(f"  Post-scaling train mean ≈ {X_train.mean():.4f}, std ≈ {X_train.std():.4f}")

    # ── Step 5: Train SVR ────────────────────────────────────────────────────────
    print(f"\n[5/6] Training SVR (kernel={SVR_KERNEL}, C={SVR_C}, epsilon={SVR_EPSILON})...")
    print("  This may take several minutes for large datasets.")
    model = SVR(
        kernel=SVR_KERNEL,   # RBF kernel maps inputs to infinite-dimensional space
        C=SVR_C,             # Regularisation (inverse of margin width)
        epsilon=SVR_EPSILON  # Half-width of the ε-insensitive tube
    )
    model.fit(X_train, y_train)
    print("  Training complete.")

    # ── Step 6: Evaluate ─────────────────────────────────────────────────────────
    print("\n[6/6] Evaluating on scaffold-held-out test set...")
    y_pred = model.predict(X_test)
    print_metrics(y_test, y_pred, label="Scaffold Test Set")

    return model, scaler


# ════════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SVR model for macrocyclic peptide PAMPA permeability prediction."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help=f"Path to CycPeptMPDB PAMPA CSV (default: {DEFAULT_CSV})"
    )
    args = parser.parse_args()
    main(csv_path=args.csv)

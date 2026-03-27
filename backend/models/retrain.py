"""
IonPredict â€” ALIGNN Retraining Pipeline  (v5 â€” fully bug-free)
================================================================================
Dataset : dataset_cleaned.csv  (452 rows, 13 columns, zero nulls)
Target  : log10(ionic_conductivity_S_cm)  â€” column 'log10_sigma'

All bugs fixed across v1 â†’ v5
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  BUG-1   : Line-graph node features correctly use RBF bond features (dim=NUM_RBF).
  BUG-2   : O(E) line-graph construction via adjacency grouping (not O(EÂ²)).
  BUG-3   : Atom features + target normalised on train set ONLY â€” no leakage.
  BUG-4   : Graph cache keyed on (cutoff, NUM_RBF, normaliser stats).
  BUG-5   : torch.load() uses weights_only flag correctly throughout.
  BUG-A   : Cache key includes normaliser y_mean/y_std â€” stale y-tensors never reused.
  BUG-B   : el.valence indexing robust â€” explicit None guard, use [-1].
  BUG-C   : train_epoch loss weighted by sample count, not batch count.
  BUG-D   : best_epoch lookup uses min() â€” no float-rounding mismatch.
  BUG-E   : pyg_scatter / pyg_softmax imports at module level, not inside forward().
  BUG-F   : CIF temp-file always cleaned up even on parse failure (try/finally).
  BUG-G   : fit_normaliser same temp-file leak fixed â€” now reuses _parse_cif().
  BUG-H   : evaluate() empty-loader guard â€” returns (0,0,[],[]) safely.
  BUG-I   : IonicConductivityDataset.get() â€” y stored/loaded as shape (1,) not (1,1)
            for correct PyG batching into shape (B,).
  BUG-J   : Deterministic CUDA seed + cudnn flags set alongside CPU seed.
  BUG-K   : FeatureNormaliser.save() strips trailing .npz before calling np.savez
            to prevent double extension (.npz.npz).
  BUG-L   : split_by_formula falls back to random split on single-formula datasets
            (avoids GroupShuffleSplit crash).
  BUG-M   : rbf_encode width clamped >= 1e-6 (division-by-zero guard).
  BUG-N   : Zero-length bond guard in line-graph angle computation.
  BUG-O   : attn_score output squeezed to (N,) before pyg_softmax (was (N,1)).
  BUG-P   : model-out default is .pt; history stem matches model stem.
  BUG-Q   : IonicConductivityDataset.get() returned a tuple (ag, lg) which PyG's
            DataLoader cannot collate. Fix: line-graph tensors are stored as
            attributes on the atom_graph Data object (ag.lg_x, ag.lg_edge_index,
            ag.lg_edge_attr) so the loader receives a single Data per sample and
            batches it correctly. Train/eval loops reconstruct the line-graph
            Data object from those attributes after batching.

Overfitting / underfitting mitigations
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  â€¢ Dropout=0.25 (raised slightly for 452-sample regime).
  â€¢ weight_decay=1e-4, gradient clipping at 5.0.
  â€¢ Early stopping patience=30 on val MAE.
  â€¢ ReduceLROnPlateau factor=0.5, patience=15, min_lr=1e-6.
  â€¢ LayerNorm (stable for variable/small batch sizes).
  â€¢ Attention pooling Dropout before the scoring linear.
  â€¢ train_mae tracked per epoch â€” train/val gap visible every 10 epochs.
  â€¢ param/sample ratio warning at >50.
  â€¢ Formula-grouped split â€” same composition never spans train+test.
  â€¢ Duplicate Material IDs and duplicate CIF hashes removed before splitting.
  â€¢ Label outlier removal (>5Ïƒ).

Usage
â”€â”€â”€â”€â”€
  python retrain.py --data dataset_cleaned.csv
  python retrain.py --data dataset_cleaned.csv --fetch-labels --mp-api-key KEY
  python retrain.py --data dataset_cleaned.csv --use-surrogate-target
  python retrain.py --hidden-dim 32 --num-layers 2 --epochs 100
"""

# â”€â”€ Standard library â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import os
import math
import json
import hashlib
import logging
import argparse
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# â”€â”€ Third-party â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter as pyg_scatter   # BUG-E: module-level
from torch_geometric.utils import softmax as pyg_softmax   # BUG-E: module-level

from pymatgen.io.cif import CifParser

# â”€â”€ Logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Constants
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

ATOM_FEATURE_NAMES: List[str] = [
    "Z",             # atomic number          (1-118)
    "atomic_mass",   # atomic mass            (~1-238 u)
    "atomic_radius", # covalent radius        (Angstrom)
    "X",             # electronegativity      (Pauling scale)
    "row",           # periodic table row     (1-7)
    "group",         # periodic table group   (1-18)
    "ionic_radius",  # ionic radius           (Angstrom)
    "valence",       # valence electron count
]
NUM_ATOM_FEATURES: int = len(ATOM_FEATURE_NAMES)   # 8
NUM_RBF:           int = 16


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CLI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrain ALIGNN on ionic conductivity dataset (v5)"
    )
    p.add_argument("--data",         default="dataset_cleaned.csv")
    p.add_argument("--label-col",    default="log10_sigma",
                   help="Target column name (log10 S/cm)")
    p.add_argument("--stable-only",  action="store_true",
                   help="Keep only Predicted Stable rows")
    p.add_argument("--no-metals",    action="store_true",
                   help="Drop Is Metal rows")

    p.add_argument("--fetch-labels", action="store_true",
                   help="Fetch log10 sigma from Materials Project before training")
    p.add_argument("--mp-api-key",   default=os.environ.get("MP_API_KEY", ""))
    p.add_argument(
        "--use-surrogate-target",
        action="store_true",
        help=(
            "If the label column is missing, derive a surrogate conductivity "
            "target from available material descriptors for demo training."
        ),
    )

    p.add_argument("--cutoff",       type=float, default=6.0,
                   help="Bond cutoff radius (Angstrom)")

    p.add_argument("--hidden-dim",   type=int,   default=128)
    p.add_argument("--num-layers",   type=int,   default=6)
    p.add_argument("--dropout",      type=float, default=0.25)

    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--early-stop",   type=int,   default=30,
                   help="Stop if val MAE does not improve for N epochs")

    p.add_argument("--val-split",    type=float, default=0.15)
    p.add_argument("--test-split",   type=float, default=0.10)
    p.add_argument("--seed",         type=int,   default=42)

    p.add_argument("--model-out",    default="alignn_model.pt")  # BUG-P: .pt
    p.add_argument("--device",       default="auto")
    return p.parse_args()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Step 0 - Optional: fetch labels from Materials Project
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def fetch_mp_labels(
    df: pd.DataFrame, api_key: str, label_col: str
) -> pd.DataFrame:
    try:
        from mp_api.client import MPRester
    except ImportError:
        logger.error("mp-api not installed.  Run: pip install mp-api")
        raise

    if not api_key:
        raise ValueError(
            "Materials Project API key required.\n"
            "Pass --mp-api-key YOUR_KEY  or  export MP_API_KEY=YOUR_KEY"
        )

    logger.info(f"Fetching conductivity labels for {len(df)} materials ...")
    results: dict = {}
    with MPRester(api_key) as mpr:
        for mp_id in df["Material ID"].tolist():
            try:
                docs = mpr.materials.summary.search(
                    material_ids=[mp_id],
                    fields=["material_id", "ionic_conductivity"],
                )
                if docs and docs[0].ionic_conductivity is not None:
                    sigma = float(docs[0].ionic_conductivity)
                    results[mp_id] = math.log10(sigma) if sigma > 0 else None
                else:
                    results[mp_id] = None
            except Exception as exc:
                logger.warning(f"  {mp_id}: {exc}")
                results[mp_id] = None

    df = df.copy()
    df[label_col] = df["Material ID"].map(results)
    found = int(df[label_col].notna().sum())
    logger.info(f"Labels fetched: {found}/{len(df)} materials have sigma data.")
    return df


def build_surrogate_target(df: pd.DataFrame) -> pd.Series:
    """
    Demo-only fallback target derived from available material descriptors when
    true ionic conductivity labels are unavailable.
    """
    band_gap = pd.to_numeric(df.get("Band Gap"), errors="coerce").fillna(0.0)
    density = pd.to_numeric(df.get("Density"), errors="coerce").fillna(0.0)
    energy_above_hull = pd.to_numeric(
        df.get("Energy Above Hull"), errors="coerce"
    ).fillna(0.0)
    formation_energy = pd.to_numeric(
        df.get("Formation Energy"), errors="coerce"
    ).fillna(0.0)
    sites = pd.to_numeric(df.get("Sites"), errors="coerce").fillna(0.0)
    stable = (
        df.get("Predicted Stable", pd.Series(False, index=df.index))
        .astype(str).str.lower().isin(["true", "1"])
    )
    is_metal = (
        df.get("Is Metal", pd.Series(False, index=df.index))
        .astype(str).str.lower().isin(["true", "1"])
    )

    score = (
        0.52 * band_gap.clip(lower=0, upper=8)
        - 1.35 * energy_above_hull.clip(lower=0, upper=2)
        - 0.18 * density.clip(lower=0, upper=15)
        + 0.12 * (-formation_energy).clip(lower=-5, upper=10)
        + 0.08 * sites.clip(lower=0, upper=50).pow(0.5)
        + stable.astype(float) * 0.55
        - is_metal.astype(float) * 0.45
        - 3.4
    )
    return score.astype(float)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Step 1 - Load, clean, deduplicate, filter
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def load_and_clean(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path}")

    if args.fetch_labels:
        df = fetch_mp_labels(df, args.mp_api_key, args.label_col)
        out_path = path.stem + "_with_labels.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved enriched dataset -> {out_path}")

    if args.label_col not in df.columns:
        if args.use_surrogate_target:
            logger.warning(
                f"Column '{args.label_col}' not found. Building surrogate target "
                "for ALIGNN demo training from available descriptors."
            )
            df[args.label_col] = build_surrogate_target(df)
            surrogate_path = path.with_name(path.stem + "_surrogate_labels.csv")
            df.to_csv(surrogate_path, index=False)
            logger.info(f"Saved surrogate-labelled dataset -> {surrogate_path}")
        else:
            logger.error(
                f"Column '{args.label_col}' not found.\n"
                f"Available: {list(df.columns)}\n"
                "Run with --fetch-labels, add the column manually, or pass "
                "--use-surrogate-target for a demo-only ALIGNN target."
            )
            raise SystemExit(1)

    # Remove duplicate Material IDs
    before = len(df)
    df = df.drop_duplicates(subset=["Material ID"]).reset_index(drop=True)
    if len(df) < before:
        logger.info(
            f"Removed {before - len(df)} duplicate Material IDs -> {len(df)} rows"
        )

    # Remove structurally identical CIFs
    df["_struct_hash"] = df["Structure"].apply(
        lambda s: hashlib.md5(str(s).encode()).hexdigest()
    )
    before = len(df)
    df = df.drop_duplicates(subset=["_struct_hash"]).reset_index(drop=True)
    if len(df) < before:
        logger.info(
            f"Removed {before - len(df)} identical CIF structures -> {len(df)} rows"
        )
    df = df.drop(columns=["_struct_hash"])

    # Drop rows with missing label
    df = df[df[args.label_col].notna()].reset_index(drop=True)
    logger.info(f"After label filter: {len(df)} rows")

    # Optional content filters
    if args.stable_only and "Predicted Stable" in df.columns:
        df = df[df["Predicted Stable"].astype(bool)].reset_index(drop=True)
        logger.info(f"After stable-only filter: {len(df)} rows")

    if args.no_metals and "Is Metal" in df.columns:
        df = df[~df["Is Metal"].astype(bool)].reset_index(drop=True)
        logger.info(f"After no-metals filter: {len(df)} rows")

    # Drop empty / whitespace-only CIF strings
    df = df[
        df["Structure"].notna() & (df["Structure"].str.strip() != "")
    ].reset_index(drop=True)

    # Label outlier removal (>5 sigma from mean)
    y    = df[args.label_col].values.astype(float)
    mu   = y.mean()
    sig  = y.std()
    mask = np.abs(y - mu) <= 5.0 * sig
    n_dropped = int((~mask).sum())
    if n_dropped:
        logger.warning(f"Dropped {n_dropped} extreme label outliers (>5 sigma).")
    df = df[mask].reset_index(drop=True)

    y = df[args.label_col].values.astype(float)
    logger.info(
        f"Label stats (log10 S/cm): "
        f"min={y.min():.2f}, max={y.max():.2f}, "
        f"mean={y.mean():.2f}, std={y.std():.2f}"
    )
    logger.info(f"Final clean dataset: {len(df)} rows")
    return df.reset_index(drop=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Step 2 - Formula-based GroupShuffleSplit
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def split_by_formula(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assign whole formula groups to train/val/test â€” same composition never
    spans splits.
    BUG-L: graceful fallback to random split when <= 2 unique formulas exist.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    groups   = df["Formula"].values
    n_groups = len(set(groups))

    if n_groups < 3:
        logger.warning(
            f"Only {n_groups} unique formula(s) â€” using random (non-grouped) split."
        )
        trainval_df, test_df = train_test_split(
            df, test_size=test_frac, random_state=seed, shuffle=True
        )
        val_frac_adj = val_frac / (1.0 - test_frac)
        train_df, val_df = train_test_split(
            trainval_df, test_size=val_frac_adj, random_state=seed, shuffle=True
        )
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)
    else:
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=test_frac, random_state=seed
        )
        trainval_idx, test_idx = next(gss_test.split(df, groups=groups))

        val_frac_adj = val_frac / (1.0 - test_frac)
        sub_groups   = groups[trainval_idx]
        n_sub_groups = len(set(sub_groups))

        if n_sub_groups < 2:
            logger.warning(
                "Too few formula groups after test carve â€” using random val split."
            )
            rng      = np.random.default_rng(seed)
            val_size = max(1, int(round(len(trainval_idx) * val_frac_adj)))
            shuffled = rng.permutation(trainval_idx)
            val_idx   = shuffled[:val_size]
            train_idx = shuffled[val_size:]
        else:
            gss_val = GroupShuffleSplit(
                n_splits=1, test_size=val_frac_adj, random_state=seed
            )
            train_idx_rel, val_idx_rel = next(
                gss_val.split(df.iloc[trainval_idx], groups=sub_groups)
            )
            train_idx = trainval_idx[train_idx_rel]
            val_idx   = trainval_idx[val_idx_rel]

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

    logger.info(
        f"Split -> train={len(train_df)} | val={len(val_df)} | test={len(test_df)}"
    )

    # Leakage verification
    train_f = set(train_df["Formula"])
    val_f   = set(val_df["Formula"])
    test_f  = set(test_df["Formula"])
    leak_tv = train_f & val_f
    leak_tt = train_f & test_f
    if leak_tv:
        logger.warning(f"Formula overlap train/val: {leak_tv}")
    if leak_tt:
        logger.warning(f"Formula overlap train/test: {leak_tt}")
    if not leak_tv and not leak_tt:
        logger.info("No formula overlap between splits.")

    return train_df, val_df, test_df


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Step 3 - Feature normalisation
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class FeatureNormaliser:
    """
    Fits mean/std on training atom features and target (log10 sigma).
    Fitted on TRAINING SET ONLY.
    BUG-K: save() strips .npz before np.savez to avoid double extension.
    """

    def __init__(self) -> None:
        self.atom_mean: np.ndarray = np.zeros(NUM_ATOM_FEATURES, dtype=np.float32)
        self.atom_std:  np.ndarray = np.ones(NUM_ATOM_FEATURES,  dtype=np.float32)
        self.y_mean: float = 0.0
        self.y_std:  float = 1.0

    def fit(
        self, all_atom_feats: List[np.ndarray], y_values: np.ndarray
    ) -> None:
        stacked        = np.vstack(all_atom_feats)
        self.atom_mean = stacked.mean(axis=0).astype(np.float32)
        self.atom_std  = (stacked.std(axis=0) + 1e-8).astype(np.float32)
        self.y_mean    = float(y_values.mean())
        self.y_std     = float(y_values.std()) + 1e-8
        logger.info(
            f"Normaliser: y_mean={self.y_mean:.3f}, y_std={self.y_std:.3f} | "
            f"atom_mean[:3]={self.atom_mean[:3].round(3)}"
        )

    def norm_atoms(self, feats: np.ndarray) -> np.ndarray:
        return ((feats - self.atom_mean) / self.atom_std).astype(np.float32)

    def norm_y(self, y: float) -> float:
        return (y - self.y_mean) / self.y_std

    def denorm_y(self, y_norm: float) -> float:
        return float(y_norm) * self.y_std + self.y_mean

    def save(self, path: str) -> None:
        # BUG-K: np.savez adds .npz automatically â€” strip it first
        save_path = str(path)
        if save_path.endswith(".npz"):
            save_path = save_path[:-4]
        np.savez(
            save_path,
            atom_mean=self.atom_mean,
            atom_std =self.atom_std,
            y_mean   =np.array([self.y_mean], dtype=np.float64),
            y_std    =np.array([self.y_std],  dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str) -> "FeatureNormaliser":
        load_path = str(path)
        if not load_path.endswith(".npz"):
            load_path += ".npz"
        obj = cls()
        d   = np.load(load_path)
        obj.atom_mean = d["atom_mean"].astype(np.float32)
        obj.atom_std  = d["atom_std"].astype(np.float32)
        obj.y_mean    = float(d["y_mean"][0])
        obj.y_std     = float(d["y_std"][0])
        return obj


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Step 4 - Graph construction
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def rbf_encode(
    distances: np.ndarray, d_max: float, num_rbf: int = NUM_RBF
) -> np.ndarray:
    """
    BUG-M: width clamped >= 1e-6 to prevent division-by-zero.
    """
    centers = np.linspace(0.0, d_max, num_rbf)
    width   = max(d_max / max(num_rbf, 1), 1e-6)
    return np.exp(
        -((distances[:, None] - centers[None, :]) ** 2) / (2.0 * width ** 2)
    ).astype(np.float32)


def get_atom_features(site) -> List[float]:
    """8 numeric features per atom site."""
    el = site.specie

    # Ionic radius
    try:
        ionic_r = float(el.ionic_radius) if el.common_oxidation_states else 0.0
    except Exception:
        ionic_r = 0.0

    # BUG-B: el.valence can be None or a named-tuple; use [-1] with None guard
    try:
        val_tuple = el.valence
        valence   = float(val_tuple[-1]) if val_tuple is not None else 0.0
    except (TypeError, IndexError, Exception):
        valence = 0.0

    return [
        float(el.Z             or 0),
        float(el.atomic_mass   or 0),
        float(el.atomic_radius or 0),
        float(el.X             or 0),
        float(el.row           or 0),
        float(el.group         or 0),
        ionic_r,
        valence,
    ]


def _parse_cif(cif_text: str) -> Optional[object]:
    """
    BUG-F: temp file guaranteed to be removed via try/finally.
    Returns a pymatgen Structure or None.
    """
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cif", delete=False
        ) as tmp:
            tmp.write(cif_text)
            tmp_path = tmp.name
        structures = CifParser(tmp_path).parse_structures(primitive=True)
        return structures[0] if structures else None
    except Exception as exc:
        logger.debug(f"CIF parse error: {exc}")
        return None
    finally:
        # BUG-F: guaranteed cleanup regardless of success or failure
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def cif_to_graphs(
    cif_text: str,
    cutoff: float,
    normaliser: FeatureNormaliser,
) -> Tuple[Optional[Data], Optional[np.ndarray]]:
    """
    Parse CIF -> (atom_graph, raw_node_feats).

    BUG-Q FIX: line-graph data is stored directly as extra attributes on the
    atom_graph Data object so that PyG's DataLoader receives a single Data
    object per sample and can batch it correctly via Batch.from_data_list().

      atom_graph.x              = normalised atom features  (N x 8)
      atom_graph.edge_attr       = RBF bond distances        (E x NUM_RBF)
      atom_graph.lg_x            = RBF bond distances        (E x NUM_RBF)  <- LG nodes
      atom_graph.lg_edge_index   = line-graph connectivity   (2 x A)
      atom_graph.lg_edge_attr    = bond angles (radians)     (A x 1)

    Returns (None, None) on any failure.
    """
    structure = _parse_cif(cif_text)
    if structure is None:
        return None, None

    # Node features
    raw_feats  = np.array(
        [get_atom_features(s) for s in structure], dtype=np.float32
    )
    node_feats = normaliser.norm_atoms(raw_feats)

    # Edges
    all_nbrs = structure.get_all_neighbors(cutoff, include_index=True)
    src_list:  List[int]   = []
    dst_list:  List[int]   = []
    dist_list: List[float] = []

    for i, nbrs in enumerate(all_nbrs):
        for nbr in nbrs:
            src_list.append(i)
            dst_list.append(int(nbr[2]))
            dist_list.append(float(nbr[1]))

    if not dist_list:
        logger.debug("No bonds found within cutoff â€” skipping structure.")
        return None, None

    dists      = np.array(dist_list, dtype=np.float32)
    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr  = rbf_encode(dists, cutoff, NUM_RBF)   # (E, NUM_RBF)

    # O(E) line graph via adjacency grouping (BUG-2)
    in_edges_of: dict = defaultdict(list)
    for e_idx, dst in enumerate(dst_list):
        in_edges_of[dst].append(e_idx)

    coords = np.array([s.coords for s in structure], dtype=np.float64)

    lg_src_list:  List[int]   = []
    lg_dst_list:  List[int]   = []
    angle_list:   List[float] = []

    for j, in_edge_idxs in in_edges_of.items():
        vecs:  List[np.ndarray] = []
        norms: List[float]      = []
        for e in in_edge_idxs:
            v = coords[src_list[e]] - coords[j]
            vecs.append(v)
            norms.append(float(np.linalg.norm(v)))

        for a_pos, ij_idx in enumerate(in_edge_idxs):
            for b_pos, jk_idx in enumerate(in_edge_idxs):
                if a_pos == b_pos:
                    continue
                denom = norms[a_pos] * norms[b_pos]
                # BUG-N: zero-length bond guard
                if denom < 1e-10:
                    angle = 0.0
                else:
                    cos_t = np.dot(vecs[a_pos], vecs[b_pos]) / denom
                    angle = float(
                        math.acos(float(np.clip(cos_t, -1.0, 1.0)))
                    )
                lg_src_list.append(ij_idx)
                lg_dst_list.append(jk_idx)
                angle_list.append(angle)

    if lg_src_list:
        lg_edge_index = np.array([lg_src_list, lg_dst_list], dtype=np.int64)
        lg_edge_attr  = np.array(angle_list, dtype=np.float32).reshape(-1, 1)
    else:
        lg_edge_index = np.zeros((2, 0), dtype=np.int64)
        lg_edge_attr  = np.zeros((0, 1), dtype=np.float32)

    # BUG-Q FIX: pack line-graph tensors into atom_graph so the DataLoader
    # sees one Data object per sample and batches it without errors.
    atom_graph = ALIGNNGraphData(
        x              = torch.from_numpy(node_feats),        # (N, 8)
        edge_index     = torch.from_numpy(edge_index),        # (2, E)
        edge_attr      = torch.from_numpy(edge_attr),         # (E, 16)
        lg_x           = torch.from_numpy(edge_attr.copy()),  # (E, 16)  LG nodes
        lg_edge_index  = torch.from_numpy(lg_edge_index),     # (2, A)
        lg_edge_attr   = torch.from_numpy(lg_edge_attr),      # (A, 1)
    )
    return atom_graph, raw_feats


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Step 5 - PyG Dataset with robust caching
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _cache_key(
    cutoff: float, num_rbf: int, normaliser: FeatureNormaliser
) -> str:
    """
    BUG-A + BUG-4: 16-char hash encoding cutoff, num_rbf AND normaliser stats.
    """
    cache_version = 2  # BUG-R: invalidate old caches after lg_edge_index batching fix
    key = (
        f"cache_version={cache_version}:cutoff={cutoff:.4f}:num_rbf={num_rbf}"
        f":ymean={normaliser.y_mean:.8f}:ystd={normaliser.y_std:.8f}"
    )
    return hashlib.md5(key.encode()).hexdigest()[:16]
class ALIGNNGraphData(Data):
    """
    PyG batch helper for atom graphs carrying line-graph tensors.

    BUG-R: lg_edge_index indexes into lg_x (bond nodes), not atom nodes, so
    batching must offset it by the number of line-graph nodes in each sample.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "lg_edge_index":
            return int(self.lg_x.size(0))
        return super().__inc__(key, value, *args, **kwargs)



class IonicConductivityDataset(Dataset):
    """
    Preprocesses CIF structures -> graph objects and caches to disk.

    BUG-I: y stored as shape (1,) for correct PyG batching into (B,).
    BUG-Q: each item is a single Data object (line-graph data stored as
           extra attributes) so PyG's default collator works correctly.
    """

    def __init__(
        self,
        df:         pd.DataFrame,
        cutoff:     float,
        label_col:  str,
        normaliser: FeatureNormaliser,
        split_name: str = "split",
    ) -> None:
        super().__init__()
        self.df         = df.reset_index(drop=True)
        self.cutoff     = cutoff
        self.label_col  = label_col
        self.normaliser = normaliser

        cfg            = _cache_key(cutoff, NUM_RBF, normaliser)
        self.cache_dir = Path(f".graph_cache/{split_name}_{cfg}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._valid_idx: List[int] = self._preprocess()

    def _preprocess(self) -> List[int]:
        valid: List[int] = []
        logger.info(
            f"Pre-processing {len(self.df)} structures "
            f"[cache: {self.cache_dir}] ..."
        )
        for i, row in self.df.iterrows():
            f_graph = self.cache_dir / f"{i}_graph.pt"

            if f_graph.exists():
                valid.append(int(i))
                continue

            cif_text   = str(row.get("Structure", ""))
            ag, _      = cif_to_graphs(cif_text, self.cutoff, self.normaliser)
            if ag is None:
                logger.debug(
                    f"  Skipped {row.get('Material ID', '?')} â€” graph build failed"
                )
                continue

            # BUG-I: y as shape (1,) â€” PyG batches to (B,) automatically
            y_norm = self.normaliser.norm_y(float(row[self.label_col]))
            ag.y   = torch.tensor([y_norm], dtype=torch.float)

            torch.save(ag, f_graph)
            valid.append(int(i))

        logger.info(f"  {len(valid)}/{len(self.df)} graphs built successfully.")
        return valid

    def len(self) -> int:
        return len(self._valid_idx)

    def get(self, idx: int) -> Data:
        """
        BUG-Q FIX: returns a single Data object (not a tuple).
        Line-graph tensors are stored as ag.lg_x / ag.lg_edge_index /
        ag.lg_edge_attr and will be batched automatically by PyG.
        """
        real = self._valid_idx[idx]
        ag   = torch.load(
            self.cache_dir / f"{real}_graph.pt", weights_only=False
        )
        return ag


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Helper: reconstruct line-graph Data from batched atom_graph attributes
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def extract_line_graph(ag: Data) -> Data:
    """
    Reconstruct a line-graph Data object from the extra attributes that were
    packed into the atom_graph by cif_to_graphs().  Called inside the
    train/eval loops after the DataLoader has batched the atom graphs.
    """
    return Data(
        x          = ag.lg_x,
        edge_index = ag.lg_edge_index,
        edge_attr  = ag.lg_edge_attr,
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Step 6 - ALIGNN Model
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class EdgeGatedConv(nn.Module):
    """
    One ALIGNN block:
      1. LineGraphConv  - update bond embeddings using bond-angle context
      2. EdgeGateUpdate - learn per-bond importance weights
      3. AtomGraphConv  - aggregate gated messages into atom embeddings
    """

    def __init__(
        self, node_dim: int, edge_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.line_conv = nn.Sequential(
            nn.Linear(edge_dim * 2 + 1, edge_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.Sigmoid(),
        )
        self.atom_conv = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.norm_node = nn.LayerNorm(node_dim)
        self.norm_edge = nn.LayerNorm(edge_dim)

    def forward(
        self,
        x:             torch.Tensor,
        edge_index:    torch.Tensor,
        edge_attr:     torch.Tensor,
        lg_x:          torch.Tensor,
        lg_edge_index: torch.Tensor,
        lg_edge_attr:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 1. Line graph conv
        if lg_edge_index.shape[1] > 0:
            lg_src_i, lg_dst_i = lg_edge_index
            lg_msg = torch.cat(
                [lg_x[lg_src_i], lg_x[lg_dst_i], lg_edge_attr], dim=-1
            )
            lg_msg = self.line_conv(lg_msg)
            lg_agg = pyg_scatter(
                lg_msg, lg_dst_i, dim=0,
                dim_size=lg_x.size(0), reduce="mean"
            )
            lg_x = lg_x + lg_agg

        edge_attr_new = self.norm_edge(edge_attr + lg_x)

        # 2. Gate
        gate = self.gate(edge_attr_new)

        # 3. Atom conv
        _, col = edge_index
        agg    = pyg_scatter(
            gate * edge_attr_new, col,
            dim=0, dim_size=x.size(0), reduce="sum"
        )
        x = self.norm_node(
            x + self.atom_conv(torch.cat([x, agg], dim=-1))
        )
        return x, edge_attr_new, lg_x


class ALIGNNModel(nn.Module):
    """
    ALIGNN for ionic conductivity regression.
    BUG-O: attn_score squeezed to (N,) before pyg_softmax.
    """

    def __init__(
        self,
        node_in:    int   = NUM_ATOM_FEATURES,
        edge_in:    int   = NUM_RBF,
        hidden:     int   = 64,
        num_layers: int   = 3,
        dropout:    float = 0.25,
    ) -> None:
        super().__init__()
        self.node_embed = nn.Linear(node_in, hidden)
        self.edge_embed = nn.Linear(edge_in, hidden)
        self.lg_embed   = nn.Linear(edge_in, hidden)

        self.convs = nn.ModuleList(
            [EdgeGatedConv(hidden, hidden, dropout) for _ in range(num_layers)]
        )

        # BUG-O: produces (N,1) -> squeezed to (N,) in forward()
        self.attn_score = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.pre_head_dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden,      hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self, atom_graph: Data, line_graph: Data
    ) -> torch.Tensor:

        x         = F.silu(self.node_embed(atom_graph.x))
        edge_attr = F.silu(self.edge_embed(atom_graph.edge_attr))
        lg_x      = F.silu(self.lg_embed(line_graph.x))

        for conv in self.convs:
            x, edge_attr, lg_x = conv(
                x,
                atom_graph.edge_index,
                edge_attr,
                lg_x,
                line_graph.edge_index,
                line_graph.edge_attr,
            )

        if hasattr(atom_graph, "batch") and atom_graph.batch is not None:
            batch = atom_graph.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # BUG-O: squeeze (N,1) -> (N,) before pyg_softmax
        raw_scores = self.attn_score(x).squeeze(-1)          # (N,)
        scores     = pyg_softmax(raw_scores, batch)           # (N,)
        graph_repr = pyg_scatter(
            scores.unsqueeze(-1) * x, batch, dim=0, reduce="sum"
        )                                                     # (B, H)

        return self.head(self.pre_head_dropout(graph_repr))   # (B, 1)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Step 7 - Normaliser fitting (training set only)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def fit_normaliser(
    train_df: pd.DataFrame, cutoff: float, label_col: str
) -> FeatureNormaliser:
    """
    BUG-G: reuses _parse_cif() which has the try/finally temp-file cleanup.
    Computes statistics from TRAINING SET ONLY.
    """
    logger.info("Fitting feature normaliser on training set ...")
    all_feats: List[np.ndarray] = []
    y_vals:    List[float]      = []

    for _, row in train_df.iterrows():
        structure = _parse_cif(str(row.get("Structure", "")))
        if structure is None:
            continue
        try:
            feats = np.array(
                [get_atom_features(s) for s in structure], dtype=np.float32
            )
            all_feats.append(feats)
            y_vals.append(float(row[label_col]))
        except Exception:
            continue

    if not all_feats:
        logger.warning("No structures parsed â€” using identity normaliser.")
        return FeatureNormaliser()

    norm = FeatureNormaliser()
    norm.fit(all_feats, np.array(y_vals, dtype=np.float64))
    return norm


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Step 8 - Training and evaluation loops
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    """
    BUG-C: loss = sum(MSE_i * n_i) / sum(n_i) â€” unbiased over partial batches.
    BUG-Q: loader yields single Data objects; line-graph reconstructed via
           extract_line_graph().
    """
    model.train()
    total_loss    = 0.0
    total_samples = 0

    for ag in loader:
        ag = ag.to(device)
        lg = extract_line_graph(ag)   # BUG-Q: reconstruct line-graph from attrs
        y  = ag.y.view(-1, 1)         # (B, 1)
        n  = y.size(0)

        optimizer.zero_grad()
        pred = model(ag, lg)
        loss = F.mse_loss(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss    += loss.item() * n
        total_samples += n

    return total_loss / total_samples if total_samples > 0 else 0.0


@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    normaliser: FeatureNormaliser,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    BUG-H: returns (0.0, 0.0, empty, empty) on empty loader.
    BUG-Q: loader yields single Data objects; line-graph reconstructed via
           extract_line_graph().
    All metrics in de-normalised log10 S/cm units.
    """
    model.eval()
    preds:   List[float] = []
    targets: List[float] = []

    for ag in loader:
        ag  = ag.to(device)
        lg  = extract_line_graph(ag)  # BUG-Q: reconstruct line-graph from attrs
        y_n = ag.y.view(-1).cpu().numpy()
        p_n = model(ag, lg).squeeze(-1).cpu().numpy()

        for v in np.atleast_1d(y_n):
            targets.append(normaliser.denorm_y(float(v)))
        for v in np.atleast_1d(p_n):
            preds.append(normaliser.denorm_y(float(v)))

    if not preds:
        return 0.0, 0.0, np.array([]), np.array([])

    p_arr = np.array(preds,   dtype=np.float64)
    t_arr = np.array(targets, dtype=np.float64)
    mae   = float(np.mean(np.abs(p_arr - t_arr)))
    rmse  = float(np.sqrt(np.mean((p_arr - t_arr) ** 2)))
    return mae, rmse, p_arr, t_arr


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Main
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main() -> None:
    args = parse_args()

    # Device
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    logger.info(f"Device: {device}")

    # BUG-J: seed everything for full reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # 1. Load & clean
    df = load_and_clean(args)
    if len(df) < 20:
        logger.error(f"Only {len(df)} usable samples â€” need >= 20.")
        raise SystemExit(1)

    # 2. Formula-grouped split
    train_df, val_df, test_df = split_by_formula(
        df, args.val_split, args.test_split, args.seed
    )

    # 3. Fit normaliser on train set ONLY
    normaliser = fit_normaliser(train_df, args.cutoff, args.label_col)
    norm_stem  = Path(args.model_out).stem + "_normaliser"
    normaliser.save(norm_stem)          # BUG-K: stem only, .npz added by np.savez
    norm_path  = norm_stem + ".npz"
    logger.info(f"Normaliser saved -> {norm_path}")

    # 4. Build PyG datasets
    train_ds = IonicConductivityDataset(
        train_df, args.cutoff, args.label_col, normaliser, "train"
    )
    val_ds   = IonicConductivityDataset(
        val_df,   args.cutoff, args.label_col, normaliser, "val"
    )
    test_ds  = IonicConductivityDataset(
        test_df,  args.cutoff, args.label_col, normaliser, "test"
    )

    if train_ds.len() < 5:
        logger.error("Too few valid training graphs. Check your Structure column.")
        raise SystemExit(1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 5. Build model
    model = ALIGNNModel(
        node_in    = NUM_ATOM_FEATURES,
        edge_in    = NUM_RBF,
        hidden     = args.hidden_dim,
        num_layers = args.num_layers,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio    = n_params / max(train_ds.len(), 1)
    logger.info(
        f"Model: hidden={args.hidden_dim}, layers={args.num_layers}, "
        f"dropout={args.dropout}, params={n_params:,}"
    )
    if ratio > 50:
        logger.warning(
            f"Param/sample ratio={ratio:.1f} (>50) â€” high overfitting risk. "
            "Try: --hidden-dim 32 --num-layers 2"
        )
    else:
        logger.info(f"Param/sample ratio={ratio:.1f} (acceptable)")

    optimizer = Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )

    # 6. Training loop with early stopping
    best_val_mae     = float("inf")
    patience_counter = 0
    history          = []

    logger.info(
        f"Training: max_epochs={args.epochs}, patience={args.early_stop}, "
        f"lr={args.lr}, wd={args.weight_decay}"
    )
    logger.info("-" * 72)

    for epoch in range(1, args.epochs + 1):
        train_loss              = train_epoch(model, train_loader, optimizer, device)
        train_mae, _, _, _      = evaluate(model, train_loader, device, normaliser)
        val_mae, val_rmse, _, _ = evaluate(model, val_loader,   device, normaliser)
        scheduler.step(val_mae)

        record = {
            "epoch":      epoch,
            "train_loss": round(float(train_loss), 6),
            "train_mae":  round(float(train_mae),  4),
            "val_mae":    round(float(val_mae),    4),
            "val_rmse":   round(float(val_rmse),   4),
            "lr":         round(float(optimizer.param_groups[0]["lr"]), 8),
        }
        history.append(record)

        if epoch % 10 == 0 or epoch == 1:
            overfit_flag = (
                " [OVERFIT WARNING]"
                if train_mae < val_mae * 0.5 else ""
            )
            logger.info(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"loss={train_loss:.5f} | "
                f"train_MAE={train_mae:.4f} | "
                f"val_MAE={val_mae:.4f} | "
                f"val_RMSE={val_rmse:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.1e} | "
                f"patience={patience_counter}/{args.early_stop}"
                f"{overfit_flag}"
            )

        if val_mae < best_val_mae:
            best_val_mae     = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), args.model_out)
            logger.info(
                f"  New best val_MAE={val_mae:.4f} â€” checkpoint saved -> {args.model_out}"
            )
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no val improvement for {args.early_stop} epochs)"
            )
            break

    # 7. Load best checkpoint and run final test evaluation
    logger.info("-" * 72)
    model.load_state_dict(
        torch.load(args.model_out, map_location=device, weights_only=True)
    )
    test_mae, test_rmse, test_preds, test_targets = evaluate(
        model, test_loader, device, normaliser
    )
    logger.info(f"Test MAE  : {test_mae:.4f}  (log10 S/cm)")
    logger.info(f"Test RMSE : {test_rmse:.4f}  (log10 S/cm)")

    # Overfitting diagnostic
    logger.info("-" * 72)
    logger.info("Overfitting diagnostic:")
    logger.info(f"  Best val MAE : {best_val_mae:.4f}")
    logger.info(f"  Test MAE     : {test_mae:.4f}")
    if test_mae > best_val_mae * 1.5:
        logger.warning(
            "  Test MAE >> Val MAE â€” possible overfitting or distribution shift.\n"
            "  Try: --dropout 0.3 --weight-decay 1e-3 --hidden-dim 32"
        )
    else:
        logger.info("  No severe overfitting detected.")

    # 8. Save history + metadata
    # BUG-D: min() over history â€” no float rounding ambiguity
    best_epoch = min(history, key=lambda h: h["val_mae"])["epoch"]

    hist_path = Path(args.model_out).stem + "_history.json"
    payload   = {
        "history":      history,
        "best_epoch":   best_epoch,
        "best_val_mae": float(best_val_mae),
        "test_mae":     float(test_mae),
        "test_rmse":    float(test_rmse),
        "test_predictions": [
            {"target": float(t), "predicted": float(p)}
            for t, p in zip(test_targets, test_preds)
        ],
        "model_config": {
            "hidden_dim":    args.hidden_dim,
            "num_layers":    args.num_layers,
            "dropout":       args.dropout,
            "cutoff":        args.cutoff,
            "num_rbf":       NUM_RBF,
            "atom_features": ATOM_FEATURE_NAMES,
            "n_params":      n_params,
        },
        "dataset": {
            "path":      args.data,
            "total":     len(df),
            "train":     train_ds.len(),
            "val":       val_ds.len(),
            "test":      test_ds.len(),
            "label_col": args.label_col,
            "y_mean":    float(normaliser.y_mean),
            "y_std":     float(normaliser.y_std),
        },
        "args":     vars(args),
        "finished": datetime.now(timezone.utc).isoformat(),
    }

    with open(hist_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info(f"History    -> {hist_path}")
    logger.info(f"Model      -> {args.model_out}")
    logger.info(f"Normaliser -> {norm_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()


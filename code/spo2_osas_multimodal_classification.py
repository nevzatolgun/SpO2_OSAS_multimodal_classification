# -*- coding: utf-8 -*-
"""OSAS severity classification analysis.

This script performs feature extraction, statistical analysis, cross-validation,
resampling tests and model interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import (
    chi2_contingency, fisher_exact, f_oneway, kruskal, mannwhitneyu, norm,
    rankdata, studentized_range, ttest_ind,
)
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

import shap

SEED = 42
FS = 16
SEGMENT_DURATION = 60
N_SPLITS = 5
DEFAULT_N_BOOTSTRAP = 5000
DEFAULT_N_PERMUTATIONS = 5000

ODI_COL = "Ortalama SpO2 Desaturasyonu (ODI) (%)"

DISPLAY_CLASS_NAMES = np.array(["Snoring", "Mild", "Moderate", "Severe"], dtype=object)
TRAIN_CLASS_NAMES = np.array(["Mild", "Moderate", "Severe", "Snoring"], dtype=object)
TRAIN_CLASSES = np.arange(len(TRAIN_CLASS_NAMES), dtype=int)
NAME_TO_TRAIN_ID = {name: i for i, name in enumerate(TRAIN_CLASS_NAMES)}
DISPLAY_CLASS_IDS = np.array([NAME_TO_TRAIN_ID[name] for name in DISPLAY_CLASS_NAMES], dtype=int)

CLASS_MAP = {
    "AĞIR": "Severe", "ORTA": "Moderate", "HAFİF": "Mild",
    "BASİT HORLAMA": "Snoring", 
}


FEATURE_LABEL_MAP = {
    "mean_spo2": "Mean SpO2",
    "std_spo2": "Standard Deviation of SpO2",
    "min_spo2": "Minimum SpO2",
    "under_75_ratio": "Proportion of Time SpO2 < 75%",
    "psd_mean": "Mean Power Spectral Density",
    "psd_max": "Max Power Spectral Density",
    "nominal_psd_sum": "Nominal PSD Sum (0.01-0.1 Hz)",
    "desat_avg_min_spo2": "Avg. Min SpO2 During Desaturations",
    "BMI": "Body Mass Index",
    "ESS": "Epworth Sleepiness Score",
    "CİNSİYET": "Gender",
    "PDW (fL)": "Platelet Distribution Width",
    "MPV (fL)": "Mean Platelet Volume",
    "RDW-CV (%)": "Red Cell Distribution Width",
    ODI_COL: "Oxygen Desaturation Index",
}
MODEL_ORDER = ["KNN", "ExtraTrees", "RandomForest"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OSAS severity classification analysis.")
    parser.add_argument("--data", default="./all_subjects_spo2_combined.csv", help="Input CSV path.")
    parser.add_argument("--output", default="./OSAS_results", help="Output directory.")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP,
                        help="Bootstrap repetitions when --bootstrap-config is used.")
    parser.add_argument("--bootstrap-config", default=None, help="A1..A11 or ALL.")
    parser.add_argument("--bootstrap-model", default="ALL", help="KNN, RandomForest, ExtraTrees, or ALL.")
    parser.add_argument("--permutations", type=int, default=DEFAULT_N_PERMUTATIONS,
                        help="Permutation repetitions when --permutation-config is used.")
    parser.add_argument("--permutation-config", default=None, help="A1..A11 or ALL.")
    parser.add_argument("--permutation-model", default="ExtraTrees", help="KNN, RandomForest, ExtraTrees, or ALL.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP value and interaction calculations.")
    return parser.parse_args()

def ensure_dirs(root: Path) -> Dict[str, Path]:
    if root.exists():
        shutil.rmtree(root)
    dirs = {
        "root": root,
        "tables": root / "tables",
        "cv": root / "cross_validation",
        "bootstrap": root / "bootstrap",
        "permutation": root / "permutation",
        "interpretability": root / "interpretability",
        "features": root / "features",
        "visualization_data": root / "visualization_data",
        "logs": root / "logs",
        "checkpoints": root / "checkpoints",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs



def setup_reproducibility() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    warnings.filterwarnings("ignore")



def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"



def fmt_num(v: float, max_decimals: int = 2) -> str:
    if not np.isfinite(v):
        return "NA"
    if abs(v - round(v)) < 1e-10:
        return str(int(round(v)))
    return f"{v:.{max_decimals}f}".rstrip("0").rstrip(".")



def resolve_column(df: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> str | None:
    for c in aliases:
        if c in df.columns:
            return c
    lower_map = {str(c).casefold(): c for c in df.columns}
    for c in aliases:
        if c.casefold() in lower_map:
            return lower_map[c.casefold()]
    if required:
        raise KeyError(f"None of the expected column names were found: {aliases}")
    return None



def normalize_gender(value: object) -> str:
    s = str(value).strip().casefold()
    male = {"e", "erkek", "male", "m", "man"}
    female = {"k", "kadın", "kadin", "female", "f", "woman"}
    if s in male:
        return "Male"
    if s in female:
        return "Female"
    return str(value).strip()



def align_probability_columns(raw_proba: np.ndarray, model_classes: Sequence[int]) -> np.ndarray:
    aligned = np.zeros((raw_proba.shape[0], len(TRAIN_CLASSES)), dtype=float)
    target = {int(c): i for i, c in enumerate(TRAIN_CLASSES)}
    for src_col, cls in enumerate(model_classes):
        aligned[:, target[int(cls)]] = raw_proba[:, src_col]
    return aligned



def safe_multiclass_auc(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, List[float], np.ndarray]:
    y_bin = label_binarize(y_true, classes=TRAIN_CLASSES)
    aucs: List[float] = []
    for i in range(len(TRAIN_CLASSES)):
        aucs.append(float(roc_auc_score(y_bin[:, i], proba[:, i])))
    return float(np.mean(aucs)), aucs, y_bin



def calculate_metrics(y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    auc, _, _ = safe_multiclass_auc(y_true, proba)
    return {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Macro_Precision": float(precision_score(y_true, pred, labels=TRAIN_CLASSES, average="macro", zero_division=0)),
        "Macro_Recall": float(recall_score(y_true, pred, labels=TRAIN_CLASSES, average="macro", zero_division=0)),
        "Macro_F1": float(f1_score(y_true, pred, labels=TRAIN_CLASSES, average="macro", zero_division=0)),
        "Macro_ROC_AUC": auc,
    }



def extract_features(signal: Sequence[float], fs: int = FS, segment_duration: int = SEGMENT_DURATION) -> pd.Series | None:
    signal = np.asarray(signal, dtype=float)
    signal = signal[signal > 0]
    if len(signal) < fs * segment_duration:
        return None
    segment_len = fs * segment_duration
    n_segments = len(signal) // segment_len
    means, stds, mins = [], [], []
    under75_ratios = []
    psd_means, psd_maxs, nominal_psd_sums = [], [], []
    desat_mins = []
    for i in range(n_segments):
        seg = signal[i * segment_len:(i + 1) * segment_len]
        freqs, psd = welch(seg, fs=fs, nperseg=min(512, len(seg)))
        psd_band = psd[(freqs >= 0.01) & (freqs <= 0.1)]
        means.append(seg.mean())
        stds.append(seg.std())  
        mins.append(seg.min())
        under75_ratios.append(np.mean(seg < 75))
        psd_means.append(psd.mean())
        psd_maxs.append(psd.max())
        nominal_psd_sums.append(psd_band.sum())
        below = seg < 75
        in_desat = False
        start = 0
        vals = []
        for j, b in enumerate(below):
            if b and not in_desat:
                in_desat = True
                start = j
            elif not b and in_desat:
                in_desat = False
                vals.append(seg[start:j].min())
        if in_desat:
            vals.append(seg[start:].min())
        desat_mins.append(np.mean(vals) if vals else 100)
    return pd.Series({
        "mean_spo2": np.mean(means),
        "std_spo2": np.mean(stds),
        "min_spo2": np.min(mins),
        "under_75_ratio": np.mean(under75_ratios),
        "psd_mean": np.mean(psd_means),
        "psd_max": np.max(psd_maxs),
        "nominal_psd_sum": np.mean(nominal_psd_sums),
        "desat_avg_min_spo2": np.mean(desat_mins),
    })



def load_and_prepare_data(data_path: Path):
    raw_df = pd.read_csv(data_path)
    df = raw_df.copy()
    df["SINIF"] = df["SINIF"].map(CLASS_MAP)
    df = df.dropna(subset=["SINIF"]).reset_index(drop=True)

    df["CİNSİYET_RAW"] = df["CİNSİYET"].copy()

    target_encoder = LabelEncoder()
    df["SINIF_ENC"] = target_encoder.fit_transform(df["SINIF"].astype(str))

    gender_encoder = LabelEncoder()
    df["CİNSİYET"] = gender_encoder.fit_transform(df["CİNSİYET"].astype(str))
    spo2_cols = [c for c in df.columns if str(c).startswith("SpO2_")]
    required_extra_cols = ["CİNSİYET", "BMI", "ESS", "PDW (fL)", "MPV (fL)", "RDW-CV (%)"]
    all_extra_cols = required_extra_cols + [ODI_COL]
    for c in all_extra_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    spo2_raw = df[spo2_cols].copy()
    extra = df[all_extra_cols].copy()
    y_raw = df["SINIF_ENC"].copy()
    features_list = []
    valid_indices = []
    for idx, row in spo2_raw.iterrows():
        feats = extract_features(row.values)
        if feats is not None:
            features_list.append(feats)
            valid_indices.append(idx)
    spo2_features = pd.DataFrame(features_list).reset_index(drop=True)
    extra_features = extra.loc[valid_indices].reset_index(drop=True)
    X_all = pd.concat([spo2_features, extra_features], axis=1)
    y = y_raw.loc[valid_indices].reset_index(drop=True)
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    spo2_raw_valid = spo2_raw.loc[valid_indices].reset_index(drop=True)
    return df_valid, spo2_raw_valid, X_all, y, valid_indices, gender_encoder



def make_feature_sets(X_all: pd.DataFrame) -> Dict[str, List[str]]:
    basic = ["mean_spo2", "std_spo2", "min_spo2"]
    psd = ["psd_mean", "psd_max", "nominal_psd_sum"]
    clinical = ["BMI", "ESS"]
    hemogram = ["PDW (fL)", "MPV (fL)", "RDW-CV (%)"]
    gender = ["CİNSİYET"]
    desat = ["under_75_ratio", "desat_avg_min_spo2"]
    a1 = basic
    a2 = a1 + psd
    a3 = a2 + clinical
    a4 = a3 + hemogram
    a5 = a4 + gender
    sets = {
        "A1_Basic_SpO2_statistics": a1,
        "A2_A1_plus_PSD_features": a2,
        "A3_A2_plus_BMI_ESS": a3,
        "A4_A3_plus_Hemogram_ODI_free": a4,
        "A5_A4_plus_Gender": a5,
        "A6_A3_plus_Desaturation_features": a3 + desat,
        "A7_A4_plus_Desaturation_features": a4 + desat,
        "A8_A4_plus_Gender_Desaturation_features": a5 + desat,
    }
    sets.update({
        "A9_A3_plus_ODI_sensitivity_only": a3 + [ODI_COL],
        "A10_A4_plus_ODI_sensitivity_only": a4 + [ODI_COL],
        "A11_A4_plus_Gender_Desaturation_ODI_sensitivity_only": a5 + desat + [ODI_COL],
    })
    return sets



def make_models() -> Dict[str, object]:
    return {
        "KNN": KNeighborsClassifier(n_neighbors=3, metric="manhattan"),
        "RandomForest": RandomForestClassifier(random_state=SEED),
        "ExtraTrees": ExtraTreesClassifier(random_state=SEED),
    }



def evaluate_configuration(
    X_all: pd.DataFrame,
    y: pd.Series,
    selected_cols: Sequence[str],
    model_name: str,
    base_model,
    shared_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    valid_indices: Sequence[int],
) -> Dict[str, object]:
    X_values = X_all[list(selected_cols)].to_numpy(dtype=float)
    n = len(y)
    oof_pred = np.full(n, -1, dtype=int)
    oof_proba = np.zeros((n, len(TRAIN_CLASSES)), dtype=float)
    oof_fold = np.full(n, -1, dtype=int)
    fold_rows, class_rows, fold_store, reports = [], [], [], []
    for fold, (train_idx, test_idx) in enumerate(shared_splits, start=1):
        model = clone(base_model)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_values[train_idx])
        X_test = scaler.transform(X_values[test_idx])
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred = model.predict(X_test)
        raw_proba = model.predict_proba(X_test)
        test_time = time.perf_counter() - t1
        proba = align_probability_columns(raw_proba, model.classes_)
        oof_pred[test_idx] = pred
        oof_proba[test_idx] = proba
        oof_fold[test_idx] = fold
        metrics = calculate_metrics(y_test, pred, proba)
        fold_rows.append({
            "Model": model_name,
            "Fold": fold,
            "N_train": len(train_idx),
            "N_test": len(test_idx),
            **metrics,
            "Train_Time_s": train_time,
            "Inference_Time_Total_s": test_time,
            "Inference_Time_Per_Subject_ms": test_time / len(test_idx) * 1000.0,
        })
        p, r, f1, support = precision_recall_fscore_support(
            y_test, pred, labels=DISPLAY_CLASS_IDS, zero_division=0
        )
        for ci, cname in enumerate(DISPLAY_CLASS_NAMES):
            class_rows.append({
                "Model": model_name,
                "Fold": fold,
                "Class": cname,
                "Precision": p[ci],
                "Recall": r[ci],
                "F1_score": f1[ci],
                "Support": int(support[ci]),
            })
        _, aucs, y_bin = safe_multiclass_auc(y_test, proba)
        fold_store.append({
            "fold": fold,
            "train_idx": np.asarray(train_idx),
            "test_idx": np.asarray(test_idx),
            "y_test": y_test,
            "pred": pred,
            "proba": proba,
            "y_bin": y_bin,
            "aucs": aucs,
            "accuracy": metrics["Accuracy"],
            "macro_f1": metrics["Macro_F1"],
        })
        reports.append(classification_report(
            y_test, pred, labels=DISPLAY_CLASS_IDS, target_names=DISPLAY_CLASS_NAMES, zero_division=0
        ))
    oof_metrics = calculate_metrics(y.to_numpy(), oof_pred, oof_proba)
    oof_frame = pd.DataFrame({
        "Subject_Index": np.arange(n),
        "Original_Data_Row": np.asarray(valid_indices),
        "Fold": oof_fold,
        "True_Label": y.to_numpy(),
        "True_Class": [TRAIN_CLASS_NAMES[i] for i in y.to_numpy()],
        "Predicted_Label": oof_pred,
        "Predicted_Class": [TRAIN_CLASS_NAMES[i] for i in oof_pred],
    })
    for cname, train_id in zip(DISPLAY_CLASS_NAMES, DISPLAY_CLASS_IDS):
        oof_frame[f"Probability_{cname}"] = oof_proba[:, train_id]
    return {
        "Model": model_name,
        "N_features": len(selected_cols),
        "Selected_columns": list(selected_cols),
        "Fold_metrics": pd.DataFrame(fold_rows),
        "Class_metrics": pd.DataFrame(class_rows),
        "OOF_metrics": oof_metrics,
        "OOF_y_true": y.to_numpy().copy(),
        "OOF_pred": oof_pred,
        "OOF_proba": oof_proba,
        "OOF_fold": oof_fold,
        "OOF_frame": oof_frame,
        "Fold_store": fold_store,
        "Reports": reports,
    }



def summarize_fold_results(result: Dict[str, object]) -> Dict[str, float]:
    f = result["Fold_metrics"]
    row = {"Model": result["Model"], "N_features": result["N_features"]}
    for col in ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Macro_ROC_AUC", "Inference_Time_Per_Subject_ms"]:
        row[f"{col}_mean"] = float(f[col].mean())
        row[f"{col}_std"] = float(f[col].std(ddof=1))
    return row



def dunn_bonferroni(groups: Sequence[np.ndarray]) -> np.ndarray:
    clean = [np.asarray(g, dtype=float)[np.isfinite(g)] for g in groups]
    values = np.concatenate(clean)
    ranks = rankdata(values, method="average")
    N = len(values)

    _, counts = np.unique(values, return_counts=True)
    tie_corr = 1.0 - np.sum(counts**3 - counts) / (N**3 - N) if N > 1 else 1.0
    variance = N * (N + 1) / 12.0 * tie_corr
    mean_ranks = []
    pos = 0
    for g in clean:
        mean_ranks.append(np.mean(ranks[pos:pos + len(g)]))
        pos += len(g)
    k = len(clean)
    m = k * (k - 1) // 2
    pmat = np.ones((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            se = math.sqrt(variance * (1 / len(clean[i]) + 1 / len(clean[j])))
            z = (mean_ranks[i] - mean_ranks[j]) / se if se > 0 else 0.0
            p = min(1.0, 2 * norm.sf(abs(z)) * m)
            pmat[i, j] = pmat[j, i] = p
    return pmat



def games_howell(groups: Sequence[np.ndarray]) -> np.ndarray:
    clean = [np.asarray(g, dtype=float)[np.isfinite(g)] for g in groups]
    k = len(clean)
    pmat = np.ones((k, k), dtype=float)
    means = [np.mean(g) for g in clean]
    variances = [np.var(g, ddof=1) for g in clean]
    ns = [len(g) for g in clean]
    for i in range(k):
        for j in range(i + 1, k):
            a = variances[i] / ns[i]
            b = variances[j] / ns[j]
            se2 = a + b
            if se2 <= 0:
                p = 1.0
            else:
                q = abs(means[i] - means[j]) / math.sqrt(se2) * math.sqrt(2.0)
                denom = (a * a) / max(ns[i] - 1, 1) + (b * b) / max(ns[j] - 1, 1)
                df = (se2 * se2) / denom if denom > 0 else np.inf
                p = float(studentized_range.sf(q, k, df))
            pmat[i, j] = pmat[j, i] = p
    return pmat



def maximal_cliques_from_nonsignificance(pmat: np.ndarray, alpha: float = 0.05) -> List[List[int]]:
    n = pmat.shape[0]
    adj = {i: {j for j in range(n) if j != i and pmat[i, j] > alpha} for i in range(n)}
    cliques: List[set] = []
    def bronk(R: set, P: set, X: set):
        if not P and not X:
            cliques.append(set(R))
            return
        for v in list(P):
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    bronk(set(), set(range(n)), set())
    # Include isolated vertices; Bron-Kerbosch already does, but keep defensive logic.
    present = set().union(*cliques) if cliques else set()
    for i in range(n):
        if i not in present:
            cliques.append({i})
    unique = []
    for c in cliques:
        if c and c not in unique:
            unique.append(c)
    unique.sort(key=lambda c: (min(c), -len(c), tuple(sorted(c))))
    return [sorted(c) for c in unique]










def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sampled = []
    for cls in TRAIN_CLASSES:
        idx = np.where(y_true == cls)[0]
        sampled.extend(rng.choice(idx, size=len(idx), replace=True).tolist())
    sampled = np.asarray(sampled, dtype=int)
    rng.shuffle(sampled)
    return sampled



def bootstrap_ci(result: Dict[str, object], n_bootstrap: int, random_state: int = 123):
    y_true = np.asarray(result["OOF_y_true"])
    pred = np.asarray(result["OOF_pred"])
    proba = np.asarray(result["OOF_proba"])
    point = calculate_metrics(y_true, pred, proba)
    rng = np.random.default_rng(random_state)
    distributions = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx = stratified_bootstrap_indices(y_true, rng)
        m = calculate_metrics(y_true[idx], pred[idx], proba[idx])
        for k, v in m.items():
            distributions[k].append(v)
    rows = []
    for k, vals in distributions.items():
        vals = np.asarray(vals, dtype=float)
        rows.append({
            "Metric": k,
            "Estimate": point[k],
            "CI_95_Lower": np.percentile(vals, 2.5),
            "CI_95_Upper": np.percentile(vals, 97.5),
            "Bootstrap_Iterations": n_bootstrap,
        })
    return pd.DataFrame(rows), pd.DataFrame(distributions)



def prepare_scaled_folds(X_values: np.ndarray, splits: Sequence[Tuple[np.ndarray, np.ndarray]]):
    prepared = []
    for train_idx, test_idx in splits:
        scaler = StandardScaler()
        prepared.append({
            "train_idx": np.asarray(train_idx),
            "test_idx": np.asarray(test_idx),
            "X_train": scaler.fit_transform(X_values[train_idx]),
            "X_test": scaler.transform(X_values[test_idx]),
        })
    return prepared



def evaluate_prepared_folds_for_labels(prepared, y_values: np.ndarray, base_model) -> Dict[str, float]:
    acc, f1s = [], []
    for fold in prepared:
        model = clone(base_model)
        model.fit(fold["X_train"], y_values[fold["train_idx"]])
        pred = model.predict(fold["X_test"])
        truth = y_values[fold["test_idx"]]
        acc.append(accuracy_score(truth, pred))
        f1s.append(f1_score(truth, pred, labels=TRAIN_CLASSES, average="macro", zero_division=0))
    return {"Accuracy": float(np.mean(acc)), "Macro_F1": float(np.mean(f1s))}



def permutation_test_configuration(
    X_values: np.ndarray,
    y_values: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    base_model,
    n_permutations: int,
    checkpoint_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_scaled_folds(X_values, splits)
    observed = evaluate_prepared_folds_for_labels(prepared, y_values, base_model)
    fingerprint = hashlib.sha256()
    fingerprint.update(np.asarray(X_values, dtype=np.float64).tobytes())
    fingerprint.update(np.asarray(y_values, dtype=np.int64).tobytes())
    fingerprint.update(f"{SEED}|{N_SPLITS}|{n_permutations}|{base_model.__class__.__name__}|fold".encode())
    fp = fingerprint.hexdigest()
    rng = np.random.default_rng(SEED + 5000)
    null_acc = np.full(n_permutations, np.nan)
    null_f1 = np.full(n_permutations, np.nan)
    completed = 0
    if checkpoint_path.exists():
        try:
            with checkpoint_path.open("rb") as f:
                ck = pickle.load(f)
            if ck.get("fingerprint") == fp and ck.get("n_permutations") == n_permutations:
                completed = int(ck.get("completed", 0))
                null_acc[:completed] = np.asarray(ck.get("accuracy", []))[:completed]
                null_f1[:completed] = np.asarray(ck.get("macro_f1", []))[:completed]
                if ck.get("rng_state") is not None:
                    rng.bit_generator.state = ck["rng_state"]
        except Exception:
            completed = 0
    for i in range(completed, n_permutations):
        perm_y = rng.permutation(y_values)
        m = evaluate_prepared_folds_for_labels(prepared, perm_y, base_model)
        null_acc[i] = m["Accuracy"]
        null_f1[i] = m["Macro_F1"]
        if (i + 1) % 100 == 0 or i == 0 or i + 1 == n_permutations:
            payload = {
                "fingerprint": fp,
                "n_permutations": n_permutations,
                "completed": i + 1,
                "accuracy": null_acc[:i+1].copy(),
                "macro_f1": null_f1[:i+1].copy(),
                "rng_state": rng.bit_generator.state,
            }
            tmp = checkpoint_path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, checkpoint_path)
    p_acc = (1 + np.sum(null_acc >= observed["Accuracy"])) / (n_permutations + 1)
    p_f1 = (1 + np.sum(null_f1 >= observed["Macro_F1"])) / (n_permutations + 1)
    summary = pd.DataFrame([
        {"Metric": "Accuracy", "Observed_fold_mean": observed["Accuracy"], "Permutation_p": p_acc, "N_permutations": n_permutations},
        {"Metric": "Macro_F1", "Observed_fold_mean": observed["Macro_F1"], "Permutation_p": p_f1, "N_permutations": n_permutations},
    ])
    dist = pd.DataFrame({"Permutation": np.arange(1, n_permutations + 1), "Accuracy": null_acc, "Macro_F1": null_f1})
    return summary, dist



def mean_sd(mean: float, sd: float, percent: bool = True, decimals: int = 2) -> str:
    mult = 100 if percent else 1
    return f"{mean * mult:.{decimals}f} ± {sd * mult:.{decimals}f}"



def make_table2(configuration_results: Dict[str, Dict[str, object]], out_path: Path) -> pd.DataFrame:
    summary = pd.DataFrame([summarize_fold_results(configuration_results[m]) for m in MODEL_ORDER])
    rows = []
    display_names = {"KNN": "kNN", "ExtraTrees": "ET", "RandomForest": "RF"}
    for _, r in summary.iterrows():
        rows.append({
            "Model": display_names[r["Model"]],
            "Accuracy (%)": mean_sd(r["Accuracy_mean"], r["Accuracy_std"]),
            "Precision (%)": mean_sd(r["Macro_Precision_mean"], r["Macro_Precision_std"]),
            "Recall (%)": mean_sd(r["Macro_Recall_mean"], r["Macro_Recall_std"]),
            "F1-score (%)": mean_sd(r["Macro_F1_mean"], r["Macro_F1_std"]),
            "ROC-AUC": mean_sd(r["Macro_ROC_AUC_mean"], r["Macro_ROC_AUC_std"], percent=False, decimals=4),
            "Inference Time (ms/subject)": f"{r['Inference_Time_Per_Subject_ms_mean']:.3f} ± {r['Inference_Time_Per_Subject_ms_std']:.3f}",
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    return table



def make_table3(bootstrap_summary: pd.DataFrame, permutation_summary: pd.DataFrame, n_permutations: int, out_path: Path) -> pd.DataFrame:
    b = bootstrap_summary.set_index("Metric")
    p = permutation_summary.set_index("Metric")
    rows = [
        {"Metric": "Accuracy", "Estimate": f"{b.loc['Accuracy','Estimate']*100:.2f}%", "95% CI": f"{b.loc['Accuracy','CI_95_Lower']*100:.2f}-{b.loc['Accuracy','CI_95_Upper']*100:.2f}"},
        {"Metric": "Macro F1-score", "Estimate": f"{b.loc['Macro_F1','Estimate']*100:.2f}%", "95% CI": f"{b.loc['Macro_F1','CI_95_Lower']*100:.2f}-{b.loc['Macro_F1','CI_95_Upper']*100:.2f}"},
        {"Metric": "Macro ROC-AUC", "Estimate": f"{b.loc['Macro_ROC_AUC','Estimate']:.4f}", "95% CI": f"{b.loc['Macro_ROC_AUC','CI_95_Lower']:.4f}-{b.loc['Macro_ROC_AUC','CI_95_Upper']:.4f}"},
        {"Metric": "Permutation p-value (Accuracy)", "Estimate": f"{p.loc['Accuracy','Permutation_p']:.4f}", "95% CI": ""},
        {"Metric": "Permutation p-value (Macro F1-score)", "Estimate": f"{p.loc['Macro_F1','Permutation_p']:.4f}", "95% CI": ""},
        {"Metric": "Number of permutations", "Estimate": str(n_permutations), "95% CI": ""},
    ]
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    return table



def make_class_wise_table (et_result: Dict[str, object], out_path: Path) -> pd.DataFrame:
    cdf = et_result["Class_metrics"]
    rows = []
    for cname in DISPLAY_CLASS_NAMES:
        sub = cdf[cdf["Class"] == cname]
        rows.append({
            "OSAS Class": "Simple Snoring" if cname == "Snoring" else f"{cname} OSAS",
            "Precision (%)": f"{sub['Precision'].mean()*100:.2f}",
            "Recall (%)": f"{sub['Recall'].mean()*100:.2f}",
            "F1-score (%)": f"{sub['F1_score'].mean()*100:.2f}",
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    return table



def ablation_label_and_features(name: str) -> Tuple[str, str]:
    mapping = {
        "A1_Basic_SpO2_statistics": ("A1", "Basic SpO₂ statistics"),
        "A2_A1_plus_PSD_features": ("A2", "A1 + PSD features"),
        "A3_A2_plus_BMI_ESS": ("A3", "A2 + BMI + ESS"),
        "A4_A3_plus_Hemogram_ODI_free": ("A4", "A3 + PDW + MPV + RDW-CV"),
        "A5_A4_plus_Gender": ("A5", "A4 + Gender"),
        "A6_A3_plus_Desaturation_features": ("A6", "A3 + Desaturation features"),
        "A7_A4_plus_Desaturation_features": ("A7", "A4 + Desaturation features"),
        "A8_A4_plus_Gender_Desaturation_features": ("A8", "A4 + Gender + Desaturation features"),
        "A9_A3_plus_ODI_sensitivity_only": ("A9", "A3 + ODI"),
        "A10_A4_plus_ODI_sensitivity_only": ("A10", "A4 + ODI"),
        "A11_A4_plus_Gender_Desaturation_ODI_sensitivity_only": ("A11", "A4 + Gender + Desaturation features + ODI"),
    }
    return mapping[name]



def make_table5(ablation_results: Dict[str, Dict[str, object]], out_path: Path) -> pd.DataFrame:
    rows = []
    for name, result in ablation_results.items():
        label, desc = ablation_label_and_features(name)
        s = summarize_fold_results(result)
        rows.append({
            "Configuration": label,
            "Feature set": desc,
            "No. of features": result["N_features"],
            "Accuracy mean ± SD (%)": mean_sd(s["Accuracy_mean"], s["Accuracy_std"]),
            "Macro F1-score mean ± SD (%)": mean_sd(s["Macro_F1_mean"], s["Macro_F1_std"]),
            "ROC-AUC mean ± SD": mean_sd(s["Macro_ROC_AUC_mean"], s["Macro_ROC_AUC_std"], percent=False, decimals=4),
        })
    rows.sort(key=lambda r: int(r["Configuration"][1:]))
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    return table



def event_minima(seg: np.ndarray, threshold: float = 75.0) -> List[float]:
    below = seg < threshold
    vals, start, in_event = [], 0, False
    for j, b in enumerate(below):
        if b and not in_event:
            start, in_event = j, True
        elif not b and in_event:
            vals.append(float(seg[start:j].min()))
            in_event = False
    if in_event:
        vals.append(float(seg[start:].min()))
    return vals



def fit_final_model(X_all: pd.DataFrame, y: pd.Series, selected_cols: Sequence[str], base_model):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all[list(selected_cols)].to_numpy(dtype=float))
    model = clone(base_model)
    model.fit(X_scaled, y.to_numpy())
    return scaler, X_scaled, model



def normalize_shap_values(shap_values, n_classes: int, n_features: int) -> List[np.ndarray]:
    if isinstance(shap_values, list):
        return [np.asarray(x) for x in shap_values]
    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[-1] == n_classes and arr.shape[1] == n_features:
            return [arr[:, :, i] for i in range(n_classes)]
        if arr.shape[0] == n_classes:
            return [arr[i] for i in range(n_classes)]
    raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")



def normalize_interactions(values, n_classes: int, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        return np.stack([np.asarray(v) for v in values], axis=-1)
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[..., np.newaxis]
    if arr.ndim == 4:
        if arr.shape[-1] == n_classes:
            return arr
        if arr.shape[0] == n_classes:
            return np.moveaxis(arr, 0, -1)
    raise ValueError(f"Unexpected SHAP interaction shape: {arr.shape}")



def configuration_label(name: str) -> str:
    return ablation_label_and_features(name)[0]



def save_result_numeric(result: Dict[str, object], config_name: str, model_name: str, base_dir: Path) -> None:
    out = base_dir / configuration_label(config_name) / model_name
    out.mkdir(parents=True, exist_ok=True)
    result["Fold_metrics"].to_csv(out / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    result["Class_metrics"].to_csv(out / "classwise_fold_metrics.csv", index=False, encoding="utf-8-sig")
    result["OOF_frame"].to_csv(out / "OOF_predictions_probabilities.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summarize_fold_results(result)]).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    with (out / "classification_reports.txt").open("w", encoding="utf-8") as f:
        f.write("\n\n".join(result["Reports"]))
    cm_rows = []
    roc_rows = []
    for item in result["Fold_store"]:
        fold = int(item["fold"])
        cm = confusion_matrix(item["y_test"], item["pred"], labels=DISPLAY_CLASS_IDS)
        for i, true_name in enumerate(DISPLAY_CLASS_NAMES):
            for j, pred_name in enumerate(DISPLAY_CLASS_NAMES):
                cm_rows.append({"Fold": fold, "True_Class": true_name, "Predicted_Class": pred_name, "Count": int(cm[i, j])})
        for cname, train_id in zip(DISPLAY_CLASS_NAMES, DISPLAY_CLASS_IDS):
            fpr, tpr, thresholds = roc_curve(item["y_bin"][:, train_id], item["proba"][:, train_id])
            auc_v = item["aucs"][train_id]
            for point_i, (fp, tp, th) in enumerate(zip(fpr, tpr, thresholds)):
                roc_rows.append({
                    "Fold": fold, "Class": cname, "Point": point_i,
                    "FPR": float(fp), "TPR": float(tp), "Threshold": float(th), "AUC": float(auc_v)
                })
    pd.DataFrame(cm_rows).to_csv(out / "confusion_matrices_all_folds_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(roc_rows).to_csv(out / "ROC_points_all_folds_classes.csv", index=False, encoding="utf-8-sig")









def save_preprocessing_summary(spo2_raw: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    seg_len = FS * SEGMENT_DURATION
    for i, row in spo2_raw.iterrows():
        raw = row.to_numpy(dtype=float)
        positive = raw[raw > 0]
        n_segments = len(positive) // seg_len
        retained = n_segments * seg_len
        rows.append({
            "Subject_Index": i,
            "Raw_Sample_Count": int(len(raw)),
            "Positive_Sample_Count": int(len(positive)),
            "Removed_Nonpositive_Count": int(len(raw) - len(positive)),
            "Complete_60s_Segments": int(n_segments),
            "Samples_Used_In_Complete_Segments": int(retained),
            "Positive_Tail_Samples_Not_Used": int(len(positive) - retained),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def save_feature_matrix(df: pd.DataFrame, X_all: pd.DataFrame, y: pd.Series,
                        valid_indices: Sequence[int], out_path: Path) -> None:
    out = pd.DataFrame({
        "Subject_Index": np.arange(len(y)),
        "Original_Data_Row": np.asarray(valid_indices),
        "Class_Label": y.to_numpy(),
        "Class": [TRAIN_CLASS_NAMES[i] for i in y.to_numpy()],
    })
    if "Subject" in df.columns:
        out.insert(1, "Subject", df["Subject"].to_numpy())
    out = pd.concat([out.reset_index(drop=True), X_all.reset_index(drop=True)], axis=1)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")


def save_fold_selection(result: Dict[str, object], out_path: Path) -> None:
    folds = result["Fold_store"]
    best = max(folds, key=lambda x: x["accuracy"])
    lowest = min(folds, key=lambda x: x["accuracy"])
    pd.DataFrame([{
        "Best_Fold": int(best["fold"]),
        "Best_Accuracy": float(best["accuracy"]),
        "Best_Macro_F1": float(best["macro_f1"]),
        "Lowest_Fold": int(lowest["fold"]),
        "Lowest_Accuracy": float(lowest["accuracy"]),
        "Lowest_Macro_F1": float(lowest["macro_f1"]),
    }]).to_csv(out_path, index=False, encoding="utf-8-sig")


def save_feature_importance_numeric(model, feature_codes: Sequence[str], out_path: Path) -> pd.DataFrame:
    out = pd.DataFrame({
        "Feature_Code": list(feature_codes),
        "Feature": [FEATURE_LABEL_MAP.get(c, c) for c in feature_codes],
        "Importance": np.asarray(model.feature_importances_, dtype=float),
    }).sort_values("Importance", ascending=False)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def save_shap_numeric(model, X_scaled: np.ndarray, feature_codes: Sequence[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = [FEATURE_LABEL_MAP.get(c, c) for c in feature_codes]
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_scaled)
    shap_by_class = normalize_shap_values(shap_values, len(TRAIN_CLASS_NAMES), len(feature_names))
    shap_stack = np.stack(shap_by_class, axis=-1)
    np.savez_compressed(
        out_dir / "shap_values_and_scaled_features.npz",
        shap_values=shap_stack,
        X_scaled=np.asarray(X_scaled, dtype=float),
        feature_codes=np.asarray(feature_codes, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
        training_class_names=np.asarray(TRAIN_CLASS_NAMES, dtype=object),
        display_class_names=np.asarray(DISPLAY_CLASS_NAMES, dtype=object),
    )
    pd.DataFrame(X_scaled, columns=feature_names).to_csv(
        out_dir / "scaled_feature_matrix.csv", index_label="Subject_Index", encoding="utf-8-sig")

    long_rows = []
    for class_id, cname in enumerate(TRAIN_CLASS_NAMES[:shap_stack.shape[-1]]):
        for fi, (fcode, fname) in enumerate(zip(feature_codes, feature_names)):
            for subject_idx, val in enumerate(shap_stack[:, fi, class_id]):
                long_rows.append({
                    "Subject_Index": subject_idx,
                    "Training_Class_Index": class_id,
                    "Training_Class": cname,
                    "Feature_Code": fcode,
                    "Feature": fname,
                    "SHAP_Value": float(val),
                })
    pd.DataFrame(long_rows).to_csv(out_dir / "shap_values_long.csv", index=False, encoding="utf-8-sig")

    shap_matrix = np.vstack([np.mean(np.abs(sv), axis=0) for sv in shap_by_class])
    order = np.argsort(shap_matrix.sum(axis=0))[::-1]
    mean_abs = pd.DataFrame(
        shap_matrix[:, order].T,
        index=[feature_names[i] for i in order],
        columns=TRAIN_CLASS_NAMES[:shap_matrix.shape[0]],
    )
    mean_abs = mean_abs[[c for c in DISPLAY_CLASS_NAMES if c in mean_abs.columns]]
    mean_abs.to_csv(out_dir / "shap_mean_abs_by_class.csv", encoding="utf-8-sig")

    interaction_values = explainer.shap_interaction_values(X_scaled)
    interaction_stack = normalize_interactions(
        interaction_values, len(TRAIN_CLASS_NAMES), len(feature_names)
    )
    np.savez_compressed(
        out_dir / "shap_interaction_values_full.npz",
        interaction_values=interaction_stack,
        feature_codes=np.asarray(feature_codes, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
        training_class_names=np.asarray(TRAIN_CLASS_NAMES, dtype=object),
        display_class_names=np.asarray(DISPLAY_CLASS_NAMES, dtype=object),
    )

    inter_abs = np.mean(np.abs(interaction_stack), axis=(0, 3))
    pd.DataFrame(inter_abs, index=feature_names, columns=feature_names).to_csv(
        out_dir / "shap_interaction_mean_abs_matrix.csv", encoding="utf-8-sig")
    total_inter = inter_abs.sum()
    inter_pct = 100.0 * inter_abs / total_inter if total_inter > 0 else inter_abs
    pd.DataFrame(inter_pct, index=feature_names, columns=feature_names).to_csv(
        out_dir / "shap_interaction_percent_matrix.csv", encoding="utf-8-sig")

    preferred_codes = ["min_spo2", "std_spo2", "psd_mean", "mean_spo2"]
    code_to_idx = {code: i for i, code in enumerate(feature_codes)}
    selected_codes = [c for c in preferred_codes if c in code_to_idx]
    if len(selected_codes) < 4:
        fallback = ["nominal_psd_sum", "psd_max"]
        selected_codes += [c for c in fallback if c in code_to_idx and c not in selected_codes]
        selected_codes = selected_codes[:4]
    idx = [code_to_idx[c] for c in selected_codes]
    selected_inter = interaction_stack[:, idx, :, :][:, :, idx, :]
    selected_features = X_scaled[:, idx]
    selected_names = [feature_names[i] for i in idx]
    np.savez_compressed(
        out_dir / "shap_interaction_selected_spo2_features.npz",
        interaction_values=selected_inter,
        X_scaled=selected_features,
        feature_codes=np.asarray(selected_codes, dtype=object),
        feature_names=np.asarray(selected_names, dtype=object),
        training_class_names=np.asarray(TRAIN_CLASS_NAMES, dtype=object),
    )
    pd.DataFrame({
        "Order": np.arange(1, len(selected_codes) + 1),
        "Feature_Code": selected_codes,
        "Feature": selected_names,
        "Source_Feature_Index": idx,
    }).to_csv(out_dir / "shap_interaction_selected_feature_order.csv", index=False, encoding="utf-8-sig")

    point_rows = []
    for class_idx, class_name in enumerate(TRAIN_CLASS_NAMES[:selected_inter.shape[-1]]):
        for col_idx in range(len(idx)):
            for row_idx in range(len(idx)):
                raw_vals = selected_inter[:, col_idx, row_idx, class_idx]
                display_vals = raw_vals.copy() if col_idx == row_idx else raw_vals * 2.0
                color_vals = selected_features[:, row_idx]
                for subject_idx, (raw_v, display_v, color_v) in enumerate(zip(raw_vals, display_vals, color_vals)):
                    point_rows.append({
                        "Subject_Index": subject_idx,
                        "Training_Class_Index": class_idx,
                        "Training_Class_Name": class_name,
                        "Column_Feature_Code": selected_codes[col_idx],
                        "Column_Feature": selected_names[col_idx],
                        "Row_Feature_Code": selected_codes[row_idx],
                        "Row_Feature": selected_names[row_idx],
                        "Raw_SHAP_Interaction": float(raw_v),
                        "Displayed_SHAP_Interaction": float(display_v),
                        "Color_Feature_Value_Scaled": float(color_v),
                    })
    pd.DataFrame(point_rows).to_csv(
        out_dir / "shap_interaction_values_selected.csv", index=False, encoding="utf-8-sig")


def resolve_requested_configs(feature_sets: Dict[str, List[str]], value: str) -> List[str]:
    req = str(value).strip().upper()
    if req == "ALL":
        return list(feature_sets.keys())
    lookup = {configuration_label(k).upper(): k for k in feature_sets}
    if req not in lookup:
        raise ValueError(f"Unknown configuration: {value}. Use A1..A11 or ALL.")
    return [lookup[req]]


def main() -> None:
    args = parse_args()
    setup_reproducibility()
    data_path = Path(args.data).resolve()
    dirs = ensure_dirs(Path(args.output).resolve())

    print("Loading data:", data_path)
    df, spo2_raw, X_all, y, valid_indices, gender_encoder = load_and_prepare_data(data_path)
    print("Class order:", list(DISPLAY_CLASS_NAMES))
    print("Class distribution:", df["SINIF"].value_counts().reindex(DISPLAY_CLASS_NAMES).to_dict())
    print("Valid subjects:", len(y))

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    shared_splits = list(cv.split(np.zeros(len(y)), y.to_numpy()))
    feature_sets = make_feature_sets(X_all)
    models = make_models()

    feature_rows = []
    for cfg_name, cols in feature_sets.items():
        label, desc = ablation_label_and_features(cfg_name)
        for pos, code in enumerate(cols, start=1):
            feature_rows.append({
                "Configuration": label,
                "Configuration_Key": cfg_name,
                "Description": desc,
                "Feature_Order": pos,
                "Feature_Code": code,
                "Feature_Display_Name": FEATURE_LABEL_MAP.get(code, code),
            })
    pd.DataFrame(feature_rows).to_csv(
        dirs["tables"] / "feature_configuration_definitions.csv", index=False, encoding="utf-8-sig")

    save_feature_matrix(df, X_all, y, valid_indices, dirs["features"] / "subject_level_feature_matrix.csv")
    save_preprocessing_summary(spo2_raw, dirs["features"] / "preprocessing_summary.csv")
  



    results: Dict[str, Dict[str, Dict[str, object]]] = {}
    summary_rows = []
    for cfg_name, cols in feature_sets.items():
        label, desc = ablation_label_and_features(cfg_name)
        results[cfg_name] = {}
        for model_name in MODEL_ORDER:
            print(f"Evaluating {label} / {model_name} ...")
            res = evaluate_configuration(
                X_all, y, cols, model_name, models[model_name], shared_splits, valid_indices
            )
            res["Configuration"] = label
            res["Configuration_Key"] = cfg_name
            results[cfg_name][model_name] = res
            save_result_numeric(res, cfg_name, model_name, dirs["cv"])
            sm = summarize_fold_results(res)
            summary_rows.append({
                "Configuration": label,
                "Configuration_Key": cfg_name,
                "Description": desc,
                "Model": model_name,
                **{k: v for k, v in sm.items() if k != "Model"},
                **{f"OOF_{k}": v for k, v in res["OOF_metrics"].items()},
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(dirs["tables"] / "all_configurations_all_models_summary.csv", index=False, encoding="utf-8-sig")
    summary_df.sort_values(
        ["Accuracy_mean", "Macro_F1_mean", "Macro_ROC_AUC_mean"], ascending=False
    ).to_csv(dirs["tables"] / "all_configurations_all_models_ranked.csv", index=False, encoding="utf-8-sig")

    for cfg_name, model_results in results.items():
        label = configuration_label(cfg_name)
        make_table2(model_results, dirs["tables"] / f"model_comparison_{label}.csv")
        for model_name, res in model_results.items():
            make_class_wise_table (res, dirs["tables"] / f"classwise_performance_{label}_{model_name}.csv")
            save_fold_selection(
                res,
                dirs["cv"] / label / model_name / "best_lowest_fold_by_accuracy.csv",
            )

    et_results = {cfg: model_results["ExtraTrees"] for cfg, model_results in results.items()}
    make_table5(et_results, dirs["tables"] / "ablation_sensitivity_ExtraTrees.csv")

    # Interpretability values are saved for every ExtraTrees feature configuration.
    for cfg_name, cols in feature_sets.items():
        label = configuration_label(cfg_name)
        print(f"Saving interpretation values for {label} / ExtraTrees ...")
        scaler, X_scaled, model = fit_final_model(X_all, y, cols, models["ExtraTrees"])
        out_dir = dirs["interpretability"] / label / "ExtraTrees"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(X_scaled, columns=[FEATURE_LABEL_MAP.get(c, c) for c in cols]).to_csv(
            out_dir / "scaled_feature_matrix.csv", index_label="Subject_Index", encoding="utf-8-sig")
        save_feature_importance_numeric(model, cols, out_dir / "feature_importance.csv")
        if not args.skip_shap:
            save_shap_numeric(model, X_scaled, cols, out_dir)

    if args.bootstrap_config:
        cfgs = resolve_requested_configs(feature_sets, args.bootstrap_config)
        model_names = MODEL_ORDER if str(args.bootstrap_model).upper() == "ALL" else [args.bootstrap_model]
        all_rows = []
        for cfg_name in cfgs:
            label = configuration_label(cfg_name)
            for model_name in model_names:
                print(f"Bootstrap {label} / {model_name}: {args.bootstrap} resamples ...")
                summary, dist = bootstrap_ci(results[cfg_name][model_name], args.bootstrap, random_state=123)
                out_dir = dirs["bootstrap"] / label / model_name
                out_dir.mkdir(parents=True, exist_ok=True)
                summary.to_csv(out_dir / "bootstrap_summary.csv", index=False, encoding="utf-8-sig")
                dist.to_csv(out_dir / "bootstrap_distribution.csv", index=False, encoding="utf-8-sig")
                for _, row in summary.iterrows():
                    all_rows.append({"Configuration": label, "Model": model_name, **row.to_dict()})
        pd.DataFrame(all_rows).to_csv(dirs["tables"] / "requested_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    if args.permutation_config:
        cfgs = resolve_requested_configs(feature_sets, args.permutation_config)
        model_names = MODEL_ORDER if str(args.permutation_model).upper() == "ALL" else [args.permutation_model]
        all_rows = []
        for cfg_name in cfgs:
            label = configuration_label(cfg_name)
            cols = feature_sets[cfg_name]
            for model_name in model_names:
                print(f"Permutation {label} / {model_name}: {args.permutations} permutations ...")
                out_dir = dirs["permutation"] / label / model_name
                out_dir.mkdir(parents=True, exist_ok=True)
                summary, dist = permutation_test_configuration(
                    X_all[cols].to_numpy(dtype=float), y.to_numpy(), shared_splits,
                    models[model_name], args.permutations,
                    out_dir / "permutation_checkpoint.pkl",
                )
                summary.to_csv(out_dir / "permutation_summary.csv", index=False, encoding="utf-8-sig")
                dist.to_csv(out_dir / "permutation_distribution.csv", index=False, encoding="utf-8-sig")
                for _, row in summary.iterrows():
                    all_rows.append({"Configuration": label, "Model": model_name, **row.to_dict()})
        pd.DataFrame(all_rows).to_csv(dirs["tables"] / "requested_permutation_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "data": str(data_path),
        "seed": SEED,
        "sampling_frequency_hz": FS,
        "segment_duration_s": SEGMENT_DURATION,
        "cv_splits": N_SPLITS,
        "scaler_mode": "fold",
        "display_class_order": DISPLAY_CLASS_NAMES.tolist(),
        "training_label_order": TRAIN_CLASS_NAMES.tolist(),
        "valid_subjects": int(len(y)),
        "feature_configurations": {configuration_label(k): v for k, v in feature_sets.items()},
        "models": {
            "KNN": {"n_neighbors": 3, "metric": "manhattan"},
            "RandomForest": {"random_state": SEED},
            "ExtraTrees": {"random_state": SEED},
        },
        "bootstrap_config": args.bootstrap_config,
        "bootstrap_model": args.bootstrap_model,
        "bootstrap_iterations": args.bootstrap,
        "permutation_config": args.permutation_config,
        "permutation_model": args.permutation_model,
        "permutation_iterations": args.permutations,
        "shap_calculated": bool(not args.skip_shap),
    }
    with (dirs["logs"] / "method_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


    print("\nCompleted.")
    print("Output directory:", dirs["root"])


if __name__ == "__main__":
    main()

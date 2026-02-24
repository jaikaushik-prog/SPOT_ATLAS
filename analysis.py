"""
prmon Anomaly Detection Pipeline — Combined Time-Series
=========================================================
Builds a single combined time-series by interleaving baseline segments with
four injected anomaly windows (subtle CPU, extreme CPU, hard memory, extreme
memory).  Three novelty-detection methods (Z-Score, LOF, One-Class SVM) are
trained on the clean baseline portion and evaluated against ground-truth
labels using precision, recall, and F1.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import precision_score, recall_score, f1_score

# ── 1. Data Loading ──────────────────────────────────────────────────────────

FEATURES = ["pss", "rss", "vmem", "utime", "stime", "nthreads"]

ALL_NUMERIC_COLS = ["pss","rss","swap","vmem","rchar","read_bytes","wchar",
                    "write_bytes","rx_bytes","rx_packets","tx_bytes","tx_packets",
                    "stime","utime","nprocs","nthreads","wtime"]

def load_prmon(path):
    df = pd.read_csv(path, sep=r"\s+", on_bad_lines="skip")
    for col in ALL_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df

baseline     = load_prmon("baseline_long.txt")
cpu_extreme  = load_prmon("anomaly_cpu_long.txt")
mem_extreme  = load_prmon("anomaly_mem_long.txt")
cpu_subtle   = load_prmon("anomaly_cpu_subtle.txt")
mem_hard     = load_prmon("anomaly_mem_hard.txt")

print(f"Loaded  baseline:     {len(baseline)} rows")
print(f"Loaded  cpu_extreme:  {len(cpu_extreme)} rows")
print(f"Loaded  cpu_subtle:   {len(cpu_subtle)} rows")
print(f"Loaded  mem_extreme:  {len(mem_extreme)} rows")
print(f"Loaded  mem_hard:     {len(mem_hard)} rows\n")

# ── 2. Build Combined Time-Series ────────────────────────────────────────────
#
# UPDATED per mentor feedback: test moderate anomalies (a few sigmas from baseline).
#
# Layout: [B0] [3sig] [B1] [5sig] [B2] [drift] [B3] [ext_cpu] [B4] [ext_mem] [B5]
# Labels:  0     1      0     2     0      3      0      4        0      5       0
#
# Windows 1-3 are SYNTHETIC moderate anomalies created by perturbing baseline PSS.
# Windows 4-5 are REAL extreme anomalies from stress-ng for contrast.

np.random.seed(42)

# -- Baseline statistics for synthetic anomaly generation --
pss_mean = baseline["pss"].mean()
pss_std  = baseline["pss"].std()
print(f"Baseline PSS: mean={pss_mean:.2f}, std={pss_std:.4f}")

# -- Helper: create a synthetic anomaly segment from baseline rows --
def make_synthetic(base_rows, pss_values, label):
    """Clone baseline rows and replace PSS with controlled values."""
    seg = base_rows.copy()
    seg["pss"] = pss_values
    seg["label"] = label
    return seg

# -- Segment definitions --
N_MOD = 40   # moderate anomaly window size
N_DFT = 50   # drift window size

# Baseline segments (gaps between anomaly windows)
seg_b0 = baseline.iloc[0:80].copy();        seg_b0["label"] = 0
seg_b1 = baseline.iloc[80:160].copy();      seg_b1["label"] = 0
seg_b2 = baseline.iloc[160:240].copy();     seg_b2["label"] = 0
seg_b3 = baseline.iloc[240:320].copy();     seg_b3["label"] = 0
seg_b4 = baseline.iloc[320:400].copy();     seg_b4["label"] = 0
seg_b5 = baseline.iloc[400:460].copy();     seg_b5["label"] = 0

# Window 1: 3-sigma anomaly (PSS shifted UP by 3*std, with natural noise)
pss_3sig = pss_mean + 3 * pss_std + np.random.normal(0, pss_std * 0.5, N_MOD)
seg_3sig = make_synthetic(baseline.iloc[0:N_MOD], pss_3sig, label=1)

# Window 2: 5-sigma anomaly (PSS shifted UP by 5*std, with natural noise)
pss_5sig = pss_mean + 5 * pss_std + np.random.normal(0, pss_std * 0.5, N_MOD)
seg_5sig = make_synthetic(baseline.iloc[0:N_MOD], pss_5sig, label=2)

# Window 3: Gradual drift (PSS linearly increases from baseline mean to +8*std)
pss_drift = np.linspace(pss_mean, pss_mean + 8 * pss_std, N_DFT)
pss_drift += np.random.normal(0, pss_std * 0.3, N_DFT)   # add slight noise
seg_drift = make_synthetic(baseline.iloc[0:N_DFT], pss_drift, label=3)

# Window 4: Extreme CPU (real stress-ng data)
seg_cpu_ext = cpu_extreme.iloc[100:150].copy()  # 50 pts
seg_cpu_ext["label"] = 4

# Window 5: Extreme memory (real stress-ng data)
seg_mem_ext = mem_extreme.iloc[80:140].copy()   # 60 pts
seg_mem_ext["label"] = 5

# -- Assemble combined series --
segments = [seg_b0, seg_3sig, seg_b1, seg_5sig, seg_b2, seg_drift,
            seg_b3, seg_cpu_ext, seg_b4, seg_mem_ext, seg_b5]
seg_names = ["B0", "3sig", "B1", "5sig", "B2", "drift",
             "B3", "cpu_ext", "B4", "mem_ext", "B5"]

combined = pd.concat(segments, ignore_index=True)
combined["t"] = np.arange(len(combined))
combined["is_anomaly"] = (combined["label"] > 0).astype(int)

# Compute segment boundaries
boundaries = {}
offset = 0
for seg, name in zip(segments, seg_names):
    boundaries[name] = (offset, offset + len(seg))
    offset += len(seg)

sig3_start, sig3_end   = boundaries["3sig"]
sig5_start, sig5_end   = boundaries["5sig"]
drift_start, drift_end = boundaries["drift"]
cpu_ext_start, cpu_ext_end = boundaries["cpu_ext"]
mem_ext_start, mem_ext_end = boundaries["mem_ext"]

n_normal = (combined["label"] == 0).sum()
n_anom   = (combined["label"] > 0).sum()
print(f"\nCombined series: {len(combined)} rows")
print(f"  Normal points:          {n_normal}")
print(f"  3-sigma anomaly:        {(combined['label']==1).sum()}  (t={sig3_start}..{sig3_end-1})  PSS~{pss_3sig.mean():.0f}")
print(f"  5-sigma anomaly:        {(combined['label']==2).sum()}  (t={sig5_start}..{sig5_end-1})  PSS~{pss_5sig.mean():.0f}")
print(f"  Gradual drift:          {(combined['label']==3).sum()}  (t={drift_start}..{drift_end-1})  PSS {pss_drift[0]:.0f}->{pss_drift[-1]:.0f}")
print(f"  Extreme CPU (real):     {(combined['label']==4).sum()}  (t={cpu_ext_start}..{cpu_ext_end-1})")
print(f"  Extreme memory (real):  {(combined['label']==5).sum()}  (t={mem_ext_start}..{mem_ext_end-1})")
print()

# ── 3. Scaling (fit on baseline-only portion) ────────────────────────────────

# Training data: only the clean baseline segments
baseline_mask = combined["label"] == 0
train_X = combined.loc[baseline_mask, FEATURES].values

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)

# Transform the entire combined series
all_X_scaled = scaler.transform(combined[FEATURES].values)

# ── 4. Method 1 – Z-Score ────────────────────────────────────────────────────

ZSCORE_THRESH = 3.0

def zscore_detect(X, threshold=ZSCORE_THRESH):
    return np.where(np.any(np.abs(X) > threshold, axis=1), -1, 1)

import time as _time

timing_results = {}

# ── Method 1: Z-Score ──
t0 = _time.perf_counter()
combined["zscore"] = zscore_detect(all_X_scaled)
t1 = _time.perf_counter()
timing_results["Z-Score"] = {"train": 0.0, "predict": t1 - t0}

# ── 5. Method 2 – Local Outlier Factor (novelty detection) ───────────────────

t0 = _time.perf_counter()
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02, novelty=True)
lof.fit(train_X_scaled)
t1 = _time.perf_counter()
combined["lof"] = lof.predict(all_X_scaled)
t2 = _time.perf_counter()
timing_results["LOF"] = {"train": t1 - t0, "predict": t2 - t1}

# ── 6. Method 3 – One-Class SVM ─────────────────────────────────────────────

t0 = _time.perf_counter()
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
ocsvm.fit(train_X_scaled)
t1 = _time.perf_counter()
combined["ocsvm"] = ocsvm.predict(all_X_scaled)
t2 = _time.perf_counter()
timing_results["OC-SVM"] = {"train": t1 - t0, "predict": t2 - t1}

# ── 6b. Method 4 -- Elliptic Envelope (Gaussian covariance) ──────────────────

t0 = _time.perf_counter()
ee = EllipticEnvelope(contamination=0.02, support_fraction=0.999)
ee.fit(train_X_scaled)
t1 = _time.perf_counter()
combined["elliptic"] = ee.predict(all_X_scaled)
t2 = _time.perf_counter()
timing_results["Elliptic Env."] = {"train": t1 - t0, "predict": t2 - t1}

# ── 6c. Method 5 -- Autoencoder (reconstruction error) ───────────────────────
# Train a small MLP to reconstruct baseline features.  Anomalies have high
# reconstruction error because the network only learned normal patterns.

t0 = _time.perf_counter()
autoencoder = MLPRegressor(
    hidden_layer_sizes=(32, 8, 32),   # bottleneck architecture
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
)
autoencoder.fit(train_X_scaled, train_X_scaled)   # input == target
t1 = _time.perf_counter()

# Compute per-sample reconstruction error (MSE)
reconstructed = autoencoder.predict(all_X_scaled)
recon_error = np.mean((all_X_scaled - reconstructed) ** 2, axis=1)

# Threshold: mean + 3*std of baseline reconstruction error
baseline_recon = recon_error[baseline_mask]
ae_threshold = baseline_recon.mean() + 3 * baseline_recon.std()
combined["autoenc"] = np.where(recon_error > ae_threshold, -1, 1)
t2 = _time.perf_counter()
timing_results["Autoencoder"] = {"train": t1 - t0, "predict": t2 - t1}

print(f"Autoencoder reconstruction threshold: {ae_threshold:.4f}")
print(f"  Baseline recon error: mean={baseline_recon.mean():.4f}, std={baseline_recon.std():.4f}")
print(f"  Max anomaly recon error: {recon_error[~baseline_mask].max():.2f}")
print()

# ── 6d. Method 6 -- Sliding-Window CUSUM Detector ────────────────────────────
# Motivation: per-point methods (Z-Score, LOF, etc.) miss gradual drift because
# individual points look normal.  A sliding window smooths noise and tests whether
# the LOCAL MEAN has shifted away from the baseline mean -- catching trends that
# point-wise methods cannot.
#
# CUSUM-inspired: flag a point if its rolling-window average deviates more than
# `cusum_zscore_thresh` standard deviations from the baseline mean.

WINDOW_SIZE = 15        # rolling window width
CUSUM_ZSCORE_THRESH = 2.0  # z-score on rolling means (lower = more sensitive)

t0 = _time.perf_counter()

# Compute statistics of baseline rolling means (the "noise floor" for windows)
baseline_pss = combined.loc[baseline_mask, "pss"].values
baseline_rolling = pd.Series(baseline_pss).rolling(WINDOW_SIZE, min_periods=1).mean()
roll_mean_base = baseline_rolling.mean()
roll_std_base  = baseline_rolling.std()

# Rolling mean over the entire combined series
pss_rolling = combined["pss"].rolling(WINDOW_SIZE, min_periods=1).mean()

# Flag: rolling mean deviates by > CUSUM_ZSCORE_THRESH sigma from baseline rolling mean
cusum_zscore = (pss_rolling - roll_mean_base) / roll_std_base
combined["cusum"] = np.where(np.abs(cusum_zscore) > CUSUM_ZSCORE_THRESH, -1, 1)

t1 = _time.perf_counter()
timing_results["CUSUM-Window"] = {"train": 0.0, "predict": t1 - t0}

n_cusum = (combined["cusum"] == -1).sum()
print(f"Sliding-Window CUSUM: window={WINDOW_SIZE}, z-thresh={CUSUM_ZSCORE_THRESH}")
print(f"  Baseline rolling mean: {roll_mean_base:.2f}, std: {roll_std_base:.4f}")
print(f"  Flagged: {n_cusum} anomalies")
print()

# ── 7. Evaluation ────────────────────────────────────────────────────────────

y_true = combined["is_anomaly"].values  # 1 = anomaly, 0 = normal

print("=" * 72)
print(f"{'Detection Results — Combined Time-Series':^72}")
print("=" * 72)
print(f"{'Method':<18} {'Precision':>10} {'Recall':>10} {'F1':>10}   {'TP':>5} {'FP':>5} {'FN':>5}")
print("-" * 72)

methods = [("zscore", "Z-Score"), ("lof", "LOF"), ("ocsvm", "OC-SVM"),
           ("elliptic", "Elliptic Env."), ("autoenc", "Autoencoder"),
           ("cusum", "CUSUM-Window")]
results = {}

for col, label in methods:
    y_pred = (combined[col] == -1).astype(int).values  # 1 = flagged anomaly
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    results[col] = {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}
    print(f"{label:<18} {p:>10.3f} {r:>10.3f} {f:>10.3f}   {tp:>5} {fp:>5} {fn:>5}")

print("=" * 72)
n_total = len(combined)
n_anom = int(y_true.sum())
print(f"Total: {n_total} points  |  Normal: {n_total - n_anom}  |  Anomalous: {n_anom}")
print()

# ── 7b. False Positive Investigation ─────────────────────────────────────────

for col, label in methods:
    fp_mask = (combined[col] == -1) & (combined["is_anomaly"] == 0)
    n_fp = fp_mask.sum()
    if n_fp > 0 and n_fp <= 20:   # only print detail for small FP counts
        print(f"── {label}: {n_fp} false positive(s) in baseline ──")
        fp_indices = combined.loc[fp_mask].index.tolist()
        for idx in fp_indices:
            row = combined.iloc[idx]
            z_vals = all_X_scaled[idx]
            # Find which features exceeded threshold
            exceeded = [(FEATURES[i], f"z={z_vals[i]:.2f}") for i in range(len(FEATURES))
                        if abs(z_vals[i]) > ZSCORE_THRESH]
            raw_vals = {f: row[f] for f in FEATURES}
            print(f"  t={int(row['t']):>3}  segment=baseline  "
                  f"pss={int(row['pss'])}  utime={int(row['utime'])}  nthreads={int(row['nthreads'])}")
            if exceeded:
                print(f"         z-score trigger: {exceeded}")
            else:
                print(f"         (flagged by {label}, not z-score)")
        print()

# ── 7c. Feature Ablation Study ───────────────────────────────────────────────
# Test which features matter most by running all 3 detectors on different subsets.

FEATURE_SETS = {
    "pss only":                    ["pss"],
    "utime only":                  ["utime"],
    "nthreads only":               ["nthreads"],
    "swap only":                   ["swap"],
    "rss only":                    ["rss"],
    "pss + utime":                 ["pss", "utime"],
    "pss + nthreads":              ["pss", "nthreads"],
    "utime + nthreads":            ["utime", "nthreads"],
    "pss + utime + nthreads":      ["pss", "utime", "nthreads"],
    "memory (pss+rss+vmem+swap)":  ["pss", "rss", "vmem", "swap"],
    "I/O (rchar+wchar+rx+tx)":     ["rchar", "wchar", "rx_bytes", "tx_bytes"],
    "ALL 6 (baseline)":            FEATURES,
}

print("=" * 100)
print("                           Feature Ablation Study")
print("=" * 100)
print(f"{'Feature Set':<30} | {'Z-Score F1':>10} {'FN':>4} | {'LOF F1':>10} {'FN':>4} | {'OC-SVM F1':>10} {'FN':>4}")
print("-" * 100)

ablation_results = {}

for set_name, feat_list in FEATURE_SETS.items():
    # Ensure all features exist in the data
    valid = [f for f in feat_list if f in combined.columns]
    if not valid:
        continue

    # Scale using baseline-only
    sc = StandardScaler()
    train_sub = sc.fit_transform(combined.loc[baseline_mask, valid].values)
    all_sub   = sc.transform(combined[valid].values)

    row_results = {}

    # Z-Score
    z_pred = np.where(np.any(np.abs(all_sub) > ZSCORE_THRESH, axis=1), 1, 0)
    z_f1 = f1_score(y_true, z_pred, zero_division=0)
    z_fn = int(((z_pred == 0) & (y_true == 1)).sum())
    row_results["zscore"] = {"f1": z_f1, "fn": z_fn}

    # LOF
    lof_sub = LocalOutlierFactor(n_neighbors=20, contamination=0.02, novelty=True)
    lof_sub.fit(train_sub)
    l_pred_raw = lof_sub.predict(all_sub)
    l_pred = (l_pred_raw == -1).astype(int)
    l_f1 = f1_score(y_true, l_pred, zero_division=0)
    l_fn = int(((l_pred == 0) & (y_true == 1)).sum())
    row_results["lof"] = {"f1": l_f1, "fn": l_fn}

    # OC-SVM
    oc_sub = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    oc_sub.fit(train_sub)
    o_pred_raw = oc_sub.predict(all_sub)
    o_pred = (o_pred_raw == -1).astype(int)
    o_f1 = f1_score(y_true, o_pred, zero_division=0)
    o_fn = int(((o_pred == 0) & (y_true == 1)).sum())
    row_results["ocsvm"] = {"f1": o_f1, "fn": o_fn}

    ablation_results[set_name] = row_results

    print(f"{set_name:<30} | {z_f1:>10.3f} {z_fn:>4} | {l_f1:>10.3f} {l_fn:>4} | {o_f1:>10.3f} {o_fn:>4}")

print("=" * 100)
print("FN = false negatives (missed anomalies out of 220)")
print()

# ── 7d. Per-Window Recall Breakdown ──────────────────────────────────────────

WINDOW_LABELS = {
    1: ("3-sigma (synthetic)",  sig3_start,    sig3_end),
    2: ("5-sigma (synthetic)",  sig5_start,    sig5_end),
    3: ("Gradual drift",        drift_start,   drift_end),
    4: ("Extreme CPU (real)",   cpu_ext_start, cpu_ext_end),
    5: ("Extreme Mem (real)",   mem_ext_start, mem_ext_end),
}

print("=" * 110)
print("             Per-Window Recall (detected / total per anomaly window)")
print("=" * 110)
print(f"{'Window':<24} {'Total':>6} | {'Z-Score':>10} | {'LOF':>10} | {'OC-SVM':>10} | {'Elliptic':>10} | {'Autoenc':>10}")
print("-" * 110)

per_window_recall = {}
for lbl_val, (wname, ws, we) in WINDOW_LABELS.items():
    window_mask = combined["label"] == lbl_val
    n_window = int(window_mask.sum())
    row_data = {}
    parts = []
    for col, mname in methods:
        detected = int(((combined[col] == -1) & window_mask).sum())
        recall_w = detected / n_window if n_window > 0 else 0
        row_data[col] = {"detected": detected, "total": n_window, "recall": recall_w}
        parts.append(f"{detected}/{n_window} ({recall_w:.0%})")
    per_window_recall[wname] = row_data
    print(f"{wname:<24} {n_window:>6} | {parts[0]:>10} | {parts[1]:>10} | {parts[2]:>10} | {parts[3]:>10} | {parts[4]:>10}")

print("=" * 110)
print()

# ── 7e. Feature Ablation Heatmap (Plot 5) ────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7))

set_names = list(ablation_results.keys())
method_names = ["Z-Score", "LOF", "OC-SVM"]
method_keys  = ["zscore", "lof", "ocsvm"]

heatmap_data = np.array([
    [ablation_results[s][mk]["f1"] for mk in method_keys]
    for s in set_names
])

im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

# Labels
ax.set_xticks(range(len(method_names)))
ax.set_xticklabels(method_names, fontsize=11)
ax.set_yticks(range(len(set_names)))
ax.set_yticklabels(set_names, fontsize=10)

# Annotate cells with F1 values
for i in range(len(set_names)):
    for j in range(len(method_names)):
        val = heatmap_data[i, j]
        color = "white" if val < 0.5 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

ax.set_title("Feature Ablation — F1 Score by Feature Set x Method", fontsize=13, pad=12)
fig.colorbar(im, ax=ax, label="F1 Score", shrink=0.8)
fig.tight_layout()
fig.savefig("plot5_feature_ablation_heatmap.png", dpi=150)
plt.close(fig)

# ── 7f. Z-Score Threshold Sensitivity ────────────────────────────────────────

thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
thresh_results = []

print("=" * 75)
print("         Z-Score Threshold Sensitivity Analysis")
print("=" * 75)
print(f"{'Threshold':>10} | {'Precision':>10} {'Recall':>10} {'F1':>10} | {'TP':>5} {'FP':>5} {'FN':>5}")
print("-" * 75)

for thr in thresholds:
    z_pred = np.where(np.any(np.abs(all_X_scaled) > thr, axis=1), 1, 0)
    p = precision_score(y_true, z_pred, zero_division=0)
    r = recall_score(y_true, z_pred, zero_division=0)
    f = f1_score(y_true, z_pred, zero_division=0)
    tp = int(((z_pred == 1) & (y_true == 1)).sum())
    fp = int(((z_pred == 1) & (y_true == 0)).sum())
    fn = int(((z_pred == 0) & (y_true == 1)).sum())
    thresh_results.append({"threshold": thr, "precision": p, "recall": r, "f1": f,
                           "tp": tp, "fp": fp, "fn": fn})
    marker = " <-- current" if thr == ZSCORE_THRESH else ""
    print(f"{thr:>10.1f} | {p:>10.3f} {r:>10.3f} {f:>10.3f} | {tp:>5} {fp:>5} {fn:>5}{marker}")

# Find optimal threshold
best = max(thresh_results, key=lambda x: (x["f1"], x["precision"]))
print("-" * 75)
print(f"Best threshold: z = {best['threshold']} (F1 = {best['f1']:.3f}, P = {best['precision']:.3f}, R = {best['recall']:.3f})")
print("=" * 75)
print()

# Plot 6: Threshold sensitivity curve
fig, ax1 = plt.subplots(figsize=(10, 5))

t_vals = [r["threshold"] for r in thresh_results]
p_vals = [r["precision"] for r in thresh_results]
r_vals = [r["recall"] for r in thresh_results]
f_vals = [r["f1"] for r in thresh_results]

ax1.plot(t_vals, p_vals, "o-", color="#42A5F5", lw=2, label="Precision", markersize=7)
ax1.plot(t_vals, r_vals, "s-", color="#66BB6A", lw=2, label="Recall", markersize=7)
ax1.plot(t_vals, f_vals, "D-", color="#FFA726", lw=2, label="F1-Score", markersize=7)
ax1.axvline(ZSCORE_THRESH, color="gray", ls="--", lw=1, alpha=0.7, label=f"Current (z={ZSCORE_THRESH})")
ax1.axvline(best["threshold"], color="red", ls=":", lw=1.5, alpha=0.8, label=f"Best (z={best['threshold']})")

ax1.set_xlabel("Z-Score Threshold", fontsize=12)
ax1.set_ylabel("Score", fontsize=12)
ax1.set_ylim(0.85, 1.02)
ax1.set_title("Z-Score Threshold Sensitivity — Precision / Recall / F1", fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Secondary axis: FP count
ax2 = ax1.twinx()
fp_vals = [r["fp"] for r in thresh_results]
ax2.bar(t_vals, fp_vals, width=0.3, alpha=0.2, color="red", label="False Positives")
ax2.set_ylabel("False Positives", fontsize=11, color="red")
ax2.tick_params(axis="y", labelcolor="red")

fig.tight_layout()
fig.savefig("plot6_zscore_threshold_sensitivity.png", dpi=150)
plt.close(fig)

# ── 8. Visualization ────────────────────────────────────────────────────────

plt.style.use("seaborn-v0_8-darkgrid")
COL_NORMAL   = "#2196F3"
COL_3SIG     = "#FFD54F"   # yellow – 3-sigma (barely visible)
COL_5SIG     = "#FFB74D"   # orange – 5-sigma (clear)
COL_DRIFT    = "#FF8A65"   # deep orange – gradual drift
COL_CPU_EXT  = "#E65100"   # dark orange – extreme CPU
COL_MEM_EXT  = "#C2185B"   # dark pink   – extreme memory
COL_FLAG     = "#D32F2F"

WINDOW_DEFS = [
    ("3-sigma",      sig3_start,    sig3_end,    COL_3SIG),
    ("5-sigma",      sig5_start,    sig5_end,    COL_5SIG),
    ("Drift",        drift_start,   drift_end,   COL_DRIFT),
    ("Extreme CPU",  cpu_ext_start, cpu_ext_end, COL_CPU_EXT),
    ("Extreme Mem",  mem_ext_start, mem_ext_end, COL_MEM_EXT),
]

def shade_windows(ax, label_it=True):
    """Add semi-transparent shading for all 5 anomaly windows."""
    for wname, ws, we, wc in WINDOW_DEFS:
        kw = {"label": wname} if label_it else {}
        ax.axvspan(ws, we - 1, alpha=0.18, color=wc, **kw)

# ────────────────────────────────────────────────────────────────────────────
# Plot 1: Multi-Feature Overview (PSS + utime + nthreads)
# ────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
fig.suptitle("Combined Time-Series -- Multi-Feature Overview (5 Anomaly Windows)", fontsize=14)

ax = axes[0]
ax.plot(combined["t"], combined["pss"], color=COL_NORMAL, lw=1, alpha=0.85)
shade_windows(ax)
ax.set_ylabel("PSS (kB)")
ax.set_title("PSS -- memory footprint")
ax.legend(loc="upper left", fontsize=7, ncol=5)

ax = axes[1]
ax.plot(combined["t"], combined["utime"], color="#7E57C2", lw=1, alpha=0.85)
shade_windows(ax, label_it=False)
ax.set_ylabel("utime (ticks)")
ax.set_title("User CPU Time (cumulative) — CPU usage indicator")

ax = axes[2]
ax.plot(combined["t"], combined["nthreads"], color="#26A69A", lw=1.2, alpha=0.85)
shade_windows(ax, label_it=False)
ax.set_ylabel("nthreads")
ax.set_title("Thread Count — concurrency indicator")
ax.set_xlabel("Sample Index")

fig.tight_layout()
fig.savefig("plot1_multifeature_overview.png", dpi=150)
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Plot 2: Detection Overlay (log-scale PSS with flagged points per method)
# ────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(len(methods), 1, figsize=(16, 12), sharex=True)
fig.suptitle("Anomaly Detection Results — Per Method (log-scale PSS)", fontsize=14, y=1.01)

for ax, (col, label) in zip(axes, methods):
    ax.plot(combined["t"], combined["pss"], color=COL_NORMAL, lw=0.8, alpha=0.5)
    shade_windows(ax)
    mask = combined[col] == -1
    ax.scatter(combined.loc[mask, "t"], combined.loc[mask, "pss"],
               color=COL_FLAG, s=14, zorder=5, label="Flagged anomaly", alpha=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("PSS (kB)")
    r = results[col]
    ax.set_title(f"{label}  —  P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1']:.2f}"
                 f"  (TP={r['tp']}, FP={r['fp']}, FN={r['fn']})")
    ax.legend(loc="upper left", fontsize=7, ncol=5)

axes[-1].set_xlabel("Sample Index")
fig.tight_layout()
fig.savefig("plot2_detection_overlay.png", dpi=150)
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Plot 3: Zoomed-In Anomaly Windows (3x2 grid, one per window)
# ────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("Zoomed-In Anomaly Windows -- PSS with Detection Flags", fontsize=14)

pad = 15
for idx, (wname, ws, we, wc) in enumerate(WINDOW_DEFS):
    ax = axes.flat[idx]
    seg = combined.iloc[max(0, ws - pad):min(len(combined), we + pad)]
    ax.plot(seg["t"], seg["pss"], color=COL_NORMAL, lw=1.2, alpha=0.8)
    ax.axvspan(ws, we - 1, alpha=0.18, color=wc)
    ax.axvline(ws, color=wc, ls="--", lw=1, alpha=0.7)
    ax.axvline(we - 1, color=wc, ls="--", lw=1, alpha=0.7)
    for col_name, lab, mk in [("zscore","Z","o"), ("lof","LOF","s"), ("ocsvm","SVM","D"),
                               ("elliptic","Ell","^"), ("autoenc","AE","v"),
                               ("cusum","CUSUM","P")]:
        m = seg[col_name] == -1
        if m.any():
            ax.scatter(seg.loc[m, "t"], seg.loc[m, "pss"], marker=mk, s=28,
                       zorder=5, alpha=0.7, label=lab)
    ax.set_ylabel("PSS (kB)")
    ax.set_title(f"{wname} (t={ws}-{we-1}, {we-ws} pts)")
    ax.legend(fontsize=6, ncol=6)

# Hide the 6th (empty) subplot
axes.flat[5].set_visible(False)

for ax in axes[-1]:
    if ax.get_visible():
        ax.set_xlabel("Sample Index")
fig.tight_layout()
fig.savefig("plot3_zoomed_windows.png", dpi=150)
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Plot 4: Method Comparison Bar Chart (Precision / Recall / F1)
# ────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
method_labels = [label for _, label in methods]
x = np.arange(len(method_labels))
w = 0.25

prec_vals = [results[col]["precision"] for col, _ in methods]
rec_vals  = [results[col]["recall"]    for col, _ in methods]
f1_vals   = [results[col]["f1"]        for col, _ in methods]

bars1 = ax.bar(x - w, prec_vals, w, label="Precision", color="#42A5F5")
bars2 = ax.bar(x,     rec_vals,  w, label="Recall",    color="#66BB6A")
bars3 = ax.bar(x + w, f1_vals,   w, label="F1-Score",  color="#FFA726")

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(method_labels)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
ax.set_title("Method Comparison — Precision / Recall / F1")
ax.legend()
fig.tight_layout()
fig.savefig("plot4_method_comparison.png", dpi=150)
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Plot 7: Autoencoder Reconstruction Error
# Shows the per-sample MSE across the time series. Normal data has near-zero
# error; anomalies spike dramatically.
# ────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(combined["t"], recon_error, color="#7E57C2", lw=0.8, alpha=0.85)
shade_windows(ax)
ax.axhline(ae_threshold, color="red", ls="--", lw=1.5, alpha=0.8,
           label=f"Threshold ({ae_threshold:.4f})")
ax.set_ylabel("Reconstruction Error (MSE)")
ax.set_xlabel("Sample Index")
ax.set_yscale("log")
ax.set_title("Autoencoder Reconstruction Error — Anomalies Spike Far Above Threshold", fontsize=13)
ax.legend(loc="upper left", fontsize=8, ncol=5)
fig.tight_layout()
fig.savefig("plot7_autoencoder_recon_error.png", dpi=150)
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Plot 8: Elliptic Envelope Decision Scores
# ────────────────────────────────────────────────────────────────────────────

ee_scores = ee.decision_function(all_X_scaled)
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(combined["t"], ee_scores, color="#26A69A", lw=0.8, alpha=0.85)
shade_windows(ax)
ax.axhline(0, color="red", ls="--", lw=1.5, alpha=0.8, label="Decision boundary (0)")
ax.set_ylabel("Decision Score")
ax.set_xlabel("Sample Index")
ax.set_title("Elliptic Envelope Decision Score — Negative = Anomaly", fontsize=13)
ax.legend(loc="lower left", fontsize=8, ncol=5)
fig.tight_layout()
fig.savefig("plot8_elliptic_envelope_scores.png", dpi=150)
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Computational Cost Table
# ────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("              Computational Cost Comparison")
print("=" * 70)
print(f"{'Method':<18} {'Train (ms)':>12} {'Predict (ms)':>14} {'Total (ms)':>12}")
print("-" * 70)
for mname in ["Z-Score", "LOF", "OC-SVM", "Elliptic Env.", "Autoencoder"]:
    tr = timing_results[mname]["train"] * 1000
    pr = timing_results[mname]["predict"] * 1000
    total = tr + pr
    print(f"{mname:<18} {tr:>12.2f} {pr:>14.2f} {total:>12.2f}")
print("=" * 70)
print()

# ────────────────────────────────────────────────────────────────────────────
# Plot 9: ROC Curves
# Uses continuous anomaly scores from methods that support decision_function.
# ────────────────────────────────────────────────────────────────────────────

from sklearn.metrics import roc_curve, roc_auc_score

fig, ax = plt.subplots(figsize=(8, 7))

# Collect anomaly scores (higher = more anomalous for sklearn convention)
roc_methods = {
    "Z-Score": np.max(np.abs(all_X_scaled), axis=1),       # max |z| per row
    "LOF":     -lof.decision_function(all_X_scaled),         # negate: lower = more anomalous
    "OC-SVM":  -ocsvm.decision_function(all_X_scaled),       # negate: lower = more anomalous
    "Elliptic Env.": -ee.decision_function(all_X_scaled),    # negate: lower = more anomalous
    "Autoencoder": recon_error,                               # higher = more anomalous
}

colors = ["#42A5F5", "#66BB6A", "#FFA726", "#26A69A", "#AB47BC"]

for (mname, scores), color in zip(roc_methods.items(), colors):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{mname} (AUC={auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.5)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All 5 Methods", fontsize=14)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
fig.tight_layout()
fig.savefig("plot9_roc_curves.png", dpi=150)
plt.close(fig)

# Print AUC scores
print("=" * 50)
print("          ROC AUC Scores")
print("=" * 50)
for mname, scores in roc_methods.items():
    auc = roc_auc_score(y_true, scores)
    print(f"  {mname:<18}  AUC = {auc:.4f}")
print("=" * 50)
print()

# ────────────────────────────────────────────────────────────────────────────
# Plot 10: Confusion Matrix Grid (5 panels)
# ────────────────────────────────────────────────────────────────────────────

from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Confusion Matrices — All 5 Methods", fontsize=14, y=1.05)

for ax, (col, label) in zip(axes, methods):
    y_pred = (combined[col] == -1).astype(int).values
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"], fontsize=8)
    ax.set_yticklabels(["Normal", "Anomaly"], fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    if ax == axes[0]:
        ax.set_ylabel("Actual", fontsize=9)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

fig.tight_layout()
fig.savefig("plot10_confusion_matrices.png", dpi=150)
plt.close(fig)

print("All plots saved: plot1-plot10.")
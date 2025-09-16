# -*- coding: utf-8 -*-
"""
evaluate_cnn_combos.py
----------------------
Standalone script to:
1) Evaluate ALL combinations of technical indicators for a CNN model (factorial design).
2) Save a CSV of results (both VALIDATION and TEST metrics: RMSE, MAE, Directional Accuracy, PnL Ratio).
3) Produce correlation heatmaps that show how indicator presence relates to metrics and
   how metrics relate to one another.
4) EXTRA: Indicator "effect sizes", pairwise synergy heatmaps, a Top-N combos heatmap,
   and a COMBINATION EFFECTIVENESS heatmap (per-combo performance matrix).

Assumptions:
- Project modules are available locally:
    data_loader.load_stock_data
    indicators.add_indicators
    preprocessing.normalize_features, df_to_windowed_df, windowed_df_to_date_X_y
    model.build_cnn_model
    evaluation.compute_metrics, evaluation.compute_directional_accuracy, evaluation.simulate_pnl
- A YAML config file lives at config/config.yaml with keys:
    tickers, paths.csv_folder, cnn.window_size, cnn.epochs, cnn.batch_size
- Output directory "output" is writable.
"""

import os
import itertools
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional but nice for heatmaps. If not available, the script falls back to matplotlib only.
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

import tensorflow as tf

from data_loader import load_stock_data
from indicators import add_indicators
from preprocessing import normalize_features, df_to_windowed_df, windowed_df_to_date_X_y
from model import build_cnn_model
from evaluation import compute_metrics, compute_directional_accuracy, simulate_pnl

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

# =========================
# Selection policy
#   - "val": avoid test leakage — choose champions by VALIDATION metrics,
#            but report the TEST metrics for the chosen combos.
#   - "test": choose champions directly by TEST metrics (legacy behavior).
# =========================
SELECT_ON = "val"   # "val" (recommended) or "test"


def all_indicator_combinations(indicator_list):
    """Return list of all non-empty combinations of the indicators (factorial design)."""
    combos = []
    for r in range(1, len(indicator_list) + 1):
        combos.extend(itertools.combinations(indicator_list, r))
    return combos


def ensure_output_dir(path="output"):
    os.makedirs(path, exist_ok=True)
    return path


def evaluate_combo(df, features, window_size, epochs, batch_size, ticker="TCKR", outdir="output"):
    """
    Train on given feature set and return BOTH validation and test metrics,
    plus test arrays (y_true_test, y_pred_test) for optional saving.

    Returns dict with keys:
      rmse_val, mae_val, da_val, pnl_val
      rmse_test, mae_test, da_test, pnl_test
      y_true_test, y_pred_test
    """
    df_subset = df[["Target"] + features].copy()

    # Scale and window
    df_scaled, _, scaler_y = normalize_features(df_subset, target_col="Target")
    windowed_df = df_to_windowed_df(df_scaled, window_size, target_col="Target")
    dates, X, y = windowed_df_to_date_X_y(windowed_df, window_size)

    # Split 90/6/4 as in the main pipeline
    n = len(dates)
    if n < 10:
        raise ValueError(f"Too few samples after windowing: n={n}")
    q_90 = int(n * 0.90)
    q_96 = int(n * 0.96)

    X_train, X_val, X_test = X[:q_90], X[q_90:q_96], X[q_96:]
    y_train, y_val, y_test = y[:q_90], y[q_90:q_96], y[q_96:]

    # Model
    model = build_cnn_model(window_size, X.shape[2])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predict (inverse-transform) — VALIDATION
    y_val_pred_scaled = model.predict(X_val, verbose=0).reshape(-1, 1)
    y_val_scaled = y_val.reshape(-1, 1)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled).flatten()
    y_val_orig = scaler_y.inverse_transform(y_val_scaled).flatten()

    # Predict (inverse-transform) — TEST
    y_test_pred_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    y_test_scaled = y_test.reshape(-1, 1)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test_scaled).flatten()

    # Clean invalids (val)
    m_val = np.isfinite(y_val_orig) & np.isfinite(y_val_pred)
    y_val_orig = y_val_orig[m_val]
    y_val_pred = y_val_pred[m_val]

    # Clean invalids (test)
    m_test = np.isfinite(y_test_orig) & np.isfinite(y_test_pred)
    y_test_orig = y_test_orig[m_test]
    y_test_pred = y_test_pred[m_test]

    if len(y_val_orig) == 0 or len(y_test_orig) == 0:
        raise ValueError("No valid samples after cleaning NaN/Inf (val/test).")

    # Metrics — VALIDATION
    rmse_val, mae_val = compute_metrics(y_val_orig, y_val_pred)
    da_val = compute_directional_accuracy(y_val_orig, y_val_pred)
    pnl_val, _ = simulate_pnl(y_val_orig, y_val_pred, initial_cash=80000, ticker=ticker, output_folder=outdir)
    pnl_val_ratio = float(pnl_val.get("P&L Ratio", 0.0))

    # Metrics — TEST
    rmse_test, mae_test = compute_metrics(y_test_orig, y_test_pred)
    da_test = compute_directional_accuracy(y_test_orig, y_test_pred)
    pnl_test, _ = simulate_pnl(y_test_orig, y_test_pred, initial_cash=80000, ticker=ticker, output_folder=outdir)
    pnl_test_ratio = float(pnl_test.get("P&L Ratio", 0.0))

    return {
        # VAL
        "rmse_val": float(rmse_val),
        "mae_val": float(mae_val),
        "da_val": float(da_val),
        "pnl_val": float(pnl_val_ratio),
        # TEST
        "rmse_test": float(rmse_test),
        "mae_test": float(mae_test),
        "da_test": float(da_test),
        "pnl_test": float(pnl_test_ratio),
        # Arrays (test)
        "y_true_test": y_test_orig,
        "y_pred_test": y_test_pred,
    }


def build_design_matrix(combo_strings, indicators):
    """
    Binary design matrix: rows = combos (from strings), columns = indicators (1 if present).
    combo_strings: list of strings like 'RSI, MACD, SMA'
    """
    X = np.zeros((len(combo_strings), len(indicators)), dtype=int)
    for i, s in enumerate(combo_strings):
        feats = [f.strip() for f in s.split(",")] if isinstance(s, str) else []
        for j, ind in enumerate(indicators):
            X[i, j] = 1 if ind in feats else 0
    return pd.DataFrame(X, columns=indicators)


def plot_heatmap(df, title, out_path):
    plt.figure(figsize=(8, 6))
    if _HAS_SNS:
        sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    else:
        im = plt.imshow(df.values, cmap="coolwarm")
        plt.colorbar(im)
        plt.xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
        plt.yticks(range(df.shape[0]), df.index)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                plt.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info("Saved heatmap → %s", out_path)


def plot_heatmap_values(df, title, out_path):
    """Like plot_heatmap but allows NaNs and doesn't force a uniform fmt."""
    plt.figure(figsize=(8, 6))
    if _HAS_SNS:
        sns.heatmap(df, annot=True, cmap="coolwarm", square=True, fmt=".2f")
    else:
        im = plt.imshow(df.values, cmap="coolwarm"); plt.colorbar(im)
        plt.xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
        plt.yticks(range(df.shape[0]), df.index)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                v = df.values[i, j]
                if not (v is None or (isinstance(v, float) and np.isnan(v))):
                    plt.text(j, i, f"{v:.2f}", ha="center", va="center")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info("Saved heatmap → %s", out_path)


def main():
    # --- Reproducibility
    os.environ["PYTHONHASHSEED"] = "42"
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ticker = config["tickers"][0]
    csv_folder = config["paths"]["csv_folder"]
    window_size = config["cnn"]["window_size"]
    epochs = config["cnn"]["epochs"]
    batch_size = config["cnn"]["batch_size"]

    outdir = ensure_output_dir("output")

    # --- Data
    df = load_stock_data(csv_folder, [ticker])[0]
    df = add_indicators(df)

    # Restore DateTimeIndex if needed
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            logging.warning("DataFrame index restored to DateTimeIndex from 'Date' column.")
        else:
            logging.warning("DataFrame lacks a DateTimeIndex; proceeding, but downstream date-based features may fail.")

    # Define candidate indicators; keep only those present in df
    candidate_indicators = ["RSI", "BB_upper", "BB_lower", "Momentum", "MACD", "SMA", "EMA", "RollingVolatility"]
    indicators = [c for c in candidate_indicators if c in df.columns]
    if not indicators:
        raise RuntimeError("No candidate indicators found in dataframe columns.")

    # Build Target (T+1 Close)
    df["Target"] = df["Close"].shift(-1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # --- Factorial design: all non-empty combinations
    combos = all_indicator_combinations(indicators)
    logging.info("Total combinations: %d", len(combos))

    # --- Evaluate all combos
    rows = []

    # Champions (we keep both selection modes internally; we'll honor SELECT_ON at the end)
    best_rmse_test = np.inf
    best_rmse_combo = None
    # Winners for DA/PnL with two selection bases
    best_da_val = -np.inf;  best_da_val_combo = None;  best_da_val_report = {}
    best_da_test = -np.inf; best_da_test_combo = None; best_da_test_report = {}

    best_pnl_val = -np.inf;  best_pnl_val_combo = None;  best_pnl_val_report = {}
    best_pnl_test = -np.inf; best_pnl_test_combo = None; best_pnl_test_report = {}

    for idx, combo in enumerate(combos, start=1):
        try:
            res = evaluate_combo(
                df=df,
                features=list(combo),
                window_size=window_size,
                epochs=epochs,
                batch_size=batch_size,
                ticker=ticker,
                outdir=outdir,
            )

            # Row for CSV (include BOTH VAL and TEST metrics)
            rows.append({
                "Index": idx,
                "Features": ", ".join(combo),
                # VALIDATION
                "RMSE_val": round(res["rmse_val"], 6),
                "MAE_val": round(res["mae_val"], 6),
                "Directional Accuracy_val": round(res["da_val"], 6),
                "PnL Ratio_val": round(res["pnl_val"], 6),
                # TEST
                "RMSE": round(res["rmse_test"], 6),
                "MAE": round(res["mae_test"], 6),
                "Directional Accuracy": round(res["da_test"], 6),
                "PnL Ratio": round(res["pnl_test"], 6),
            })

            # --- Update winners (RMSE by TEST — legacy)
            if res["rmse_test"] < best_rmse_test:
                best_rmse_test = res["rmse_test"]
                best_rmse_combo = combo

            # --- DA winners (VAL selection)
            if res["da_val"] > best_da_val:
                best_da_val = res["da_val"]
                best_da_val_combo = combo
                # store the TEST metrics of the VAL-selected combo for reporting
                best_da_val_report = {
                    "rmse_test": res["rmse_test"],
                    "mae_test": res["mae_test"],
                    "da_test": res["da_test"],
                    "pnl_test": res["pnl_test"],
                }

            # --- DA winners (TEST selection)
            if res["da_test"] > best_da_test:
                best_da_test = res["da_test"]
                best_da_test_combo = combo
                best_da_test_report = {
                    "rmse_test": res["rmse_test"],
                    "mae_test": res["mae_test"],
                    "da_test": res["da_test"],
                    "pnl_test": res["pnl_test"],
                }

            # --- PnL winners (VAL selection)
            if res["pnl_val"] > best_pnl_val:
                best_pnl_val = res["pnl_val"]
                best_pnl_val_combo = combo
                best_pnl_val_report = {
                    "rmse_test": res["rmse_test"],
                    "mae_test": res["mae_test"],
                    "da_test": res["da_test"],
                    "pnl_test": res["pnl_test"],
                }

            # --- PnL winners (TEST selection)
            if res["pnl_test"] > best_pnl_test:
                best_pnl_test = res["pnl_test"]
                best_pnl_test_combo = combo
                best_pnl_test_report = {
                    "rmse_test": res["rmse_test"],
                    "mae_test": res["mae_test"],
                    "da_test": res["da_test"],
                    "pnl_test": res["pnl_test"],
                }

            logging.info(
                "✅ %d/%d %s | VAL: DA %.3f PnL %.3f | TEST: RMSE %.4f MAE %.4f DA %.3f PnL %.3f",
                idx, len(combos), combo,
                res["da_val"], res["pnl_val"],
                res["rmse_test"], res["mae_test"], res["da_test"], res["pnl_test"]
            )

        except Exception as e:
            logging.warning("❌ Failed on combo %s: %s", combo, str(e))

    if not rows:
        raise RuntimeError("No successful combinations evaluated. Check data and pipeline.")

    # =========================
    # Save results (master + sorted views)
    # =========================
    results_df = pd.DataFrame(rows)

    outdir = ensure_output_dir("output")
    csv_master = os.path.join(outdir, f"cnn_combo_results_{ticker}.csv")
    results_df.to_csv(csv_master, index=False)

    # Sorted convenience views
    csv_by_rmse = os.path.join(outdir, f"cnn_combo_results_by_rmse_{ticker}.csv")
    csv_by_da_val = os.path.join(outdir, f"cnn_combo_results_by_daVAL_{ticker}.csv")
    csv_by_da_test = os.path.join(outdir, f"cnn_combo_results_by_daTEST_{ticker}.csv")
    csv_by_pnl_val = os.path.join(outdir, f"cnn_combo_results_by_pnlVAL_{ticker}.csv")
    csv_by_pnl_test = os.path.join(outdir, f"cnn_combo_results_by_pnlTEST_{ticker}.csv")

    results_df.sort_values(by="RMSE", ascending=True).to_csv(csv_by_rmse, index=False)
    results_df.sort_values(by="Directional Accuracy_val", ascending=False).to_csv(csv_by_da_val, index=False)
    results_df.sort_values(by="Directional Accuracy", ascending=False).to_csv(csv_by_da_test, index=False)
    results_df.sort_values(by="PnL Ratio_val", ascending=False).to_csv(csv_by_pnl_val, index=False)
    results_df.sort_values(by="PnL Ratio", ascending=False).to_csv(csv_by_pnl_test, index=False)

    print(f"✅ Saved master: {csv_master}")
    print(f"   Sorted by RMSE (TEST): {csv_by_rmse}")
    print(f"   Sorted by DA (VAL):    {csv_by_da_val}")
    print(f"   Sorted by DA (TEST):   {csv_by_da_test}")
    print(f"   Sorted by PnL (VAL):   {csv_by_pnl_val}")
    print(f"   Sorted by PnL (TEST):  {csv_by_pnl_test}")

    # =========================
    # Heatmaps & analyses (use TEST metrics to keep continuity with previous visuals)
    # =========================
    # Build binary design matrix from the 'Features' strings
    D = build_design_matrix(results_df["Features"].tolist(), indicators)

    # Metrics (TEST) for visuals
    M = results_df[["RMSE", "MAE", "Directional Accuracy", "PnL Ratio"]].reset_index(drop=True)

    # For interpretability, flip RMSE/MAE sign so "higher is better"
    M_adj = M.copy()
    M_adj["-RMSE"] = -M_adj["RMSE"]
    M_adj["-MAE"]  = -M_adj["MAE"]
    M_adj = M_adj[["-RMSE", "-MAE", "Directional Accuracy", "PnL Ratio"]]

    corr_ind_vs_metrics = pd.DataFrame(index=indicators, columns=M_adj.columns, dtype=float)
    for ind in indicators:
        for met in M_adj.columns:
            corr_ind_vs_metrics.loc[ind, met] = np.corrcoef(D[ind].values, M_adj[met].values)[0, 1]

    # Heatmap 1: indicator presence vs metrics
    heatmap1_path = os.path.join(outdir, f"heatmap_indicator_vs_metrics_{ticker}.png")
    plot_heatmap(corr_ind_vs_metrics, "Indicator Presence vs Metrics (corr)", heatmap1_path)

    # Heatmap 2: metric-to-metric correlations across combos
    corr_metrics = M.corr()
    heatmap2_path = os.path.join(outdir, f"heatmap_metric_to_metric_{ticker}.png")
    plot_heatmap(corr_metrics, "Metric-to-Metric Correlations (TEST)", heatmap2_path)

    # =========================
    # EXTRA ANALYSES (EFFECTIVENESS)
    # =========================

    # A) Indicator effect sizes (present − absent) for each metric (with -RMSE/-MAE)
    effects = pd.DataFrame(index=D.columns, columns=M_adj.columns, dtype=float)
    for ind in D.columns:
        present_mask = D[ind] == 1
        absent_mask  = D[ind] == 0
        for met in M_adj.columns:
            effects.loc[ind, met] = M_adj.loc[present_mask, met].mean() - M_adj.loc[absent_mask, met].mean()

    effects_csv = os.path.join(outdir, f"indicator_effect_sizes_{ticker}.csv")
    effects.to_csv(effects_csv)
    effects_png = os.path.join(outdir, f"indicator_effect_sizes_{ticker}.png")
    plot_heatmap(effects, "Indicator Effect Sizes (present − absent)\n(higher = better)", effects_png)

    # B) Pairwise synergy heatmaps for three key metrics
    def pairwise_synergy(metric_name):
        cols = list(D.columns)
        mat = np.full((len(cols), len(cols)), np.nan, dtype=float)
        for i, a in enumerate(cols):
            A = (D[a] == 1)
            if A.sum() < 2:
                continue
            for j, b in enumerate(cols):
                if i == j:
                    continue
                B = (D[b] == 1)
                AB = A & B
                if AB.sum() < 2 or B.sum() < 2:
                    continue
                mAB = M_adj.loc[AB, metric_name].mean()
                mA  = M_adj.loc[A,  metric_name].mean()
                mB  = M_adj.loc[B,  metric_name].mean()
                mat[i, j] = mAB - 0.5 * (mA + mB)
        return pd.DataFrame(mat, index=cols, columns=cols)

    for metric in ["-RMSE", "Directional Accuracy", "PnL Ratio"]:
        S = pairwise_synergy(metric)
        synergy_png = os.path.join(outdir, f"pairwise_synergy_{metric.replace(' ', '_')}_{ticker}.png")
        plot_heatmap_values(S, f"Pairwise Synergy Heatmap — {metric}", synergy_png)

    # C) Heatmap of Top-N combinations across raw TEST metrics
    Z = M_adj.apply(lambda s: (s - s.mean()) / s.std(ddof=0))  # standardized “higher is better”
    composite = Z.sum(axis=1)
    topN = 20 if len(results_df) >= 20 else len(results_df)
    top_idx = np.argsort(-composite.values)[:topN]
    H = results_df.iloc[top_idx][["RMSE", "MAE", "Directional Accuracy", "PnL Ratio"]].copy()
    H.index = results_df.iloc[top_idx]["Features"].tolist()

    top_png = os.path.join(outdir, f"top_combos_heatmap_{ticker}.png")
    plt.figure(figsize=(9.5, 8))
    if _HAS_SNS:
        sns.heatmap(H, annot=True, fmt=".3g", cmap="coolwarm")
    else:
        im = plt.imshow(H.values, cmap="coolwarm"); plt.colorbar(im)
        plt.xticks(range(H.shape[1]), H.columns, rotation=45, ha="right")
        plt.yticks(range(H.shape[0]), H.index)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                plt.text(j, i, f"{H.values[i, j]:.3g}", ha="center", va="center")
    plt.title("Top Combinations — Performance Heatmap (TEST)")
    plt.tight_layout()
    plt.savefig(top_png, dpi=300)
    plt.close()
    logging.info("Saved heatmap → %s", top_png)

    # D) Combination Effectiveness heatmap (ALL combos, per-combo z-scores on TEST metrics)
    E = M_adj.copy()
    E.index = results_df["Features"]
    E = E[["-RMSE", "-MAE", "Directional Accuracy", "PnL Ratio"]]
    Ez = E.apply(lambda s: (s - s.mean()) / s.std(ddof=0))

    combo_scores = pd.DataFrame({
        "Features": results_df["Features"],
        "RMSE": results_df["RMSE"],
        "MAE": results_df["MAE"],
        "Directional Accuracy": results_df["Directional Accuracy"],
        "PnL Ratio": results_df["PnL Ratio"],
        "CompositeScore": Ez.sum(axis=1).values
    }).sort_values("CompositeScore", ascending=False)
    combo_scores_csv = os.path.join(outdir, f"combo_scores_{ticker}.csv")
    combo_scores.to_csv(combo_scores_csv, index=False)

    Ez_sorted = Ez.loc[combo_scores["Features"].values]
    height = max(8, 0.25 * len(Ez_sorted))
    plt.figure(figsize=(10, height))
    if _HAS_SNS:
        sns.heatmap(Ez_sorted, cmap="coolwarm", center=0, cbar=True)
    else:
        im = plt.imshow(Ez_sorted.values, cmap="coolwarm"); plt.colorbar(im)
        plt.xticks(range(Ez_sorted.shape[1]), Ez_sorted.columns, rotation=45, ha="right")
        plt.yticks(range(Ez_sorted.shape[0]), Ez_sorted.index)
    plt.title("Combination Effectiveness Heatmap (TEST z-scores; higher = better)")
    plt.xlabel("Metrics")
    plt.ylabel("Indicator Combination")
    plt.tight_layout()
    combos_heatmap_png = os.path.join(outdir, f"combos_effectiveness_heatmap_{ticker}.png")
    plt.savefig(combos_heatmap_png, dpi=300)
    plt.close()
    logging.info("Saved heatmap → %s", combos_heatmap_png)

    if _HAS_SNS:
        clus_height = max(8, 0.25 * len(Ez))
        cg = sns.clustermap(Ez, cmap="coolwarm", center=0, figsize=(10, clus_height))
        cg.fig.suptitle("Combination Effectiveness Clustered Heatmap (TEST z-scores)", y=1.02)
        combos_clustermap_png = os.path.join(outdir, f"combos_effectiveness_clustermap_{ticker}.png")
        cg.fig.savefig(combos_clustermap_png, dpi=300, bbox_inches="tight")
        plt.close(cg.fig)
        logging.info("Saved heatmap → %s", combos_clustermap_png)
    else:
        combos_clustermap_png = None

    # =========================
    # Decide final DA & PnL winners per SELECT_ON policy
    # =========================
    if SELECT_ON.lower() == "val":
        da_winner_combo = best_da_val_combo
        da_winner_sel_value = best_da_val
        da_winner_report = best_da_val_report

        pnl_winner_combo = best_pnl_val_combo
        pnl_winner_sel_value = best_pnl_val
        pnl_winner_report = best_pnl_val_report
        sel_note = "selected by VALIDATION; reporting TEST metrics below"
    else:
        da_winner_combo = best_da_test_combo
        da_winner_sel_value = best_da_test
        da_winner_report = best_da_test_report

        pnl_winner_combo = best_pnl_test_combo
        pnl_winner_sel_value = best_pnl_test
        pnl_winner_report = best_pnl_test_report
        sel_note = "selected by TEST metrics"

    # =========================
    # Save text summary of winners
    # =========================
    best_txt = os.path.join(outdir, f"best_combos_{ticker}.txt")
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write("=== Best Indicator Combinations ===\n")
        f.write(f"Selection policy: {SELECT_ON.upper()} ({sel_note})\n\n")
        f.write(f"By RMSE (TEST, lowest): {best_rmse_combo} | RMSE_test={best_rmse_test:.6f}\n\n")
        f.write(f"By Directional Accuracy (winner {SELECT_ON.upper()}): {da_winner_combo} | "
                f"DA_{SELECT_ON.lower()}={da_winner_sel_value:.6f} | "
                f"TEST -> RMSE={da_winner_report.get('rmse_test', np.nan):.6f}, "
                f"MAE={da_winner_report.get('mae_test', np.nan):.6f}, "
                f"DA={da_winner_report.get('da_test', np.nan):.6f}, "
                f"PnL={da_winner_report.get('pnl_test', np.nan):.6f}\n\n")
        f.write(f"By PnL Ratio (winner {SELECT_ON.upper()}): {pnl_winner_combo} | "
                f"PnL_{SELECT_ON.lower()}={pnl_winner_sel_value:.6f} | "
                f"TEST -> RMSE={pnl_winner_report.get('rmse_test', np.nan):.6f}, "
                f"MAE={pnl_winner_report.get('mae_test', np.nan):.6f}, "
                f"DA={pnl_winner_report.get('da_test', np.nan):.6f}, "
                f"PnL={pnl_winner_report.get('pnl_test', np.nan):.6f}\n")

    # =========================
    # Console summary
    # =========================
    print(f"🏆 Selection policy: {SELECT_ON.upper()} ({sel_note})")
    print(f"🏆 Best by RMSE (TEST): {best_rmse_combo} (RMSE={best_rmse_test:.6f})")
    print(f"🏆 Best by DA ({SELECT_ON.upper()}): {da_winner_combo} "
          f"(DA_{SELECT_ON.lower()}={da_winner_sel_value:.6f}) | "
          f"TEST: RMSE={da_winner_report.get('rmse_test', np.nan):.6f}, "
          f"MAE={da_winner_report.get('mae_test', np.nan):.6f}, "
          f"DA={da_winner_report.get('da_test', np.nan):.6f}, "
          f"PnL={da_winner_report.get('pnl_test', np.nan):.6f}")
    print(f"🏆 Best by PnL ({SELECT_ON.upper()}): {pnl_winner_combo} "
          f"(PnL_{SELECT_ON.lower()}={pnl_winner_sel_value:.6f}) | "
          f"TEST: RMSE={pnl_winner_report.get('rmse_test', np.nan):.6f}, "
          f"MAE={pnl_winner_report.get('mae_test', np.nan):.6f}, "
          f"DA={pnl_winner_report.get('da_test', np.nan):.6f}, "
          f"PnL={pnl_winner_report.get('pnl_test', np.nan):.6f}")

    print("🖼  Heatmaps saved:")
    print(" -", heatmap1_path)
    print(" -", heatmap2_path)
    print(" -", effects_png)
    print(" -", os.path.join(outdir, f"pairwise_synergy_-RMSE_{ticker}.png"))
    print(" -", os.path.join(outdir, f"pairwise_synergy_Directional_Accuracy_{ticker}.png"))
    print(" -", os.path.join(outdir, f"pairwise_synergy_PnL_Ratio_{ticker}.png"))
    print(" -", top_png)
    print(" -", combos_heatmap_png)
    if _HAS_SNS:
        print(" -", os.path.join(outdir, f"combos_effectiveness_clustermap_{ticker}.png"))
    print("📄 Tables saved:")
    print(" -", csv_master)
    print(" -", csv_by_rmse)
    print(" -", csv_by_da_val)
    print(" -", csv_by_da_test)
    print(" -", csv_by_pnl_val)
    print(" -", csv_by_pnl_test)
    print(" -", combo_scores_csv)


if __name__ == "__main__":
    main()

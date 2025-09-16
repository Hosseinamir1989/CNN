# main_cnn.py
import os
import random
import numpy as np
import tensorflow as tf
import yaml
import logging
from data_loader import load_stock_data
from indicators import add_indicators
from preprocessing import normalize_features, df_to_windowed_df, windowed_df_to_date_X_y
from model import build_cnn_model
from evaluation import (
    plot_predictions, compute_metrics, compute_directional_accuracy,
    evaluate_multi_horizon_accuracy, evaluate_directional_accuracy_by_horizon,
    simulate_pnl
)
from evaluation import plot_test_zoomed

from feature_analysis import feature_drop_analysis
from evaluation_tables import create_evaluation_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")



# CONFIG
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)



tickers = config["tickers"]
dfs = load_stock_data(config["paths"]["csv_folder"], tickers)

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)


# Use the first ticker
df = dfs[0]
df = add_indicators(df)

# >>> Stationarity (features) <<<
stationarity_cfg = config.get("stationarity", {})
if stationarity_cfg.get("enabled", False):
    methods = stationarity_cfg.get("methods", {"_default": stationarity_cfg.get("method", "diff")})
    df = make_stationary(df, methods=methods, target_col="Close")
    logging.info("Applied feature stationarity with methods: %s", methods)


# NORMALIZE
df_scaled, scaler_X, scaler_y = normalize_features(df)
feature_cols = [c for c in df_scaled.columns if c != "Close"]

# WINDOWED DATA
window_size = config["cnn"]["window_size"]
windowed_df = df_to_windowed_df(df_scaled, window_size)
windowed_df = windowed_df.dropna()  # Drop any remaining NaNs
dates, X, y = windowed_df_to_date_X_y(windowed_df, window_size)

# SPLIT
q_90 = int(len(dates) * 0.90)
q_96 = int(len(dates) * 0.96)
X_train, X_val, X_test = X[:q_90], X[q_90:q_96], X[q_96:]
y_train, y_val, y_test = y[:q_90], y[q_90:q_96], y[q_96:]
dates_train, dates_val, dates_test = dates[:q_90], dates[q_90:q_96], dates[q_96:]

# MODEL
learning_rate = config["cnn"]["learning_rate"]
epochs = config["cnn"]["epochs"]
batch_sz = config["cnn"]["batch_size"]
n_features = X.shape[2]
model = build_cnn_model(window_size, n_features)
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_sz, validation_data=(X_val, y_val))

import shap
explainer = shap.GradientExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])
shap.image_plot(shap_values, -X_test[:10])



# SAVE MODEL
model_save_path = f"cnn_model_{tickers[0]}.h5"
model.save(model_save_path)
logging.info(f"Model saved to {model_save_path}")

# PREDICTIONS
train_pred = scaler_y.inverse_transform(model.predict(X_train)).flatten()
val_pred = scaler_y.inverse_transform(model.predict(X_val)).flatten()
test_pred = scaler_y.inverse_transform(model.predict(X_test)).flatten()
y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# PLOT
plot_predictions(
    dates_train, y_train_orig, train_pred,
    dates_val, y_val_orig, val_pred,
    dates_test, y_test_orig, test_pred
)
plot_test_zoomed(dates_test, y_test_orig, test_pred, zoom=100)



# ➕ Delta and Correlation Analysis
delta = test_pred - y_test_orig
correlation = np.corrcoef(test_pred, y_test_orig)[0, 1]
logging.info(f"📊 Correlation (Predicted vs Actual): {correlation:.4f}")

# Save deltas for analysis
import pandas as pd
df_delta = pd.DataFrame({
    "Date": dates_test,
    "Actual": y_test_orig,
    "Predicted": test_pred,
    "Delta": delta
})
df_delta.to_csv(f"output/test_deltas_{tickers[0]}.csv", index=False)
logging.info(f"📈 Deltas saved to 'output/test_deltas_{tickers[0]}.csv'")
# METRICS
import numpy as np

# Remove NaNs from both arrays (keep only indices where both are finite)
mask = np.isfinite(y_test_orig) & np.isfinite(test_pred)
y_test_orig_clean = y_test_orig[mask]
test_pred_clean = test_pred[mask]

if len(y_test_orig_clean) == 0 or len(test_pred_clean) == 0:
    logging.warning("No valid test samples after removing NaNs. Skipping metric calculation.")
    rmse_test, mae_test = np.nan, np.nan
else:
    rmse_test, mae_test = compute_metrics(y_test_orig_clean, test_pred_clean)
    logging.info(f"Test RMSE: {rmse_test:.4f}")
    logging.info(f"Test MAE: {mae_test:.4f}")

# Directional Accuracy
dir_acc_test = compute_directional_accuracy(y_test_orig, test_pred)
with open(f"output/directional_accuracy_{tickers[0]}.txt", "w") as f:
    f.write(f"Directional Accuracy: {dir_acc_test:.4f}")

# Multi-Horizon Evaluation
multi_horizon_results = evaluate_multi_horizon_accuracy(y_test_orig, test_pred)
logging.info("\n📈 Multi-Horizon Evaluation:")
for r in multi_horizon_results:
    logging.info(f"{r['Horizon']}: RMSE = {r['RMSE']:.4f}, MAE = {r['MAE']:.4f}")

# Directional Accuracy by Horizon
dir_acc_horizon_results = evaluate_directional_accuracy_by_horizon(y_test_orig, test_pred)
logging.info("\n🚦 Directional Accuracy by Horizon:")
for r in dir_acc_horizon_results:
    logging.info(f"{r['Horizon']}: Directional Accuracy = {r['Directional Accuracy']:.4f}")




# FEATURE DROP ANALYSIS
results_df = feature_drop_analysis(
    df_scaled,
    feature_cols,
    window_size,
    X,
    y,
    scaler_y,
    dates_test,
    y_test_orig,
    rmse_test,
    mae_test,
    q_90,
    q_96,
    learning_rate,
    epochs,
    batch_sz
)

# CREATE & SAVE TABLES
create_evaluation_tables(
    tickers,
    rmse_test,
    mae_test,
    dir_acc_test,
    results_df,
    output_folder="output"
)

def classify_signals(y_true, y_pred, threshold=0.005):
    """
    Classify predictions into Buy/Hold/Sell based on percentage change threshold.
    Returns a dictionary with counts.
    """
    signals = []
    for i in range(len(y_pred) - 1):
        pct_change = (y_pred[i+1] - y_true[i]) / y_true[i]
        if pct_change > threshold:
            signals.append("Buy")
        elif pct_change < -threshold:
            signals.append("Sell")
        else:
            signals.append("Hold")

    from collections import Counter
    return dict(Counter(signals))

# PnL Simulation (Corrected)
pnl_result, trade_log_df = simulate_pnl(
    y_test_orig,
    test_pred,
    initial_cash=80000,  
    ticker=tickers[0],
    output_folder="output"
)


# 📊 Log PnL results
logging.info("\n💰 PnL Simulation Result:")
for k, v in pnl_result.items():
    logging.info("%s: %s", k, v)


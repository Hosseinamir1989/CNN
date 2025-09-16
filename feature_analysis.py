import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model import build_cnn_model
from preprocessing import df_to_windowed_df, windowed_df_to_date_X_y
import logging

def feature_drop_analysis(
    df_scaled,
    feature_cols,
    window_size,
    X,
    y,
    scaler_y,
    test_dates,
    y_test_orig,
    base_rmse,
    base_mae,
    q_90,
    q_96,
    learning_rate,
    epochs,
    batch_sz
):
    results = []
    for feat in feature_cols:
        drop_cols = [col for col in feature_cols if feat in col]
        df_dropped = df_scaled.drop(columns=drop_cols)

        # Use imported functions from preprocessing.py
        windowed_df_dropped = df_to_windowed_df(df_dropped, window_size)
        _, X_dropped, y_dropped = windowed_df_to_date_X_y(windowed_df_dropped, window_size)

        X_train_dropped, X_test_dropped = X_dropped[:q_90], X_dropped[q_96:]
        y_train_dropped, y_test_dropped = y_dropped[:q_90], y_dropped[q_96:]

        model_dropped = build_cnn_model(window_size, X_train_dropped.shape[2])
        model_dropped.fit(
            X_train_dropped,
            y_train_dropped,
            epochs=epochs,
            batch_size=batch_sz,
            validation_split=0.1,
            verbose=0
        )

        y_pred_dropped = scaler_y.inverse_transform(model_dropped.predict(X_test_dropped)).flatten()
        y_test_dropped_orig = scaler_y.inverse_transform(y_test_dropped.reshape(-1,1)).flatten()

        # Filter NaNs
        mask = np.isfinite(y_test_dropped_orig) & np.isfinite(y_pred_dropped)
        y_test_dropped_orig_clean = y_test_dropped_orig[mask]
        y_pred_dropped_clean = y_pred_dropped[mask]

        if len(y_test_dropped_orig_clean) == 0 or len(y_pred_dropped_clean) == 0:
            logging.warning("No valid samples after removing NaNs. Skipping this feature.")
            continue

        # Compute metrics
        rmse_dropped = np.sqrt(mean_squared_error(y_test_dropped_orig_clean, y_pred_dropped_clean))
        mae_dropped = mean_absolute_error(y_test_dropped_orig_clean, y_pred_dropped_clean)

        results.append({
            'Dropped Feature': feat,
            'RMSE': rmse_dropped,
            'MAE': mae_dropped,
            'RMSE Increase': rmse_dropped - base_rmse,
            'MAE Increase': mae_dropped - base_mae
        })

    return pd.DataFrame(results).sort_values(by='RMSE Increase', ascending=False)

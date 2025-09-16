import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import os
def plot_predictions(dates_train, y_train, train_pred,
                     dates_val, y_val, val_pred,
                     dates_test, y_test, test_pred):
    plt.figure(figsize=(14,6))
    plt.plot(dates_train, y_train, label='Train Actual')
    plt.plot(dates_train, train_pred, label='Train Predicted')
    plt.plot(dates_val, y_val, label='Validation Actual')
    plt.plot(dates_val, val_pred, label='Validation Predicted')
    plt.plot(dates_test, y_test, label='Test Actual')
    plt.plot(dates_test, test_pred, label='Test Predicted')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('CNN Predictions')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_test_zoomed(dates_test, y_test, test_pred, zoom=100):
    """
    Plot only the last 'zoom' values of the test set for detailed inspection.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(dates_test[-zoom:], y_test[-zoom:], label="Actual", linewidth=2)
    plt.plot(dates_test[-zoom:], test_pred[-zoom:], label="Predicted", linestyle="--")
    plt.title(f"Zoomed-in Test Set (last {zoom} samples)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    logging.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae

def compute_directional_accuracy(y_true, y_pred):
    """
    Computes the directional accuracy of predictions: 
    how often the model predicts the correct direction of change.
    """
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    correct_direction = np.sign(y_true_diff) == np.sign(y_pred_diff)
    dir_acc = np.mean(correct_direction)
    logging.info(f"Directional Accuracy: {dir_acc:.4f}")
    return dir_acc


def evaluate_multi_horizon_accuracy(y_true, y_pred, horizon_days=[1, 2, 3]):
    """
    Evaluate RMSE and MAE at multiple horizons (T+1, T+2, T+3).
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    results = []

    for horizon in horizon_days:
        if len(y_true) <= horizon or len(y_pred) <= horizon:
            continue

        shifted_true = y_true[horizon:]
        valid_pred = y_pred[:-horizon]

        # Remove NaNs for this horizon
        mask = np.isfinite(shifted_true) & np.isfinite(valid_pred)
        shifted_true_clean = shifted_true[mask]
        valid_pred_clean = valid_pred[mask]

        if len(shifted_true_clean) == 0 or len(valid_pred_clean) == 0:
            continue

        rmse = np.sqrt(mean_squared_error(shifted_true_clean, valid_pred_clean))
        mae = mean_absolute_error(shifted_true_clean, valid_pred_clean)

        results.append({
            "Horizon": f"T+{horizon}",
            "RMSE": rmse,
            "MAE": mae
        })

    return results


def evaluate_directional_accuracy_by_horizon(y_true, y_pred, horizon_days=[1, 2, 3]):
    """s
    Evaluate directional accuracy for multiple horizons (T+1, T+2, T+3).
    """
    results = []

    for horizon in horizon_days:
        if len(y_true) <= horizon or len(y_pred) <= horizon:
            continue

        shifted_true = y_true[horizon:]
        valid_pred = y_pred[:-horizon]

        # True direction: compare horizon-day forward difference with today
        true_direction = np.sign(shifted_true - y_true[:-horizon])
        pred_direction = np.sign(valid_pred - y_true[:-horizon])

        dir_acc = np.mean(true_direction == pred_direction)
        results.append({
            "Horizon": f"T+{horizon}",
            "Directional Accuracy": dir_acc
        })

    return results

import pandas as pd

# P&L = (Total Gain / Winning Trades) / (Total Loss / Losing Trades)

def simulate_pnl(y_true, y_pred, initial_cash=80000, ticker="Unknown", output_folder="output"):
    """
    Simulates P&L using the formula from the research paper:
    P&L = (Total Gain / #Winning Trades) / (Total Loss / #Losing Trades)

    Assumptions:
    - Buy when prediction for next day > today's price.
    - Sell when prediction for next day < today's price.
    - One position at a time (long only).
    - Trade executes on the next day at close price.
    """

    cash = initial_cash
    shares = 0
    position = None
    entry_price = 0
    trade_log = []

    total_gain = 0
    total_loss = 0
    winning_trades = 0
    losing_trades = 0

    for i in range(len(y_true) - 1):
        today_price = y_true[i]
        tomorrow_price = y_true[i + 1]
        predicted_price = y_pred[i + 1]

        if predicted_price > today_price and shares == 0:
            shares = int(cash // tomorrow_price)
            entry_price = tomorrow_price
            cash -= shares * tomorrow_price
            trade_log.append({'Day': i+1, 'Action': 'Buy', 'Price': tomorrow_price, 'Shares': shares})

        elif predicted_price < today_price and shares > 0:
            exit_price = tomorrow_price
            pnl = (exit_price - entry_price) * shares
            if pnl > 0:
                total_gain += pnl
                winning_trades += 1
            else:
                total_loss += abs(pnl)
                losing_trades += 1
            cash += shares * exit_price
            trade_log.append({'Day': i+1, 'Action': 'Sell', 'Price': exit_price, 'Shares': shares, 'PnL': pnl})
            shares = 0

    if shares > 0:
        exit_price = y_true[-1]
        pnl = (exit_price - entry_price) * shares
        if pnl > 0:
            total_gain += pnl
            winning_trades += 1
        else:
            total_loss += abs(pnl)
            losing_trades += 1
        cash += shares * exit_price
        trade_log.append({'Day': len(y_true)-1, 'Action': 'Sell (Final)', 'Price': exit_price, 'Shares': shares, 'PnL': pnl})
        shares = 0

    # P&L formula from the paper
    avg_gain = total_gain / winning_trades if winning_trades > 0 else 0
    avg_loss = total_loss / losing_trades if losing_trades > 0 else 1  # avoid div by zero
    pnl_ratio = avg_gain / avg_loss if avg_loss != 0 else 0

    # Final stats
    final_value = cash
    result = {
        "Ticker": ticker,
        "Final Cash": round(final_value, 2),
        "Total Gain": round(total_gain, 2),
        "Total Loss": round(total_loss, 2),
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "P&L Ratio": round(pnl_ratio, 2)
    }

    trade_log_df = pd.DataFrame(trade_log)
    os.makedirs(output_folder, exist_ok=True)
    trade_log_df.to_csv(os.path.join(output_folder, f"pnl_research_style_{ticker}.csv"), index=False)

    return result, trade_log_df
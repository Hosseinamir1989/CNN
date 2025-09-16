import pandas as pd
from tabulate import tabulate
import os

def create_evaluation_tables(
    ticker,
    rmse_test,
    mae_test,
    dir_acc_test,
    results_df,
    output_folder="output"
):
    """
    Creates evaluation tables for RMSE, MAE, Directional Accuracy, and feature drop analysis.
    Saves the tables as CSVs and prints them nicely to the console.
    Each table is tagged with the ticker.
    """

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # 1️⃣ Summary Metrics Table
    summary_data = {
        "Ticker": [ticker, ticker, ticker],
        "Metric": ["Test RMSE", "Test MAE", "Test Directional Accuracy"],
        "Value": [f"{rmse_test:.4f}", f"{mae_test:.4f}", f"{dir_acc_test:.4f}"]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_folder, f"summary_metrics_{ticker}.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # Print summary table
    print(f"\n🔎 Summary Metrics Table for {ticker}:")
    print(tabulate(summary_df, headers='keys', tablefmt='pretty', showindex=False))

    # 2️⃣ Feature Drop Analysis Table
    results_df.insert(0, "Ticker", [ticker] * len(results_df))
    results_csv_path = os.path.join(output_folder, f"feature_drop_analysis_{ticker}.csv")
    results_df.to_csv(results_csv_path, index=False)

    print(f"\n📊 Feature Drop Analysis Table for {ticker}:")
    print(tabulate(results_df, headers='keys', tablefmt='pretty', showindex=False))

    # 3️⃣ Best Indicator
    best_row = results_df.loc[results_df['RMSE'].idxmin()]
    best_indicator = best_row['Dropped Feature']

    best_indicator_df = pd.DataFrame({
        "Ticker": [ticker],
        "Best Indicator Dropped": [best_indicator]
    })
    best_indicator_csv_path = os.path.join(output_folder, f"best_indicator_{ticker}.csv")
    best_indicator_df.to_csv(best_indicator_csv_path, index=False)

    print(f"\n🎯 Best Indicator Dropped for {ticker}: {best_indicator}")

    # Done
    print(f"\n✅ All evaluation tables saved in '{output_folder}' directory for {ticker}.")

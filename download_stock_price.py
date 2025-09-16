import yfinance as yf
import pandas as pd

# Top 5 DAX companies by market cap (as of recent years)
tickers = ['SAP.DE', 'SIE.DE', 'DTE.DE', 'VOW3.DE', 'BAS.DE']  # SAP, Siemens, Deutsche Telekom, Volkswagen, BASF

start_date = "2020-01-01"
end_date = "2025-01-01"

# Download data
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)

# Save each ticker to a separate CSV
for ticker in tickers:
    df = data[ticker].copy()
    df.dropna(inplace=True)
    df.to_csv(f"{ticker.replace('.', '_')}_2020_2025.csv")
    print(f"✅ Saved {ticker} data to CSV.")
d
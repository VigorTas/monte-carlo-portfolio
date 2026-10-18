"""Refresh data/sp500_monthly.csv from Yahoo Finance (needs `pip install yfinance pandas`)."""
import pandas as pd
import yfinance as yf

df = yf.download("^GSPC", start="1950-01-01", interval="1mo", auto_adjust=True, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
s = df["Close"].dropna().iloc[:-1]  # drop the unfinished current month
s.index = s.index.strftime("%Y-%m")
s.round(2).to_csv("data/sp500_monthly.csv", header=["close"], index_label="month")
print(len(s), "months:", s.index[0], "to", s.index[-1])

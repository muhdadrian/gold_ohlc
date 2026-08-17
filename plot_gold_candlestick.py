"""
Plot Gold Price Candlestick Chart (MYR/gram)
==============================================
Reads the CSV built by gold_myr_tracker.py and renders an interactive
candlestick chart with Plotly.

SETUP
-----
pip install pandas plotly

Run:
    python plot_gold_candlestick.py
"""

import pandas as pd
import plotly.graph_objects as go

CSV_PATH = "gold_myr_ohlc.csv"


def main():
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    df = df.sort_values("Date")

    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Gold (MYR/g)",
        increasing_line_color="green",
        decreasing_line_color="red",
    )])

    fig.update_layout(
        title="Gold Price — Daily OHLC (MYR per gram)",
        xaxis_title="Date",
        yaxis_title="Price (RM/gram)",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )

    fig.show()


if __name__ == "__main__":
    main()

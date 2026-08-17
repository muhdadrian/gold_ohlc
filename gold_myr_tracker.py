"""
Gold Price Tracker (MYR per gram) — Personal Use
====================================================
Polls the Metals.Dev API during Bursa's trading session (8:30am - 11:50pm MYT)
and builds a daily OHLC row:
    Open  = first price observed after market open
    High  = highest price observed during the session
    Low   = lowest price observed during the session
    Close = last price observed before/at market close

Appends one row per day to a CSV file that you can then load into
pandas/plotly for your candlestick chart.

SETUP
-----
1. Get a free API key (no credit card needed): https://metals.dev
   -> Sign up -> Dashboard -> copy your API Key
2. pip install -r requirements.txt
3. Set your API key as an environment variable (don't hardcode it):
       export METALS_DEV_API_KEY="your_key_here"      # Linux/macOS
       set METALS_DEV_API_KEY=your_key_here            # Windows cmd
4a. Continuous mode (leave running all day):
       python gold_myr_tracker.py
4b. Cron / Task Scheduler mode (recommended to stay within the free
    100 requests/month quota — see note at the bottom):
       python gold_myr_tracker.py --once
    Schedule this to run a few fixed times a day, e.g. 08:30, 14:00,
    20:00, 23:50 MYT. Running state (today's Open/High/Low so far) is
    saved to gold_myr_day_state.json between runs and finalized into
    gold_myr_ohlc.csv once a poll lands at/after market close.
"""

import argparse
import requests
import schedule
import time
import csv
import os
from datetime import datetime, time as dtime

# ------------------------- CONFIG -------------------------
# Set your API key as an environment variable instead of hardcoding it:
#   Linux/macOS:  export METALS_DEV_API_KEY="your_key_here"
#   Windows (cmd): set METALS_DEV_API_KEY=your_key_here
#   Windows (PowerShell): $env:METALS_DEV_API_KEY="your_key_here"
API_KEY = os.environ.get("METALS_DEV_API_KEY")
GRAMS_PER_TROY_OZ = 31.1035

MARKET_OPEN = dtime(8, 30)             # Bursa Gold Dinar session open
MARKET_CLOSE = dtime(23, 50)           # Bursa Gold Dinar official session close
                                        # (this is the trigger used to finalize
                                        # and write the day's CSV row)
POLL_WINDOW_END = dtime(23, 59, 59)    # accept polls up to just before midnight,
                                        # so a poll that fires a few seconds after
                                        # the exact 23:50:00 instant (e.g. via cron)
                                        # is still accepted and still finalizes the
                                        # day, instead of being skipped entirely.

POLL_INTERVAL_MINUTES = 30             # how often to check price during the day
                                        # (30 min x ~15.5 hr session = ~31 calls/day
                                        #  well within the 100/month free quota if
                                        #  you don't poll every single day —
                                        #  see note at bottom on staying in budget)

CSV_PATH = "gold_myr_ohlc.csv"
STATE_PATH = "gold_myr_day_state.json"   # persists today's running O/H/L/C across runs
# -------------------------------------------------------------

import json

_DEFAULT_STATE = {
    "date": None,
    "open": None,
    "high": None,
    "low": None,
    "close": None,
}


def load_day_state():
    """Load today's running state from disk (needed when running via cron,
    since each --once invocation is a fresh process)."""
    if os.path.isfile(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULT_STATE)


def save_day_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


_day_state = load_day_state()


def fetch_gold_price_myr_per_oz(max_retries=4, retry_delay_seconds=10):
    """Fetch current gold spot price in MYR per troy ounce.

    Retries a few times with a short delay in between. This matters most
    for the market-close poll (run via cron at 23:50): if the Mac just
    woke from a display-sleep state, Wi-Fi can take a few seconds to
    reconnect, causing a DNS/connection failure on the very first
    attempt. Without a retry, that would cost the entire day's row.
    """
    url = "https://api.metals.dev/v1/metal/spot"
    params = {
        "api_key": API_KEY,
        "metal": "gold",
        "currency": "MYR",
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "success":
                raise RuntimeError(f"API error: {data.get('error_message', 'unknown error')}")

            return data["rate"]["price"]  # MYR per troy oz
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                print(f"  Attempt {attempt}/{max_retries} failed ({e}); "
                      f"retrying in {retry_delay_seconds}s...")
                time.sleep(retry_delay_seconds)

    # All retries exhausted — raise the last error so poll_once() can log it
    raise last_error


def myr_per_gram(price_per_oz):
    return price_per_oz / GRAMS_PER_TROY_OZ


def is_within_market_hours(now):
    t = now.time()
    return MARKET_OPEN <= t <= POLL_WINDOW_END


def write_csv_row(date_str, o, h, l, c):
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Open", "High", "Low", "Close"])
        writer.writerow([date_str, f"{o:.2f}", f"{h:.2f}", f"{l:.2f}", f"{c:.2f}"])
    print(f"[{date_str}] Saved OHLC -> O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f}")


def poll_once():
    """Fetch one price point and update today's running Open/High/Low/Close."""
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    if not is_within_market_hours(now):
        print(f"[{now.strftime('%H:%M')}] Outside market hours — skipping poll.")
        return

    try:
        price_oz = fetch_gold_price_myr_per_oz()
        price_g = myr_per_gram(price_oz)
    except Exception as e:
        print(f"[{now.strftime('%H:%M')}] Fetch failed: {e}")
        return

    # New trading day -> reset state
    if _day_state["date"] != today_str:
        _day_state["date"] = today_str
        _day_state["open"] = price_g
        _day_state["high"] = price_g
        _day_state["low"] = price_g
        _day_state["close"] = price_g
        print(f"[{now.strftime('%H:%M')}] New day started. Open = RM{price_g:.2f}/g")
    else:
        _day_state["high"] = max(_day_state["high"], price_g)
        _day_state["low"] = min(_day_state["low"], price_g)
        _day_state["close"] = price_g
        print(f"[{now.strftime('%H:%M')}] Price = RM{price_g:.2f}/g "
              f"(H:{_day_state['high']:.2f} L:{_day_state['low']:.2f})")

    # Persist state so it survives across separate cron process runs
    save_day_state(_day_state)

    # If this poll is at/after market close, finalize and write the row
    if now.time() >= MARKET_CLOSE:
        write_csv_row(
            _day_state["date"],
            _day_state["open"],
            _day_state["high"],
            _day_state["low"],
            _day_state["close"],
        )
        # Reset so tomorrow starts fresh
        _day_state.update(_DEFAULT_STATE)
        save_day_state(_day_state)


def main():
    if not API_KEY:
        print("⚠️  METALS_DEV_API_KEY environment variable is not set.")
        print("    Set it first, e.g.:")
        print('    export METALS_DEV_API_KEY="your_key_here"   # Linux/macOS')
        print('    set METALS_DEV_API_KEY=your_key_here        # Windows cmd')
        return

    parser = argparse.ArgumentParser(description="Gold MYR/gram OHLC tracker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll and exit (for use with cron/Task Scheduler) "
             "instead of looping continuously all day.",
    )
    args = parser.parse_args()

    if args.once:
        # One-shot mode: call this a few fixed times a day via cron, e.g.
        # 08:30, 14:00, 20:00, 23:50 -> ~4 calls/day -> ~120/month.
        # Open/Close only need one call each; call it more often only if
        # you also want decent High/Low sampling.
        poll_once()
        return

    print(f"Gold MYR/gram tracker started. Polling every {POLL_INTERVAL_MINUTES} min "
          f"between {MARKET_OPEN} and {MARKET_CLOSE}.")

    schedule.every(POLL_INTERVAL_MINUTES).minutes.do(poll_once)

    # Run once immediately on startup too
    poll_once()

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------
# NOTE ON FREE TIER BUDGET (100 requests/month on Metals.Dev free plan)
# -------------------------------------------------------------------
# Polling every 30 min for a ~15.5 hr session = ~31 calls/day, which
# would burn through 100 calls in about 3 days. For sustainable personal
# use, either:
#   (a) Increase POLL_INTERVAL_MINUTES to something larger, e.g. 240
#       (4 hours) -> ~4-5 calls/day -> ~120-150/month (still tight), or
#   (b) Only run this script on days you actually want a candle
#       (e.g. via cron at 08:30, 14:00, 20:00, 23:50 = 4 calls/day
#       -> ~120/month), or
#   (c) Upgrade to Metals.Dev's cheapest paid tier if you want denser
#       intraday sampling (their paid plans start very cheap for
#       personal-scale usage).
# The Open and Close only need ONE call each per day; it's the
# High/Low sampling in between that eats your quota fastest.

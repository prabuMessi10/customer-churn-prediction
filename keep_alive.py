"""Keep the deployed Streamlit app awake.

Free-tier Streamlit Community Cloud apps fall asleep after a period of
inactivity. This script pings the live app URL every 3 minutes so the
app stays warm.

Run it in the background on any always-on machine:

    python keep_alive.py

It prints one line per ping. Stop with Ctrl+C.
"""
import time
import urllib.error
import urllib.request
from datetime import datetime

APP_URL = "https://customer-churn-prediction-on.streamlit.app"
INTERVAL_SECONDS = 180  # every 3 minutes


def ping() -> None:
    req = urllib.request.Request(APP_URL, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        # 3xx means the app responded (waking up / handing out a session).
        status = f"HTTP {exc.code} (redirect -> waking)"
    except Exception as exc:  # noqa: BLE001 - report and keep going
        status = f"ERROR: {exc}"
    print(f"{datetime.now().isoformat(timespec='seconds')}  {status}", flush=True)


def main() -> None:
    print(
        f"Keep-alive started. Pinging {APP_URL} "
        f"every {INTERVAL_SECONDS} seconds."
    )
    while True:
        ping()
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
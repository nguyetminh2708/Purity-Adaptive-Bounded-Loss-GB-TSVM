"""Save a run's results to a timestamped CSV and append a line to results/RUNS.md."""
from __future__ import annotations
from datetime import datetime
import pandas as pd
from config import RESULTS


def save_run(df: pd.DataFrame, tag: str, params: dict) -> str:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    csv = RESULTS / f"{ts}_{tag}.csv"
    df.to_csv(csv, index=False)
    log = RESULTS / "RUNS.md"
    fresh = not log.exists()
    with open(log, "a", encoding="utf-8") as f:
        if fresh:
            f.write("# Run log\n\nMỗi dòng = một lần chạy: giờ, file kết quả, script, tham số.\n\n")
            f.write("| time | file | script | params |\n|---|---|---|---|\n")
        p = ", ".join(f"{k}={v}" for k, v in params.items())
        f.write(f"| {ts} | `{csv.name}` | {tag} | {p} |\n")
    return csv.name

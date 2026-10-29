"""Per-step throughput and latency from a Locust --csv run (the *_stats_history.csv file).

python scripts/summarize_locust.py load_tests/results/run_stats_history.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict


def mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r[key] not in ("", "N/A")]
    return sum(vals) / len(vals) if vals else float("nan")


def main(path: str) -> None:
    by_users: dict[int, list[dict[str, str]]] = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["Name"] == "Aggregated" and r["User Count"] != "0":
                by_users[int(r["User Count"])].append(r)
    print("| users | req/s | p50 ms | p95 ms | failures/s |")
    print("|---|---|---|---|---|")
    for users, rows in sorted(by_users.items()):
        rows = rows[len(rows) // 3 :] or rows  # drop the ramp-in third of each step
        cells = [mean(rows, "Requests/s"), mean(rows, "50%"), mean(rows, "95%")]
        print(
            f"| {users} | {cells[0]:.2f} | {cells[1]:.0f} | {cells[2]:.0f} "
            f"| {mean(rows, 'Failures/s'):.2f} |"
        )


if __name__ == "__main__":
    main(sys.argv[1])

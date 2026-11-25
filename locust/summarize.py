"""Turn Locust CSVs into a Markdown table and a chart for the README."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = Path(__file__).parent / "results"
rows = []
for n in (1, 2, 4):
    f = RES / f"{n}_containers_stats.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f)
    r = df[df["Name"] == "/predict"].iloc[0]
    rows.append({"containers": n, "requests": int(r["Request Count"]), "failures": int(r["Failure Count"]),
                 "avg_ms": round(r["Average Response Time"]), "p50_ms": round(r["50%"]),
                 "p95_ms": round(r["95%"]), "p99_ms": round(r["99%"]), "max_ms": round(r["Max Response Time"]),
                 "rps": round(r["Requests/s"], 1)})
table = pd.DataFrame(rows)
md = table.to_markdown(index=False)
(RES / "summary.md").write_text(md + "\n")
print(md)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
x = table["containers"].astype(str)
ax[0].plot(x, table["p50_ms"], marker="o", lw=2, color="#2a78d6", label="median")
ax[0].plot(x, table["p95_ms"], marker="o", lw=2, color="#eb6834", label="95th percentile")
ax[0].set(title="/predict latency", xlabel="API containers", ylabel="ms")
ax[0].legend(frameon=False)
ax[1].bar(x, table["rps"], color="#2a78d6", width=0.5)
ax[1].set(title="Throughput", xlabel="API containers", ylabel="requests / s")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color="#ecebe7")
    a.set_axisbelow(True)
fig.tight_layout()
fig.savefig(RES / "scaling.png", dpi=150)

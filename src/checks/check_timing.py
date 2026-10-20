"""Two checks on one Case Input file.

    python check_timing.py path/to/CaseInput.json [FACILITY]

1. pdiSentOut vs energy.sentOut for one facility (first 12 cases).
2. Forecast vs actual over the whole file, with two alignments:
     same case : forecast(t) vs actual(t)
     next case : forecast(t) vs actual(t + 5 min)
   actual = demand + withdraw  (net injection, comparable to the forecast)
   The alignment with the smaller error / bias is the right one.
"""
import sys

import ijson

path = sys.argv[1]
fac = sys.argv[2] if len(sys.argv) > 2 else "COLLIE_ESR5"

rows = []
print(f"--- {fac}: energy.sentOut vs pdiSentOut ---")
print(f"{'interval':>26} {'sentOut':>10} {'pdiSentOut':>11} {'prev sentOut':>13}")
prev = None
with open(path, "rb") as f:
    for i, case in enumerate(ijson.items(f, "data.caseData.item", use_float=True)):
        tags = {s["tag"]: s["value"] for s in case["scada"]}
        so = tags.get(f"{fac}.energy.sentOut")
        pdi = tags.get(f"{fac}.pdiSentOut")
        if i < 12:
            print(f"{case['dispatchInterval']:>26} {so!s:>10} {pdi!s:>11} {prev!s:>13}")
        prev = so

        vals = [float(v) for t, v in tags.items() if t.endswith(".energy.sentOut") and v not in (None, "")]
        withdraw = sum(v for v in vals if v < 0)
        actual = float(tags["dispatchCondition.demand"]) + withdraw
        forecast = case["constraintLookUp"]["forecast.energy-forecastOperationalDemand-expected"]
        rows.append((forecast, actual))

def summary(pairs):
    err = [fc - ac for fc, ac in pairs]
    return sum(abs(e) for e in err) / len(err), sum(err) / len(err)

same = rows
nxt = [(rows[k][0], rows[k + 1][1]) for k in range(len(rows) - 1)]
print(f"\n--- forecast vs actual over {len(rows)} cases (MW) ---")
for name, pairs in [("same case", same), ("next case", nxt)]:
    mae, bias = summary(pairs)
    print(f"{name:>10}: mean abs error {mae:6.1f}   bias (forecast - actual) {bias:+6.1f}")
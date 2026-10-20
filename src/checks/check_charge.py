"""What are chargeLevel, chargeLevelV2 and dischargeLevelV2?

    python check_charge.py path/to/CaseInput.json [path/to/DispatchSolution.json]

Part 1: first 6 cases for one battery, to eyeball.
Part 2: per-battery summary over the whole file:
  - signs: is chargeLevelV2 always <= 0 and dischargeLevelV2 always >= 0?
  - total: is dischargeV2 - chargeV2 constant? (= usable MWh, like the Dispatch Solution)
  - v1 vs v2: how does chargeLevel relate to dischargeLevelV2?
  - energy check: does dischargeV2 drop by (average sentOut) x 5/60 between cases?
Part 3 (optional): if a Dispatch Solution from the same day is given, compare the
  V2 tags with projectedChargeLevel / projectedDischargeLevel at its primary interval.
"""
import json
import statistics as st
import sys

import ijson

ci_path = sys.argv[1]
ds_path = sys.argv[2] if len(sys.argv) > 2 else None


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---- read the four tags for every battery, every case
series = {}          # battery -> list of dicts, one per case
with open(ci_path, "rb") as f:
    for case in ijson.items(f, "data.caseData.item", use_float=True):
        tags = {s["tag"]: s["value"] for s in case["scada"]}
        bats = {t.rsplit(".", 1)[0] for t in tags if t.endswith(".chargeLevelV2")}
        for b in bats:
            series.setdefault(b, []).append({
                "t": case["dispatchInterval"],
                "v1": num(tags.get(f"{b}.chargeLevel")),
                "c2": num(tags.get(f"{b}.chargeLevelV2")),
                "d2": num(tags.get(f"{b}.dischargeLevelV2")),
                "mw": num(tags.get(f"{b}.energy.sentOut")),
            })

# ---- part 1
first = sorted(series)[0]
print(f"--- {first}: first 6 cases ---")
print(f"{'interval':>26} {'chargeLevel':>12} {'chargeV2':>10} {'dischargeV2':>12} {'d2 - c2':>9} {'sentOut':>9}")
for r in series[first][:6]:
    tot = r["d2"] - r["c2"] if None not in (r["d2"], r["c2"]) else None
    print(f"{r['t']:>26} {r['v1']!s:>12} {r['c2']!s:>10} {r['d2']!s:>12} {tot!s:>9} {r['mw']!s:>9}")

# ---- part 2
print("\n--- per battery, whole file ---")
print(f"{'battery':>16} {'n':>4} {'c2<=0':>6} {'d2>=0':>6} {'d2-c2 mean':>11} {'sd':>6} "
      f"{'v1-d2 mean':>11} {'sd':>6} {'energy err':>11}")
for b in sorted(series):
    rows = [r for r in series[b] if None not in (r["v1"], r["c2"], r["d2"])]
    if not rows:
        print(f"{b:>16}  no complete rows")
        continue
    c2_ok = sum(r["c2"] <= 0 for r in rows) / len(rows)
    d2_ok = sum(r["d2"] >= 0 for r in rows) / len(rows)
    tot = [r["d2"] - r["c2"] for r in rows]
    v1d2 = [r["v1"] - r["d2"] for r in rows]
    # energy check: change in dischargeV2 vs -(average MW over the 5 min) * 5/60
    errs = []
    for a, b2 in zip(rows, rows[1:]):
        if a["mw"] is not None and b2["mw"] is not None:
            expected = -(a["mw"] + b2["mw"]) / 2 * 5 / 60
            errs.append(abs((b2["d2"] - a["d2"]) - expected))
    e = st.mean(errs) if errs else float("nan")
    sd = lambda x: st.pstdev(x) if len(x) > 1 else 0.0
    print(f"{b:>16} {len(rows):>4} {c2_ok:>6.0%} {d2_ok:>6.0%} {st.mean(tot):>11.1f} {sd(tot):>6.1f} "
          f"{st.mean(v1d2):>11.1f} {sd(v1d2):>6.1f} {e:>11.2f}")

# ---- part 3
if ds_path:
    ds = json.load(open(ds_path))["data"]
    primary = ds["primaryDispatchInterval"]
    sol = next(s for s in ds["solutionData"] if s["dispatchInterval"] == primary)
    print(f"\n--- vs Dispatch Solution at {primary} ---")
    print(f"{'battery':>16} {'chargeV2':>10} {'projCharge':>11} {'dischargeV2':>12} {'projDischarge':>14}")
    for fd in sol["facilityScheduleDetails"]:
        b = fd["facilityCode"]
        if b in series:
            r = next((r for r in series[b] if r["t"] == primary), None)
            if r:
                print(f"{b:>16} {r['c2']!s:>10} {fd['projectedChargeLevel']!s:>11} "
                      f"{r['d2']!s:>12} {fd['projectedDischargeLevel']!s:>14}")
            else:
                print(f"{b:>16}  no case at {primary} in the Case Input file (use files from the same day)")
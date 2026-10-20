"""Check what the demand tags in a Case Input file mean.

    python check_demand.py path/to/CaseInput.json [number_of_cases]

Columns:
  inject     sum of positive energy.sentOut (facilities sending power in)
  withdraw   sum of negative energy.sentOut (e.g. batteries charging)
  dem-inj    demand - inject            -> ~0 means demand = total injection
  und-dem    underlying - demand
  dpv        dispatchCondition.dpv
  und-dem-dpv                           -> ~0 means underlying = demand + rooftop PV
"""
import sys

import ijson

path = sys.argv[1]
n_cases = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def num(tags, name):
    v = tags.get(name)
    return float(v) if v not in (None, "") else float("nan")


cols = ["interval", "demand", "inject", "withdraw", "dem-inj", "forecast", "underlying", "und-dem", "dpv", "und-dem-dpv"]
print(f"{cols[0]:>25}" + "".join(f"{c:>12}" for c in cols[1:]))
with open(path, "rb") as f:
    for i, case in enumerate(ijson.items(f, "data.caseData.item", use_float=True)):
        if i >= n_cases:
            break
        tags = {s["tag"]: s["value"] for s in case["scada"]}
        sent_out = [float(v) for t, v in tags.items() if t.endswith(".energy.sentOut") and v not in (None, "")]
        inject = sum(v for v in sent_out if v > 0)
        withdraw = sum(v for v in sent_out if v < 0)
        demand = num(tags, "dispatchCondition.demand")
        underlying = num(tags, "dispatchCondition.underlyingDemand")
        dpv = num(tags, "dispatchCondition.dpv")
        forecast = case["constraintLookUp"].get("forecast.energy-forecastOperationalDemand-expected", float("nan"))
        row = [demand, inject, withdraw, demand - inject, forecast, underlying, underlying - demand, dpv, underlying - demand - dpv]
        print(f"{case['dispatchInterval']:>25}" + "".join(f"{v:12.1f}" for v in row))
"""Are inServiceQuantities / availableQuantities just the energy offer steps added up by capacityType?

    python check_capacity.py path/to/CaseInput.json path/to/DispatchSolution.json

Both files must be for the same trading day. Uses the Dispatch Solution's primary interval.
"""

import json
import sys
from collections import defaultdict

import ijson

ci_path, ds_path = sys.argv[1], sys.argv[2]

ds = json.load(open(ds_path))["data"]
primary = ds["primaryDispatchInterval"]
sol = next(s for s in ds["solutionData"] if s["dispatchInterval"] == primary)

totals = defaultdict(float)  # (capacityType, injection/withdrawal) -> MW
found = False
with open(ci_path, "rb") as f:
    for case in ijson.items(f, "data.caseData.item", use_float=True):
        if case["dispatchInterval"] != primary:
            continue
        found = True
        for fac in case["markets"]["energy"]["facilities"]:
            if "tranches" not in fac:
                print("no tranches:", fac.get("facilityCode"), sorted(fac.keys()))
                continue
            for tr in fac["tranches"]:
                q = tr["quantity"] or 0.0
                side = "injection" if q >= 0 else "withdrawal"
                totals[(tr["capacityType"], side)] += q
        break
if not found:
    sys.exit(f"No case at {primary}: use files from the same trading day.")

print(f"interval {primary}\n")
print(f"{'':<34} {'offer steps':>12} {'solution':>10}")
for label, key, value in [
    ("IN-SERVICE injection", ("IN-SERVICE", "injection"), sol["inServiceQuantities"]["energyInjectionCapacity"]),
    ("IN-SERVICE withdrawal", ("IN-SERVICE", "withdrawal"), sol["inServiceQuantities"]["energyWithdrawalCapacity"]),
    ("AVAILABLE injection", ("AVAILABLE", "injection"), sol["availableQuantities"]["energyInjectionCapacity"]),
    ("AVAILABLE withdrawal", ("AVAILABLE", "withdrawal"), sol["availableQuantities"]["energyWithdrawalCapacity"]),
]:
    print(f"{label:<34} {totals.get(key, 0.0):12.1f} {value:10.1f}")
others = {k: v for k, v in totals.items() if k[0] not in ("IN-SERVICE", "AVAILABLE")}
if others:
    print("\nother capacityType values:", {f"{k[0]} {k[1]}": round(v, 1) for k, v in others.items()})

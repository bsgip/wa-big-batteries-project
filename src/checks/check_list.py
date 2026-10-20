"""Work through the 'What to check next' list in the WEMDE data dictionary.

    python check_list.py [folder]

Reads every .json file in the folder (default: your sample_raw_data folder).
  Case Input files   (data.caseData)      -> checks 4, 6, and inputs for 2, 3
  Dispatch Solutions (data.solutionData)  -> checks 1, 2, 3 (Dispatch) and 5 (Pre-Dispatch)
Checks 2 and 3 need a Case Input file that covers a Dispatch Solution's primary interval
(i.e. the same trading day).
"""
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import ijson

DEFAULT = "/Users/noahdimonti/Developer/Work/wa-big-batteries-project/data/sample_raw_data"
KNOWN_BATTERIES = {"COLLIE_ESR1", "COLLIE_ESR4", "COLLIE_ESR5", "COLLIE_BESS2",
                   "KWINANA_ESR1", "KWINANA_ESR2", "ALINTA_WGP_ESR1"}
folder = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def file_kind(path):
    """'case', 'solution' or None, from the first array key under data."""
    with open(path, "rb") as f:
        for prefix, event, _ in ijson.parse(f):
            if event == "start_array" and prefix == "data.caseData":
                return "case"
            if event == "start_array" and prefix == "data.solutionData":
                return "solution"
    return None


def read_solution(path):
    """Stream a Dispatch Solution: per interval, per battery -> targets and projected levels."""
    primary = None
    with open(path, "rb") as f:
        for prefix, _, value in ijson.parse(f):
            if prefix == "data.primaryDispatchInterval":
                primary = value
                break
    rows = []
    with open(path, "rb") as f:
        for sol in ijson.items(f, "data.solutionData.item", use_float=True):
            q = {(svc["marketService"], fs["facilityCode"]): fs["quantity"] or 0.0
                 for svc in sol["schedule"] for fs in svc["facilitySchedule"]}
            bats = {}
            for d in sol["facilityScheduleDetails"]:
                c = d["facilityCode"]
                if c in KNOWN_BATTERIES or d.get("projectedDischargeLevel") is not None:
                    bats[c] = {"e": q.get(("energy", c), 0.0), "rr": q.get(("regulationRaise", c), 0.0),
                               "rl": q.get(("regulationLower", c), 0.0),
                               "dis": d.get("projectedDischargeLevel"), "chg": d.get("projectedChargeLevel"),
                               "init": d.get("initialMw")}
            rows.append((sol["dispatchInterval"], sol["dispatchType"], bats))
    return primary, rows


def projection_errors(rows, fac, minutes):
    """Max error of: drop in discharge level = [(P_prev+P)/2 + 0.25 RR - 0.25 RL] x minutes/60."""
    errs = []
    for (_, _, a), (_, _, b) in zip(rows, rows[1:]):
        if fac in a and fac in b and None not in (a[fac]["dis"], b[fac]["dis"]):
            pred = ((a[fac]["e"] + b[fac]["e"]) / 2 + 0.25 * b[fac]["rr"] - 0.25 * b[fac]["rl"]) * minutes / 60
            errs.append(abs((a[fac]["dis"] - b[fac]["dis"]) - pred))
    return max(errs) if errs else None


files = sorted(folder.glob("*.json"))
cases, sols = [], []
for p in files:
    k = file_kind(p)
    (cases if k == "case" else sols if k == "solution" else []).append(p)
print(f"Folder: {folder}\nCase Input files: {len(cases)}   Dispatch Solution files: {len(sols)}\n")

solutions = {p.name: read_solution(p) for p in sols}

# ---------------------------------------------------------------- 1 and 5
for name, (primary, rows) in solutions.items():
    dtype = rows[0][1] if rows else "?"
    minutes = 5 if dtype == "Dispatch" else 30
    label = "1. Projection formula" if dtype == "Dispatch" else "5. Pre-Dispatch projected levels"
    print(f"{label}  [{name}, {dtype}, {len(rows)} intervals]")
    for fac in sorted({f for _, _, b in rows for f in b}):
        dis = [b[fac]["dis"] for _, _, b in rows if fac in b]
        chg = [b[fac]["chg"] for _, _, b in rows if fac in b]
        nulls = sum(v is None for v in dis)
        if nulls == len(dis):
            print(f"   {fac:<16} all projected levels are null")
            continue
        vals = [v for v in dis if v is not None]
        totals = [d - c for d, c in zip(dis, chg) if None not in (d, c)]
        err = projection_errors(rows, fac, minutes)
        frozen = "  FROZEN (never changes)" if len(set(vals)) == 1 else ""
        print(f"   {fac:<16} max formula error {err if err is None else round(err, 4)!s:>8} MWh   "
              f"usable MWh {st.mean(totals):8.1f} (sd {st.pstdev(totals):.3f})   nulls {nulls}{frozen}")
    print()

# ---------------------------------------------------------------- one pass over each Case Input file
wanted = {primary for primary, _ in solutions.values() if primary}
live = {}                                   # interval -> {battery: tags}
storage_flags = defaultdict(set)            # battery -> {(flag, effective_from)}
print("6. Case Input structure")
for p in cases:
    prim, di, msp, scen, n, no_v2 = set(), set(), set(), set(), 0, 0
    with open(p, "rb") as f:
        for case in ijson.items(f, "data.caseData.item", use_float=True):
            n += 1
            scen.add(case.get("scenario"))
            if not any(s["tag"].endswith(".dischargeLevelV2") for s in case["scada"]):
                no_v2 += 1
            prim.add(case["primaryDispatchInterval"])
            di.add(case["dispatchInterval"])
            msp.add(case.get("constraintLookUp", {}).get("solver.minutesSincePrimary"))
            for r in case.get("registrationData", []):
                if r["facilityCode"] in KNOWN_BATTERIES:
                    sd = r.get("facilityStandingData") or {}
                    storage_flags[r["facilityCode"]].add((sd.get("storageConstraints"), sd.get("effectiveDispatchIntervalFrom")))
            if case["dispatchInterval"] in wanted:
                tags = {s["tag"]: s["value"] for s in case["scada"]}
                live[case["dispatchInterval"]] = {
                    b: {k: num(tags.get(f"{b}.{k}")) for k in
                        ("chargeLevelV2", "dischargeLevelV2", "energy.sentOut", "pdiSentOut")}
                    for b in KNOWN_BATTERIES}
    print(f"   {p.name}: {n} cases, {len(prim)} unique primary intervals, {len(di)} unique intervals, "
          f"minutesSincePrimary values {sorted(x for x in msp if x is not None)}")
    print(f"   first {min(prim)}  last {max(prim)}   scenarios {sorted(map(str, scen))}   "
          f"cases with no V2 tags: {no_v2}")
print()

# ---------------------------------------------------------------- 4
print("4. storageConstraints flag (from registrationData)")
for b in sorted(KNOWN_BATTERIES):
    vals = storage_flags.get(b)
    if not vals:
        print(f"   {b:<16} not found")
        continue
    changed = "  CHANGES within these files" if len({v[0] for v in vals}) > 1 else ""
    print(f"   {b:<16} {sorted(vals, key=str)}{changed}")
print()

# ---------------------------------------------------------------- 2 and 3
print("2-3. Live tags vs Dispatch Solution at the primary interval")
matched = False
for name, (primary, rows) in solutions.items():
    if rows[0][1] != "Dispatch" or primary not in live:
        continue
    matched = True
    first = next(b for t, _, b in rows if t == primary)
    print(f"   [{name}]  live tags are ~35 s before {primary}; projected levels are for the END of that interval")
    print(f"   {'battery':<16} {'dischV2':>9} {'projDisch':>10} {'diff':>7} {'chgV2':>9} {'projChg':>9} {'diff':>7}"
          f" {'sentOut':>8} {'pdiSentOut':>10} {'initialMw':>9}")
    for b in sorted(KNOWN_BATTERIES):
        L, s = live[primary].get(b, {}), first.get(b, {})
        d2, pd = L.get("dischargeLevelV2"), s.get("dis")
        c2, pc = L.get("chargeLevelV2"), s.get("chg")
        dd = round(d2 - pd, 2) if None not in (d2, pd) else None
        dc = round(c2 - pc, 2) if None not in (c2, pc) else None
        print(f"   {b:<16} {d2!s:>9} {pd!s:>10} {dd!s:>7} {c2!s:>9} {pc!s:>9} {dc!s:>7}"
              f" {L.get('energy.sentOut')!s:>8} {L.get('pdiSentOut')!s:>10} {s.get('init')!s:>9}")
if not matched:
    print("   No match: put a Dispatch Solution and a Case Input file from the SAME trading day in the folder.")
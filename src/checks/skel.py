import json
import sys


d = json.load(open(sys.argv[1]))
def skel(o, depth=0):
    pad = "  " * depth
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                n = f"[{len(v)}]" if isinstance(v, list) else "{}"
                print(f"{pad}{k}: {type(v).__name__}{n}")
                skel(v, depth + 1)
            else:
                print(f"{pad}{k}: {repr(v)[:60]}")
    elif isinstance(o, list) and o:
        skel(o[0], depth + 1)
skel(d)
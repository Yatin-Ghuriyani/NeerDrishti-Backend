import urllib.request
import json
import sys

endpoints = [
    "/api/health",
    "/api/model/metadata",
    "/api/model/surface?variable=temperature",
    "/api/model/depth-slice?variable=temperature&depth_m=0",
    "/api/model/point?lat=16.662&lon=78.343",
    "/api/comparison/profile?platform_number=2902199&cycle_number=255",
    "/api/bathymetry/grid",
    "/api/bathymetry/depth?lat=16.662&lon=78.343"
]

base = "https://sih2026-xdr2.onrender.com"

print("==================================================================")
print("AUDITING ORIGINAL RENDER BACKEND ENDPOINTS (https://sih2026-xdr2.onrender.com)")
print("==================================================================")

for ep in endpoints:
    url = base + ep
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print(f"\n[OK] [{ep}]")
            if isinstance(data, dict):
                keys = list(data.keys())
                print(f"   Keys ({len(keys)}): {keys}")
                for k in keys:
                    val = data[k]
                    if isinstance(val, list):
                        print(f"   - {k}: list of len {len(val)} (sample: {val[:2] if len(val)>0 else []})")
                    elif isinstance(val, dict):
                        print(f"   - {k}: dict with keys {list(val.keys())}")
                    else:
                        print(f"   - {k}: {val}")
            else:
                print(f"   Response type: {type(data)}")
    except Exception as e:
        print(f"\n[FAIL] [{ep}] Failed: {e}")

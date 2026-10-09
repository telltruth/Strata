#!/usr/bin/env python3
"""Fail closed unless a running Strata server actually allocated 8x256K."""
import argparse
import json
import os
from urllib.request import Request, urlopen

def check(status, slots=8, context=262144):
    if not status.get("loaded"):
        raise ValueError("model is not loaded")
    actual_context = (status.get("context") or {}).get("max_positions")
    actual_slots = (status.get("concurrency") or {}).get("serving")
    if not isinstance(actual_context, int) or actual_context < context:
        raise ValueError(f"context only {actual_context}, expected at least {context}")
    if actual_slots != slots:
        raise ValueError(f"allocated {actual_slots} slots, expected {slots}")
    return {"verified_context_ceiling": actual_context, "allocated_slots": actual_slots,
            "model": status.get("model"),
            "warning": "Allocation proves neither full simultaneous histories nor 200 tok/s."}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--key", default=os.environ.get("STRATA_API_KEY", ""))
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--context", type=int, default=262144)
    a = parser.parse_args()
    headers = {"Authorization": f"Bearer {a.key}"} if a.key else {}
    with urlopen(Request(a.url.rstrip("/") + "/v1/status", headers=headers), timeout=30) as response:
        status = json.load(response)
    print(json.dumps(check(status, a.slots, a.context), indent=2))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure real 1-8 slot end-to-end throughput via Strata's OpenAI API."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time
from urllib.request import Request, urlopen

def ask(url, key, model, prompt, max_tokens, barrier=None, timeout=600):
    if barrier is not None:
        barrier.wait(timeout=30)
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "reasoning_effort": "xhigh",
                          "stream": False}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = Request(url.rstrip("/") + "/v1/chat/completions",
                      data=payload, headers=headers, method="POST")
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    count = (result.get("usage") or {}).get("completion_tokens")
    if not isinstance(count, int) or count < 0:
        raise RuntimeError("response lacks usage.completion_tokens")
    choices = result.get("choices") or []
    return {"tokens": count, "latency_s": round(time.perf_counter()-started, 3),
            "finish_reason": choices[0].get("finish_reason") if choices else None}

def measure(url, key, model, slots, max_tokens, prompt, timeout, target):
    if not 1 <= slots <= 8:
        raise ValueError("slots must be 1..8")
    # Warmup is not included in the measured interval.
    ask(url, key, model, "Reply ready.", 32, timeout=timeout)
    barrier = threading.Barrier(slots)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=slots) as executor:
        futures = [executor.submit(ask, url, key, model, prompt,
                    max_tokens, barrier, timeout) for _ in range(slots)]
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - start
    total = sum(item["tokens"] for item in results)
    rate = total / elapsed if elapsed > 0 else 0.0
    return {"slots": slots, "aggregate_tok_s": round(rate, 2),
            "total_completion_tokens": total, "wall_time_s": round(elapsed, 3),
            "target_tok_s": target, "target_reached": rate >= target,
            "requests": results,
            "measurement": "end-to-end including prefill, scheduling and waiting"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--key", default=os.environ.get("STRATA_API_KEY", ""))
    parser.add_argument("--model", default="strata")
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--target", type=float, default=200.0)
    parser.add_argument("--prompt", default="Analyze a Linux kernel driver race with two competing interrupts. Consider failure modes, propose a minimal fix, regression tests and side effects.")
    parser.add_argument("--output", help="write JSON results to this file")
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    result = measure(args.url, args.key, args.model, args.slots,
                     args.max_tokens, args.prompt, args.timeout, args.target)
    result_json = json.dumps(result, indent=2) + "\n"
    print(result_json)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(result_json)

if __name__ == "__main__":
    main()

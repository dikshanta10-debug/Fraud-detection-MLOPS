"""
Latency and throughput benchmark for the inference service.

Produces the numbers you can quote directly: p50/p95/p99 inference latency,
single-request throughput, and batch throughput.

Usage:
    python -m scripts.benchmark                    # in-process (no server needed)
    python -m scripts.benchmark --url http://localhost:8000   # against a running server
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np


def build_transaction(rng: np.random.Generator, fraudulent: bool = False) -> dict:
    """Generate one transaction payload drawn from the training distribution."""
    v = rng.standard_normal(28)
    if fraudulent:
        for idx in (2, 3, 9, 10, 11, 13, 15, 16, 17):
            v[idx] += rng.normal(2.2, 0.8)

    return {
        "Time": float(rng.uniform(0, 172_800)),
        "Amount": round(float(rng.lognormal(3.2, 1.1)), 2),
        **{f"V{i + 1}": float(v[i]) for i in range(28)},
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarise(name: str, latencies_ms: list[float], wall_seconds: float, n: int) -> dict:
    stats = {
        "scenario": name,
        "requests": n,
        "wall_seconds": round(wall_seconds, 3),
        "throughput_per_sec": round(n / max(wall_seconds, 1e-9), 1),
        "mean_ms": round(statistics.fmean(latencies_ms), 3),
        "p50_ms": round(percentile(latencies_ms, 50), 3),
        "p95_ms": round(percentile(latencies_ms, 95), 3),
        "p99_ms": round(percentile(latencies_ms, 99), 3),
        "max_ms": round(max(latencies_ms), 3),
    }
    print(
        f"\n{name}\n{'-' * len(name)}\n"
        f"  requests      {stats['requests']:,}\n"
        f"  throughput    {stats['throughput_per_sec']:,.1f} req/sec\n"
        f"  mean          {stats['mean_ms']:.3f} ms\n"
        f"  p50           {stats['p50_ms']:.3f} ms\n"
        f"  p95           {stats['p95_ms']:.3f} ms\n"
        f"  p99           {stats['p99_ms']:.3f} ms\n"
        f"  max           {stats['max_ms']:.3f} ms"
    )
    return stats


def run_in_process(n_requests: int, batch_size: int) -> list[dict]:
    """Benchmark via TestClient — measures the full request path without network."""
    from fastapi.testclient import TestClient
    from app.main import app

    rng = np.random.default_rng(42)
    results = []

    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"], "Model not loaded — run training first"

        # Warm-up: first calls pay one-off allocation costs.
        for _ in range(50):
            client.post("/predict", json=build_transaction(rng))

        # ── Single prediction ─────────────────────────
        payloads = [build_transaction(rng, fraudulent=(i % 250 == 0)) for i in range(n_requests)]
        model_latencies, e2e_latencies = [], []

        start = time.perf_counter()
        for payload in payloads:
            t0 = time.perf_counter()
            r = client.post("/predict", json=payload)
            e2e_latencies.append((time.perf_counter() - t0) * 1000)
            model_latencies.append(r.json()["inference_latency_ms"])
        wall = time.perf_counter() - start

        results.append(summarise("Model inference only", model_latencies, wall, n_requests))
        results.append(summarise("End-to-end HTTP request", e2e_latencies, wall, n_requests))

        # ── Batch prediction ──────────────────────────
        n_batches = max(n_requests // batch_size, 1)
        batch_latencies = []
        start = time.perf_counter()
        for _ in range(n_batches):
            batch = [build_transaction(rng) for _ in range(batch_size)]
            r = client.post("/predict/batch", json={"transactions": batch})
            batch_latencies.append(r.json()["total_latency_ms"])
        wall = time.perf_counter() - start

        total_rows = n_batches * batch_size
        print(
            f"\nBatch inference (batch_size={batch_size})\n"
            f"{'-' * 40}\n"
            f"  batches       {n_batches:,}\n"
            f"  rows scored   {total_rows:,}\n"
            f"  throughput    {total_rows / wall:,.0f} rows/sec\n"
            f"  per-batch p95 {percentile(batch_latencies, 95):.3f} ms\n"
            f"  per-row mean  {statistics.fmean(batch_latencies) / batch_size:.4f} ms"
        )
        results.append({
            "scenario": f"batch_{batch_size}",
            "rows": total_rows,
            "throughput_rows_per_sec": round(total_rows / wall, 1),
            "per_row_mean_ms": round(statistics.fmean(batch_latencies) / batch_size, 4),
        })

    return results


def run_against_server(url: str, n_requests: int) -> list[dict]:
    """Benchmark a running server over real HTTP."""
    import httpx

    rng = np.random.default_rng(42)

    with httpx.Client(base_url=url, timeout=30.0) as client:
        health = client.get("/health").json()
        assert health["model_loaded"], "Model not loaded on server"

        for _ in range(50):
            client.post("/predict", json=build_transaction(rng))

        model_latencies, e2e_latencies = [], []
        start = time.perf_counter()
        for _ in range(n_requests):
            payload = build_transaction(rng)
            t0 = time.perf_counter()
            r = client.post("/predict", json=payload)
            e2e_latencies.append((time.perf_counter() - t0) * 1000)
            model_latencies.append(r.json()["inference_latency_ms"])
        wall = time.perf_counter() - start

    return [
        summarise("Model inference only", model_latencies, wall, n_requests),
        summarise("End-to-end HTTP (network)", e2e_latencies, wall, n_requests),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the fraud detection API")
    parser.add_argument("--url", default=None, help="Benchmark a running server instead of in-process")
    parser.add_argument("-n", "--requests", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--out", default="benchmark_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Fraud Detection API — benchmark ({args.requests:,} requests)")
    print("=" * 60)

    if args.url:
        results = run_against_server(args.url, args.requests)
    else:
        results = run_in_process(args.requests, args.batch_size)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}\n")


if __name__ == "__main__":
    main()

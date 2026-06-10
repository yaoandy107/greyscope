#!/usr/bin/env python3
"""Latency + memory benchmark: Greyscope vs the open trained detectors, on this machine.

Times a single forward pass (detection is one pass, no generation) at a few sequence
lengths. Latency depends on architecture, size, and dtype — not weight values — so the
two LoRA detectors are reconstructed from their cached bases; roberta loads its released
checkpoint. Zero-shot detectors are excluded (undisclosed backbones, nothing to time).

Without flash-linear-attention (CUDA-only), Qwen3.5 takes the portable torch path, so
Mac numbers are a conservative floor.

    python scripts/bench_mac_latency.py
    python scripts/bench_mac_latency.py --dtype float32 --runs 30
    python scripts/bench_mac_latency.py --greyscope-path outputs/production_v2/merged
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import torch

# Detectors to time. `reconstruct` rebuilds a seq-cls head on the cached base
# (faithful for latency/memory); else the real checkpoint at `repo` is loaded.
DETECTORS = [
    {"label": "Greyscope (Qwen3.5-4B GDN)", "repo": "unsloth/Qwen3.5-4B-Base",
     "reconstruct": True, "n_labels": 4, "max_seq": 1024},
    {"label": "editlens-Llama-3.2-3B", "repo": "unsloth/Llama-3.2-3B",
     "reconstruct": True, "n_labels": 4, "max_seq": 1024},
    {"label": "editlens-roberta-large", "repo": "pangram/editlens_roberta-large",
     "reconstruct": False, "n_labels": None, "max_seq": 512},
]
SEQ_LENS = [128, 512, 1024]


def _device_and_dtype(dtype_arg: str):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_arg]
    return device, dtype


def _mem_gb(device: str) -> float:
    """Live process allocation on the accelerator (GB). Unified memory on MPS."""
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1024**3
    if device == "cuda":
        return torch.cuda.memory_allocated() / 1024**3
    return float("nan")


def _free(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _load(spec: dict, device: str, dtype):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["repo"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if spec["reconstruct"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            spec["repo"], num_labels=spec["n_labels"], dtype=dtype)
        # Qwen3.5's VLM-composite config silently ignores num_labels; resize the head.
        if hasattr(model, "score") and model.score.out_features != spec["n_labels"]:
            model.score = torch.nn.Linear(model.score.in_features, spec["n_labels"], bias=False).to(dtype=dtype)
            model.config.num_labels = spec["n_labels"]
    else:
        model = AutoModelForSequenceClassification.from_pretrained(spec["repo"], dtype=dtype)

    model.config.pad_token_id = tok.pad_token_id
    return model.to(device).eval(), tok


def _bench(spec: dict, device: str, dtype, runs: int, warmup: int) -> dict:
    print(f"\n=== {spec['label']}  ({spec['repo']}) ===", flush=True)
    _free(device)
    base_mem = _mem_gb(device)

    t0 = time.perf_counter()
    model, _ = _load(spec, device, dtype)
    _sync(device)
    load_s = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters()) / 1e9
    peak_mem = _mem_gb(device)
    print(f"  loaded in {load_s:.1f}s | params={params:.2f}B | mem≈{peak_mem - base_mem:.1f}GB", flush=True)

    vocab = int(getattr(model.config, "vocab_size", 32000))
    seq_lens = [s for s in SEQ_LENS if s <= spec["max_seq"]]
    rows = []
    for seq_len in seq_lens:
        g = torch.Generator().manual_seed(seq_len)
        ids = torch.randint(0, vocab, (1, seq_len), generator=g).to(device)
        mask = torch.ones_like(ids)

        def _step(m=model):
            with torch.no_grad():
                m(input_ids=ids, attention_mask=mask)
            _sync(device)

        for _ in range(warmup):
            _step()
        times = []
        for _ in range(runs):
            s = time.perf_counter()
            _step()
            times.append((time.perf_counter() - s) * 1000)
        peak_mem = max(peak_mem, _mem_gb(device))
        med = statistics.median(times)
        rows.append({"seq_len": seq_len, "median_ms": med, "tok_per_s": seq_len / (med / 1000)})
        print(f"  seq={seq_len:>4}: median={med:8.1f} ms  ({seq_len/(med/1000):,.0f} tok/s)", flush=True)

    del model
    _free(device)
    return {"label": spec["label"], "params": params, "mem_gb": peak_mem - base_mem,
            "load_s": load_s, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--greyscope-path", default=None,
                    help="Local path to the merged Greyscope seq-cls dir (else reconstruct from base).")
    args = ap.parse_args()

    device, dtype = _device_and_dtype(args.dtype)
    print(f"device={device}  dtype={dtype}  torch={torch.__version__}", flush=True)

    detectors = [dict(d) for d in DETECTORS]
    if args.greyscope_path:  # measure the real merged model if weights are available locally
        detectors[0]["repo"], detectors[0]["reconstruct"] = args.greyscope_path, False

    results = [_bench(d, device, dtype, args.runs, args.warmup) for d in detectors]

    print("\n" + "=" * 78)
    print(f"{'detector':<28} {'params':>7} {'mem':>7} {'128tok':>9} {'512tok':>9} {'1024tok':>9}")
    print("-" * 78)
    for r in results:
        ms = {row["seq_len"]: f"{row['median_ms']:.0f}ms" for row in r["rows"]}
        print(f"{r['label']:<28} {r['params']:>6.2f}B {r['mem_gb']:>6.1f}G "
              f"{ms.get(128,'-'):>9} {ms.get(512,'-'):>9} {ms.get(1024,'-'):>9}")
    print("=" * 78)
    print(f"{device}, {str(dtype).split('.')[-1]}, batch=1, median of {args.runs} runs. roberta caps at 512 tokens.")
    print("Greyscope uses the portable GDN path here (no flash-linear-attention); its CUDA "
          "kernel is faster. Latency/memory are architecture-faithful (weight values don't affect them).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Produces a tiny test model + reference forward output.

  y = ((x @ W1) silu) @ W2

Writes test_mlp.gguf (weights + reference output) and
test_mlp.topology.json (graph spec for the C runner).

Pure stdlib — no numpy / no PyTorch. Computes the reference in plain
Python in fp32, then quantises down to fp16 for storage.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from booki_pack import Tensor, pack


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def randn(rng: random.Random) -> float:
    return rng.gauss(0.0, 1.0)


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Plain Python row-major matrix multiply."""
    M, K = len(a), len(a[0])
    Kb, N = len(b), len(b[0])
    assert K == Kb
    out = [[0.0] * N for _ in range(M)]
    for i in range(M):
        ai = a[i]
        oi = out[i]
        for k in range(K):
            aik = ai[k]
            bk  = b[k]
            for j in range(N):
                oi[j] += aik * bk[j]
    return out


def f16_round(x: float) -> float:
    """Round-trip a Python float through IEEE-754 binary16."""
    import struct
    return struct.unpack("<e", struct.pack("<e", x))[0]


def make_model(seed: int = 7,
               batch: int = 4, in_dim: int = 8,
               hidden: int = 16, out_dim: int = 5):
    rng = random.Random(seed)
    x  = [[randn(rng) * 0.3 for _ in range(in_dim)]  for _ in range(batch)]
    W1 = [[randn(rng) * 0.3 for _ in range(hidden)]  for _ in range(in_dim)]
    W2 = [[randn(rng) * 0.3 for _ in range(out_dim)] for _ in range(hidden)]

    # Cast inputs + weights through fp16 first so the oracle matches
    # what the runtime sees after dtype conversion.
    x  = [[f16_round(v) for v in row] for row in x]
    W1 = [[f16_round(v) for v in row] for row in W1]
    W2 = [[f16_round(v) for v in row] for row in W2]

    h = matmul(x, W1)
    h = [[silu(v) for v in row] for row in h]
    y = matmul(h, W2)

    def flat(M):
        return [v for row in M for v in row]

    tensors = [
        Tensor.from_floats_f16("x",     (batch, in_dim),       flat(x)),
        Tensor.from_floats_f16("W1",    (in_dim, hidden),      flat(W1)),
        Tensor.from_floats_f16("W2",    (hidden, out_dim),     flat(W2)),
        Tensor.from_floats_f16("y_ref", (batch, out_dim),      flat(y)),
    ]
    topology = {
        "nodes": [
            {"op": "input",  "name": "x",  "dtype": "f16",
             "shape": [batch, in_dim]},
            {"op": "weight", "name": "W1"},
            {"op": "weight", "name": "W2"},
            {"op": "matmul", "inputs": ["x", "W1"],     "name": "z1"},
            {"op": "silu",   "inputs": ["z1"],          "name": "z1_act"},
            {"op": "matmul", "inputs": ["z1_act", "W2"], "name": "y"},
        ],
        "output": "y",
    }
    return tensors, topology


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-gguf",     required=True, type=Path)
    ap.add_argument("--out-topology", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    tensors, topology = make_model(seed=args.seed)
    pack(args.out_gguf, tensors,
         metadata={"booki.test_model": "mlp_silu_2_layer"})
    args.out_topology.write_text(json.dumps(topology, indent=2))

    total = sum(len(t.data) for t in tensors)
    print(f"wrote {args.out_gguf} ({total} bytes of tensors)", file=sys.stderr)
    print(f"wrote {args.out_topology}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Inspect an ONNX model and report what the Booki runtime would need
to consume it end-to-end.

Outputs:
  - op-type histogram (what ops the model uses + how often)
  - coverage report (which of those ops Booki already implements)
  - tensor count + total parameter bytes
  - input / output signatures
  - the first N nodes in execution order (architecture sketch)

Run with the venv that has `onnx` installed:
  ~/.booki-venv/bin/python tools/inspect_onnx.py path/to/model.onnx
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import onnx
from onnx import numpy_helper

# Operators booki_graph already exposes a constructor for. Keep this list
# in sync with runtime/native/booki_graph.h.
SUPPORTED_BOOKI_OPS = {
    "MatMul":           "booki_graph_matmul",
    "Gemm":             "booki_graph_matmul (+ bias-add)",
    "Add":              "booki_graph_add",
    "Mul":              "booki_graph_mul",
    "Softmax":          "booki_graph_softmax",
    "Gather":           "booki_graph_embedding",
    "Conv":             "booki_graph_conv1d (only if 1-D)",
    # Activations
    "Silu":             "booki_graph_silu",
    "Sigmoid":          "compose: x * sigmoid(x)",
    "Gelu":             "booki_graph_gelu",
    "Tanh":             "compose / NEON poly",
    # Norms — only RMSNorm right now
    "RmsNorm":          "booki_graph_rmsnorm",
    "LayerNormalization": "MISSING (needs booki_graph_layernorm)",
    # Attention (depending on representation)
    "Attention":        "booki_graph_attention (single composed op)",
    "MultiHeadAttention": "booki_graph_attention (single composed op)",
}

PASSTHROUGH_OPS = {
    # ONNX ops we can essentially ignore or fuse during conversion.
    "Identity", "Reshape", "Transpose", "Concat", "Slice", "Cast",
    "Constant", "Unsqueeze", "Squeeze", "Shape", "Gather", "Range",
    "Where", "Equal", "Less", "Greater", "Not", "Pow", "Sqrt", "Div",
    "ReduceMean", "ReduceSum", "Expand",
}


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def shape_str(tensor_type) -> str:
    dims = []
    for d in tensor_type.shape.dim:
        if d.dim_value:    dims.append(str(d.dim_value))
        elif d.dim_param:  dims.append(d.dim_param)
        else:              dims.append("?")
    return "[" + ", ".join(dims) + "]"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("--head", type=int, default=20,
                    help="show the first N nodes for architecture sketch")
    args = ap.parse_args(argv)

    model = onnx.load(args.model)
    graph = model.graph

    # 1. Op histogram
    histogram = collections.Counter(node.op_type for node in graph.node)

    # 2. Coverage classification
    rows = []
    for op, count in histogram.most_common():
        if op in SUPPORTED_BOOKI_OPS:
            status = f"OK  {SUPPORTED_BOOKI_OPS[op]}"
        elif op in PASSTHROUGH_OPS:
            status = "OK  (fold in converter, no kernel needed)"
        else:
            status = "MISSING"
        rows.append((count, op, status))

    print(f"Model: {args.model}")
    print(f"Producer: {model.producer_name} v{model.producer_version}")
    print(f"IR version: {model.ir_version}, opset(s): {[op.version for op in model.opset_import]}")
    print()

    # 3. IO
    print("Inputs:")
    for inp in graph.input:
        print(f"  {inp.name:40s} {inp.type.tensor_type.elem_type:>2}  {shape_str(inp.type.tensor_type)}")
    print("Outputs:")
    for out in graph.output:
        print(f"  {out.name:40s} {out.type.tensor_type.elem_type:>2}  {shape_str(out.type.tensor_type)}")
    print()

    # 4. Params
    total_params = sum(numpy_helper.to_array(init).nbytes for init in graph.initializer)
    print(f"Initializers: {len(graph.initializer)}, total bytes: {fmt_bytes(total_params)}")
    print(f"Graph nodes: {len(graph.node)}")
    print()

    # 5. Op coverage table
    print(f"{'count':>6}  {'op':30s}  status")
    print("-" * 78)
    missing = 0
    for count, op, status in rows:
        print(f"{count:>6}  {op:30s}  {status}")
        if status.startswith("MISSING"):
            missing += count
    print()
    print(f"Currently MISSING op-invocations: {missing} "
          f"({missing/sum(c for c,_,_ in rows):.1%} of total)")
    print()

    # 6. Architecture sketch — first N nodes
    print(f"First {args.head} nodes (execution order):")
    for node in graph.node[: args.head]:
        attrs = ", ".join(f"{a.name}" for a in node.attribute) or "-"
        ins = ",".join(node.input[:3])
        if len(node.input) > 3: ins += f",…+{len(node.input)-3}"
        outs = ",".join(node.output)
        print(f"  {node.op_type:25s}  in={ins:40s}  attrs={attrs}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

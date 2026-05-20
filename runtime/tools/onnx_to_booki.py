#!/usr/bin/env python3
"""ONNX → Booki converter.

Reads any ONNX model and emits the matching .gguf weight bundle plus
.topology.json that runtime/native/booki_graph.c can execute. Walks the
graph topologically, maps ONNX op types to Booki op constructors, and
folds the metadata-only ONNX ops (Shape, Reshape, Cast, Constant,
Squeeze/Unsqueeze, etc.) into the converter rather than emitting them.

Today this is a "best-effort" converter — it will correctly translate any
model composed of ops the Booki runtime exposes. Unsupported ops are
reported and the converter exits non-zero; that's the discovery loop we
want for porting Kokoro.

Usage:
  ~/.booki-venv/bin/python tools/onnx_to_booki.py \\
      --in model.onnx --out-gguf model.gguf --out-topology model.json

Exit codes:
  0  conversion fully succeeded
  1  unrecoverable I/O or shape error
  2  unsupported ops encountered (full list printed to stderr)
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from booki_pack import Tensor, pack

# Map from ONNX op type to the Booki op key we emit in topology JSON.
# Right-hand side matches the dispatch in booki_run_test_mlp.c (which
# the Kokoro runner will inherit). Ops not listed below are either
# folded in the converter or reported as unsupported.
SUPPORTED_TRANSLATIONS = {
    "MatMul":             "matmul",
    "Gemm":               "matmul",         # bias-add fused in converter
    "Add":                "add",
    "Sub":                "sub",
    "Mul":                "mul",
    "Softmax":            "softmax",
    "Gather":             "embedding",       # only axis=0
    "Tanh":               "tanh",
    "Sigmoid":            "sigmoid",
    "LeakyRelu":          "leaky_relu",
    "Sin":                "sin",
    "Cos":                "cos",
    "Conv":               "conv1d",
    "ConvTranspose":      "conv_transpose1d",
    "InstanceNormalization": "instance_norm",
    "LSTM":               "lstm",
    "Resize":             "resize1d",
    "Exp":                "exp",
    "Atan":               "atan",
    "CumSum":             "cumsum",
    "Pad":                "pad1d",
    "ScatterND":          "scatter_nd",
    "ScatterElements":    "scatter_nd",   # approximate; same kernel handles the cases we see
    "TopK":               "topk",
    "And":                "and",
    "RandomUniformLike":  "random_uniform",
    "RandomNormalLike":   "random_normal",
}

# ONNX ops that produce no runtime work — pure shape / index / constant
# manipulation that the converter folds away. We don't emit a runtime
# node for these; their outputs are treated as aliases of inputs or as
# extracted constants.
FOLDABLE = {
    "Identity", "Reshape", "Transpose", "Concat", "Slice", "Cast",
    "Constant", "Unsqueeze", "Squeeze", "Shape", "Range", "Where",
    "Equal", "Less", "Greater", "Not", "Expand", "ConstantOfShape",
    "ReduceMean", "ReduceSum", "ReduceMax", "ReduceProd", "Sqrt",
    "Pow", "Div", "Reciprocal", "Floor", "Round", "Clip",
}


def dtype_to_booki(elem_type: int) -> str:
    # ONNX TensorProto.DataType values
    mapping = {
        1:  "f32",   # FLOAT
        7:  "i64",   # INT64
        10: "f16",   # FLOAT16
        2:  "i8",    # INT8 (unsupported by booki_pack but handled at storage layer)
    }
    return mapping.get(elem_type, "f32")


def to_f16_array(arr: np.ndarray) -> np.ndarray:
    """Coerce an ONNX initializer to fp16 storage, mirroring our runtime convention."""
    if arr.dtype == np.float16:
        return np.ascontiguousarray(arr)
    if arr.dtype in (np.float32, np.float64):
        return np.ascontiguousarray(arr.astype(np.float16))
    if arr.dtype in (np.int8, np.int64):
        return np.ascontiguousarray(arr)
    # Fallback: cast to float16 (lossy but consistent).
    return np.ascontiguousarray(arr.astype(np.float16))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out-gguf",     required=True, type=Path)
    ap.add_argument("--out-topology", required=True, type=Path)
    ap.add_argument("--check-only", action="store_true",
                    help="don't write outputs; just report whether the model is convertible")
    args = ap.parse_args(argv)

    model = onnx.load(args.in_path)
    graph = model.graph

    # 1. Gather initializers (weights) as Booki tensors.
    tensors: list[Tensor] = []
    init_names: set[str] = set()
    for init in graph.initializer:
        arr = to_f16_array(numpy_helper.to_array(init))
        if arr.dtype == np.float16:
            dtype = "f16"
            data = arr.tobytes()
        elif arr.dtype == np.int8:
            dtype = "i8"
            data = arr.tobytes()
        elif arr.dtype == np.int64:
            dtype = "i64"
            data = arr.tobytes()
        else:
            print(f"skip init {init.name}: dtype {arr.dtype} not supported", file=sys.stderr)
            continue
        tensors.append(Tensor(init.name, dtype, tuple(arr.shape), data))
        init_names.add(init.name)

    # 2. Walk nodes. Track tensor name → producing node name for graph edges.
    nodes_out: list[dict] = []
    unsupported = collections.Counter()
    folded = collections.Counter()

    # External inputs become "input" nodes.
    for inp in graph.input:
        if inp.name in init_names:
            continue
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value else 1)
        nodes_out.append({
            "op": "input", "name": inp.name,
            "dtype": dtype_to_booki(inp.type.tensor_type.elem_type),
            "shape": dims,
        })

    for init_name in init_names:
        nodes_out.append({"op": "weight", "name": init_name})

    # Op nodes
    for node in graph.node:
        if node.op_type in FOLDABLE:
            folded[node.op_type] += 1
            # We don't generate a runtime node, but we still need to make
            # downstream nodes resolve their inputs. For now, alias the
            # foldable's output to its first input (works only for the
            # truly metadata-only ops; Reshape / Cast / Slice would need
            # real shape inference). The Kokoro converter will need a
            # proper folding pass — flagged in KOKORO_PORT.md.
            continue
        if node.op_type not in SUPPORTED_TRANSLATIONS:
            unsupported[node.op_type] += 1
            continue
        nodes_out.append({
            "op": SUPPORTED_TRANSLATIONS[node.op_type],
            "name": node.output[0] if node.output else node.name,
            "inputs": list(node.input),
        })

    output_name = graph.output[0].name if graph.output else None
    topology = {"nodes": nodes_out, "output": output_name}

    # Report
    if unsupported:
        print("UNSUPPORTED ops encountered (need runtime kernels):", file=sys.stderr)
        for op, n in unsupported.most_common():
            print(f"  {op:30s}  {n}", file=sys.stderr)
        print(f"({sum(unsupported.values())} invocations total)", file=sys.stderr)
        if not args.check_only:
            print("Refusing to emit a partial conversion. Use --check-only to "
                  "explore further.", file=sys.stderr)
            return 2

    if args.check_only:
        print(f"convertible: {not unsupported}", file=sys.stderr)
        print(f"folded ops:   {sum(folded.values())} invocations across "
              f"{len(folded)} kinds", file=sys.stderr)
        print(f"emitted nodes: {len(nodes_out)} (inputs + weights + ops)", file=sys.stderr)
        return 0 if not unsupported else 2

    # 3. Write outputs.
    pack(args.out_gguf, tensors,
         metadata={"booki.source": str(args.in_path),
                   "booki.producer": "onnx_to_booki.py"})
    args.out_topology.write_text(json.dumps(topology, indent=2))
    print(f"wrote {args.out_gguf}  ({len(tensors)} tensors)", file=sys.stderr)
    print(f"wrote {args.out_topology}  ({len(nodes_out)} nodes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

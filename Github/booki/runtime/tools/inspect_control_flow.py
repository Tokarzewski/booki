#!/usr/bin/env python3
"""Dump the bodies of If / Loop / SplitToSequence / ConcatFromSequence /
SequenceEmpty nodes in a model. Helps decide whether the runtime needs
real subgraph dispatch or whether the converter can fold them away.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx


def describe_subgraph(g, depth=2):
    indent = "    " * depth
    print(f"{indent}inputs:")
    for i in g.input:
        print(f"{indent}  {i.name}")
    print(f"{indent}outputs:")
    for o in g.output:
        print(f"{indent}  {o.name}")
    print(f"{indent}nodes ({len(g.node)}):")
    for n in g.node[:8]:
        attrs = ",".join(a.name for a in n.attribute) or "-"
        print(f"{indent}  {n.op_type:25s} in={','.join(n.input)[:60]:60s} attrs={attrs}")
    if len(g.node) > 8:
        print(f"{indent}  ... and {len(g.node) - 8} more nodes")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    args = ap.parse_args(argv)
    model = onnx.load(args.model)

    TARGETS = {"If", "Loop", "SplitToSequence", "ConcatFromSequence", "SequenceEmpty"}
    for node in model.graph.node:
        if node.op_type not in TARGETS:
            continue
        print(f"== {node.op_type}  name={node.name}")
        print(f"  inputs:  {list(node.input)}")
        print(f"  outputs: {list(node.output)}")
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                print(f"  attr {attr.name} (subgraph):")
                describe_subgraph(attr.g)
            else:
                print(f"  attr {attr.name} = {attr}")
        print()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

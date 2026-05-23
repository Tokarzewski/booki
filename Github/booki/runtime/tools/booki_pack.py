#!/usr/bin/env python3
"""Pack tensors into a Booki GGUF v3 file. Pure stdlib — no numpy.

The on-disk layout matches runtime/native/booki_gguf.c's reader:

  magic      "GGUF"
  uint32     version = 3
  uint64     tensor_count
  uint64     metadata_kv_count
  <metadata kv>*
  <tensor info>*
  <pad to 32-byte alignment>
  <tensor data>

Type mapping (must match runtime/native/booki_gguf.c):
  F32  = 0
  F16  = 1
  I8   = 24
  I64  = 27
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence

# Metadata value types
GGUF_TYPE_STRING = 8

# ggml tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I8  = 24
GGML_TYPE_I64 = 27

DTYPE_NAME_TO_GGML = {"f32": GGML_TYPE_F32, "f16": GGML_TYPE_F16,
                     "i8": GGML_TYPE_I8, "i64": GGML_TYPE_I64}

ALIGN_TO = 32


class Tensor:
    """Wrap a flat-byte buffer with shape + dtype metadata.

    For our test pipeline we generate values in pure Python, so this
    class also helps shuffle Python floats into the right encoding.
    """

    def __init__(self, name: str, dtype: str, shape: Sequence[int], data: bytes):
        assert dtype in DTYPE_NAME_TO_GGML, f"unsupported dtype {dtype}"
        self.name = name
        self.dtype = dtype
        self.shape = tuple(shape)
        self.data = data

    @classmethod
    def from_floats_f16(cls, name: str, shape: Sequence[int],
                        values: Sequence[float]) -> "Tensor":
        n = 1
        for d in shape:
            n *= d
        assert n == len(values), (n, len(values))
        # struct supports the 'e' format spec for IEEE-754 binary16 since
        # Python 3.6 — perfect for our fp16 tensors.
        data = struct.pack(f"<{n}e", *values)
        return cls(name, "f16", shape, data)

    @classmethod
    def from_floats_f32(cls, name: str, shape: Sequence[int],
                        values: Sequence[float]) -> "Tensor":
        n = 1
        for d in shape:
            n *= d
        assert n == len(values), (n, len(values))
        data = struct.pack(f"<{n}f", *values)
        return cls(name, "f32", shape, data)


def _w_u32(out, v: int) -> None: out.write(struct.pack("<I", v))
def _w_u64(out, v: int) -> None: out.write(struct.pack("<Q", v))
def _w_str(out, s: str) -> None:
    data = s.encode("utf-8")
    _w_u64(out, len(data))
    out.write(data)


def pack(path: Path, tensors: Sequence[Tensor],
         metadata: dict[str, str] | None = None) -> None:
    metadata = metadata or {}
    if not tensors:
        raise ValueError("at least one tensor required")

    with path.open("wb") as out:
        out.write(b"GGUF")
        _w_u32(out, 3)
        _w_u64(out, len(tensors))
        _w_u64(out, len(metadata))

        for key, value in metadata.items():
            _w_str(out, key)
            _w_u32(out, GGUF_TYPE_STRING)
            _w_str(out, str(value))

        # Tensor info entries with placeholder offsets to patch later.
        offset_patches: list[int] = []
        for t in tensors:
            _w_str(out, t.name)
            _w_u32(out, len(t.shape))
            for dim in reversed(t.shape):           # GGUF stores innermost-first
                _w_u64(out, dim)
            _w_u32(out, DTYPE_NAME_TO_GGML[t.dtype])
            offset_patches.append(out.tell())
            _w_u64(out, 0)

        # Align to ALIGN_TO; data section starts at this position.
        cur = out.tell()
        pad = (-cur) % ALIGN_TO
        out.write(b"\x00" * pad)
        data_start = out.tell()

        offsets = []
        for t in tensors:
            offsets.append(out.tell() - data_start)
            out.write(t.data)

        for pos, off in zip(offset_patches, offsets):
            cur = out.tell()
            out.seek(pos)
            _w_u64(out, off)
            out.seek(cur)

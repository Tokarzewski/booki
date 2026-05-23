/* booki_sequence — variable-length list of tensors with arena-backed
 * storage. Supports the ONNX sequence + Loop ops needed by Kokoro's
 * frame-expansion path.
 *
 * Design choices:
 *   - Sequence storage lives in the parent arena, so it goes away when
 *     the parent arena resets. No separate lifetime to manage.
 *   - Tensor entries hold copies of the descriptor metadata + a pointer
 *     into the underlying arena; the data itself isn't copied.
 */

#include "booki_native.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SEQ_INITIAL_CAP 16

struct booki_sequence {
    booki_tensor* items;
    int           len;
    int           cap;
    booki_arena*  arena;   /* used for grow + element allocations */
};

booki_sequence* booki_sequence_new(booki_arena* arena) {
    if (!arena) return NULL;
    /* Allocate the sequence header from the arena so it dies with it. */
    booki_sequence* s = (booki_sequence*)booki_arena_alloc(
        arena, sizeof(booki_sequence), 8);
    if (!s) return NULL;
    s->len = 0;
    s->cap = SEQ_INITIAL_CAP;
    s->items = (booki_tensor*)booki_arena_alloc(
        arena, sizeof(booki_tensor) * s->cap, 16);
    if (!s->items) return NULL;
    s->arena = arena;
    return s;
}

int booki_sequence_len(const booki_sequence* s) { return s ? s->len : 0; }

int booki_sequence_push(booki_sequence* s, const booki_tensor* t) {
    if (!s || !t) return -1;
    if (s->len == s->cap) {
        int new_cap = s->cap * 2;
        booki_tensor* new_items = (booki_tensor*)booki_arena_alloc(
            s->arena, sizeof(booki_tensor) * new_cap, 16);
        if (!new_items) return -2;
        memcpy(new_items, s->items, sizeof(booki_tensor) * s->len);
        s->items = new_items;
        s->cap = new_cap;
    }
    s->items[s->len++] = *t;
    return 0;
}

int booki_sequence_at(const booki_sequence* s, int idx, booki_tensor* out) {
    if (!s || !out || idx < 0 || idx >= s->len) return -1;
    *out = s->items[idx];
    return 0;
}

/* ------------------------------------------------------------------------- */
/* SplitToSequence                                                            */
/*                                                                            */
/* Splits [x] along [axis] into ceil(x.shape[axis]/chunk) tensors. All but   */
/* possibly the last share the chunk size; the last has the remainder. Each  */
/* slice is materialised as a separate contiguous tensor in [arena].         */
/* ------------------------------------------------------------------------- */

int booki_split_to_sequence_f16(const booki_tensor* x, int axis, int64_t chunk,
                                booki_sequence* out_seq, booki_arena* arena) {
    if (!x || !out_seq || !arena) return -1;
    if (x->dtype != BOOKI_DTYPE_F16) return -2;
    if (axis < 0 || axis >= x->rank) return -3;
    if (chunk <= 0) return -4;

    int64_t axis_len = x->shape[axis];
    int64_t n_chunks = (axis_len + chunk - 1) / chunk;

    /* Outer + inner strides relative to the split axis. */
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i)         outer *= x->shape[i];
    for (int i = axis + 1; i < x->rank; ++i) inner *= x->shape[i];

    const uint16_t* xp = (const uint16_t*)x->data;

    for (int64_t c = 0; c < n_chunks; ++c) {
        int64_t start = c * chunk;
        int64_t this_chunk = (start + chunk <= axis_len) ? chunk : (axis_len - start);

        /* Shape of this slice: same as x but with axis dim = this_chunk. */
        int64_t shape[BOOKI_MAX_RANK];
        for (int i = 0; i < x->rank; ++i) shape[i] = x->shape[i];
        shape[axis] = this_chunk;

        booki_tensor slice = booki_tensor_arena(arena, BOOKI_DTYPE_F16, x->rank, shape);
        if (!slice.data) return -5;
        uint16_t* sp = (uint16_t*)slice.data;

        /* Strided memcpy. */
        int64_t per_outer_in  = axis_len * inner;
        int64_t per_outer_out = this_chunk * inner;
        for (int64_t o = 0; o < outer; ++o) {
            const uint16_t* src = xp + o * per_outer_in + start * inner;
            uint16_t*       dst = sp + o * per_outer_out;
            memcpy(dst, src, this_chunk * inner * sizeof(uint16_t));
        }

        int rc = booki_sequence_push(out_seq, &slice);
        if (rc) return -6;
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* ConcatFromSequence                                                         */
/* ------------------------------------------------------------------------- */

int booki_concat_from_sequence_f16(const booki_sequence* seq, int axis,
                                   booki_arena* arena, booki_tensor* out) {
    if (!seq || !arena || !out) return -1;
    int N = booki_sequence_len(seq);
    if (N == 0) return -2;

    const booki_tensor* first = &seq->items[0];
    if (first->dtype != BOOKI_DTYPE_F16) return -3;
    if (axis < 0 || axis >= first->rank) return -4;

    /* Compute output shape: sum along the concat axis, identical elsewhere. */
    int64_t shape[BOOKI_MAX_RANK];
    for (int i = 0; i < first->rank; ++i) shape[i] = first->shape[i];
    int64_t total_axis = 0;
    for (int n = 0; n < N; ++n) {
        const booki_tensor* t = &seq->items[n];
        if (t->rank != first->rank) return -5;
        for (int i = 0; i < first->rank; ++i) {
            if (i == axis) continue;
            if (t->shape[i] != first->shape[i]) return -6;
        }
        total_axis += t->shape[axis];
    }
    shape[axis] = total_axis;

    if (out->data == NULL) {
        *out = booki_tensor_arena(arena, BOOKI_DTYPE_F16, first->rank, shape);
        if (!out->data) return -7;
    } else {
        if (out->rank != first->rank) return -8;
        for (int i = 0; i < first->rank; ++i)
            if (out->shape[i] != shape[i]) return -9;
    }

    /* Copy each piece into its slot. */
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i)            outer *= shape[i];
    for (int i = axis + 1; i < first->rank; ++i) inner *= shape[i];

    uint16_t* op = (uint16_t*)out->data;
    int64_t out_axis_offset = 0;
    for (int n = 0; n < N; ++n) {
        const booki_tensor* t = &seq->items[n];
        int64_t in_axis = t->shape[axis];
        const uint16_t* tp = (const uint16_t*)t->data;
        for (int64_t o = 0; o < outer; ++o) {
            uint16_t* dst = op + (o * total_axis + out_axis_offset) * inner;
            const uint16_t* src = tp + o * in_axis * inner;
            memcpy(dst, src, in_axis * inner * sizeof(uint16_t));
        }
        out_axis_offset += in_axis;
    }
    return 0;
}

/* Multi-head attention sanity test.
 *
 *  With H == 1 the output must match booki_attention_f16 exactly (modulo
 *  the extra arena scratch the MHA path uses for per-head copies). Beyond
 *  that we sanity-check that increasing H gives different (i.e. non-trivial)
 *  outputs and that H>1 still produces finite numbers.
 */

#include "booki_native.h"

#include <math.h>
#include <stdio.h>

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } \
} while (0)

static booki_tensor mk2(booki_arena* a, int64_t M, int64_t N) {
    int64_t s[2] = { M, N }; return booki_tensor_arena(a, BOOKI_DTYPE_F16, 2, s);
}
static void fill(booki_tensor* t, const float* src) {
    uint16_t* p = (uint16_t*)t->data;
    int64_t n = booki_tensor_elements(t);
    for (int64_t i = 0; i < n; ++i) p[i] = booki_f32_to_f16(src[i]);
}
static float read(const booki_tensor* t, int64_t i) {
    return booki_f16_to_f32(((const uint16_t*)t->data)[i]);
}

int main(void) {
    booki_arena* a = booki_arena_create(2 << 20);
    CHECK(a, "arena");

    /* M=N=4, embedding=8, num_heads=2 → head dim 4. */
    float qv[32], kv[32], vv[32];
    for (int i = 0; i < 32; ++i) {
        qv[i] = (float)((i * 7  + 3) % 11) * 0.1f - 0.5f;
        kv[i] = (float)((i * 5  + 1) % 13) * 0.1f - 0.6f;
        vv[i] = (float)((i * 11 + 2) % 17) * 0.1f - 0.7f;
    }

    booki_tensor Q  = mk2(a, 4, 8); fill(&Q, qv);
    booki_tensor K  = mk2(a, 4, 8); fill(&K, kv);
    booki_tensor V  = mk2(a, 4, 8); fill(&V, vv);
    booki_tensor Y1 = mk2(a, 4, 8);
    booki_tensor Y2 = mk2(a, 4, 8);

    /* H=1 path should equal single-head attention. */
    booki_arena* scratch = booki_arena_create(1 << 20);
    CHECK(booki_attention_f16(&Q, &K, &V, scratch, &Y1) == 0, "single-head");
    booki_arena_destroy(scratch);

    booki_arena* scratch2 = booki_arena_create(1 << 20);
    CHECK(booki_multihead_attention_f16(&Q, &K, &V, /*H=*/1, scratch2, &Y2) == 0, "mha H=1");
    booki_arena_destroy(scratch2);

    for (int i = 0; i < 32; ++i) {
        float a1 = read(&Y1, i), a2 = read(&Y2, i);
        CHECK(fabsf(a1 - a2) < 5e-3f, "H=1 mismatch at %d: %.4f vs %.4f", i, a1, a2);
    }

    /* H=2 should give a different (non-trivial) result + all finite. */
    booki_arena* scratch3 = booki_arena_create(1 << 20);
    booki_tensor Y3 = mk2(a, 4, 8);
    CHECK(booki_multihead_attention_f16(&Q, &K, &V, /*H=*/2, scratch3, &Y3) == 0, "mha H=2");
    booki_arena_destroy(scratch3);

    int finite = 1, differs = 0;
    for (int i = 0; i < 32; ++i) {
        float a3 = read(&Y3, i);
        if (!isfinite(a3)) finite = 0;
        if (fabsf(a3 - read(&Y1, i)) > 5e-3f) differs = 1;
    }
    CHECK(finite, "H=2 output had non-finite values");
    CHECK(differs, "H=2 produced identical output to H=1 — likely a bug");

    booki_arena_destroy(a);
    printf("mha ok\n");
    return 0;
}

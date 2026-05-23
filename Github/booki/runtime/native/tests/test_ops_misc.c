/* Tests for the final batch of Kokoro-port kernels: topk, and, random. */

#include "booki_native.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } \
} while (0)

static booki_tensor mk_f16(booki_arena* a, int rank, const int64_t* shape) {
    return booki_tensor_arena(a, BOOKI_DTYPE_F16, rank, shape);
}
static booki_tensor mk_i64(booki_arena* a, int rank, const int64_t* shape) {
    return booki_tensor_arena(a, BOOKI_DTYPE_I64, rank, shape);
}
static booki_tensor mk_i8(booki_arena* a, int rank, const int64_t* shape) {
    return booki_tensor_arena(a, BOOKI_DTYPE_I8, rank, shape);
}

static int test_topk(booki_arena* a) {
    /* Row 0: [3, 1, 4, 1, 5, 9, 2, 6]. Top-3 = (9 @ 5), (6 @ 7), (5 @ 4).
     * Row 1: [10, 20, 5, 7, 3, 8, 1, 0]. Top-3 = (20 @ 1), (10 @ 0), (8 @ 5). */
    float xv[16] = {
        3, 1, 4, 1, 5, 9, 2, 6,
        10, 20, 5, 7, 3, 8, 1, 0,
    };
    int64_t sx[2] = { 2, 8 };
    int64_t sy[2] = { 2, 3 };
    booki_tensor X = mk_f16(a, 2, sx);
    uint16_t* xp = (uint16_t*)X.data;
    for (int i = 0; i < 16; ++i) xp[i] = booki_f32_to_f16(xv[i]);

    booki_tensor V = mk_f16(a, 2, sy);
    booki_tensor I = mk_i64(a, 2, sy);

    int rc = booki_topk_f16(&X, /*k=*/3, &V, &I);
    CHECK(rc == 0, "topk rc=%d", rc);

    float want_v[6] = { 9, 6, 5, 20, 10, 8 };
    int64_t want_i[6] = { 5, 7, 4, 1, 0, 5 };
    uint16_t* vp = (uint16_t*)V.data;
    int64_t*  ip = (int64_t*)I.data;
    for (int i = 0; i < 6; ++i) {
        CHECK(fabsf(booki_f16_to_f32(vp[i]) - want_v[i]) < 1e-2f,
              "v[%d]=%.2f want %.2f", i, booki_f16_to_f32(vp[i]), want_v[i]);
        CHECK(ip[i] == want_i[i],
              "i[%d]=%lld want %lld", i, (long long)ip[i], (long long)want_i[i]);
    }
    return 0;
}

static int test_and(booki_arena* a) {
    int64_t s[1] = { 6 };
    booki_tensor A = mk_i8(a, 1, s);
    booki_tensor B = mk_i8(a, 1, s);
    booki_tensor O = mk_i8(a, 1, s);
    int8_t av[6] = { 0, 1, 1, 0, 1, 1 };
    int8_t bv[6] = { 0, 1, 0, 1, 1, 0 };
    memcpy(A.data, av, 6); memcpy(B.data, bv, 6);
    CHECK(booki_and_i8(&A, &B, &O) == 0, "and rc");
    int8_t want[6] = { 0, 1, 0, 0, 1, 0 };
    int8_t* op = (int8_t*)O.data;
    for (int i = 0; i < 6; ++i)
        CHECK(op[i] == want[i], "and[%d]=%d want %d", i, op[i], want[i]);
    return 0;
}

static int test_random(booki_arena* a) {
    /* Determinism: same seed → same bits.
     * Distribution sanity: 1024 samples of N(0,1) should have mean ≈ 0,
     * stddev ≈ 1 within wide tolerance. */
    int64_t s[1] = { 1024 };
    booki_tensor X = mk_f16(a, 1, s);
    booki_tensor Y = mk_f16(a, 1, s);

    CHECK(booki_random_normal_f16(&X, 0.0f, 1.0f, 42) == 0, "normal rc");
    CHECK(booki_random_normal_f16(&Y, 0.0f, 1.0f, 42) == 0, "normal rc 2");
    /* Deterministic: identical for same seed. */
    int64_t n = booki_tensor_elements(&X);
    uint16_t* xp = (uint16_t*)X.data;
    uint16_t* yp = (uint16_t*)Y.data;
    for (int64_t i = 0; i < n; ++i)
        CHECK(xp[i] == yp[i], "seed not deterministic at %lld", (long long)i);

    /* Stats sanity. */
    double sum = 0, sumsq = 0;
    for (int64_t i = 0; i < n; ++i) {
        float v = booki_f16_to_f32(xp[i]);
        sum += v; sumsq += v * v;
    }
    double mean = sum / n;
    double var  = sumsq / n - mean * mean;
    CHECK(fabs(mean)         < 0.1, "normal mean %f", mean);
    CHECK(fabs(sqrt(var) - 1.0) < 0.1, "normal stddev %f", sqrt(var));

    /* Uniform [-2, 2]. */
    CHECK(booki_random_uniform_f16(&X, -2.0f, 2.0f, 13) == 0, "uniform rc");
    float lo = 1000, hi = -1000;
    for (int64_t i = 0; i < n; ++i) {
        float v = booki_f16_to_f32(xp[i]);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    CHECK(lo >= -2.05f && lo <= -1.5f, "uniform low %f", lo);
    CHECK(hi <=  2.05f && hi >=  1.5f, "uniform high %f", hi);
    return 0;
}

int main(void) {
    booki_arena* a = booki_arena_create(1 << 20);
    int rc = 0;
    rc |= test_topk(a);   booki_arena_reset(a);
    rc |= test_and(a);    booki_arena_reset(a);
    rc |= test_random(a);
    booki_arena_destroy(a);
    if (rc == 0) printf("ops_misc ok\n");
    return rc;
}

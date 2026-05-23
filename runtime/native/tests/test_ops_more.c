/* Tests for the second batch of Kokoro-port ops: exp, atan, cumsum, pad1d, scatter_nd. */

#include "booki_native.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } \
} while (0)

static booki_tensor mk2(booki_arena* a, int64_t M, int64_t N) {
    int64_t s[2] = { M, N }; return booki_tensor_arena(a, BOOKI_DTYPE_F16, 2, s);
}
static booki_tensor mk2_i64(booki_arena* a, int64_t M, int64_t N) {
    int64_t s[2] = { M, N }; return booki_tensor_arena(a, BOOKI_DTYPE_I64, 2, s);
}
static void fill(booki_tensor* t, const float* src) {
    uint16_t* p = (uint16_t*)t->data;
    int64_t n = booki_tensor_elements(t);
    for (int64_t i = 0; i < n; ++i) p[i] = booki_f32_to_f16(src[i]);
}
static float read(const booki_tensor* t, int64_t i) {
    return booki_f16_to_f32(((const uint16_t*)t->data)[i]);
}

static int test_exp(booki_arena* a) {
    float xv[5] = { 0.0f, 1.0f, -1.0f, 2.5f, -3.0f };
    booki_tensor X = mk2(a, 1, 5); fill(&X, xv);
    booki_tensor Y = mk2(a, 1, 5);
    CHECK(booki_exp_f16(&X, &Y) == 0, "exp rc");
    for (int i = 0; i < 5; ++i) {
        float want = expf(xv[i]);
        CHECK(fabsf(read(&Y, i) - want) < 2e-2f * fmaxf(1.0f, want),
              "exp[%d]=%.4f want %.4f", i, read(&Y, i), want);
    }
    return 0;
}

static int test_atan(booki_arena* a) {
    float xv[7] = { 0.0f, 0.5f, 1.0f, 2.0f, -1.0f, -3.0f, 10.0f };
    booki_tensor X = mk2(a, 1, 7); fill(&X, xv);
    booki_tensor Y = mk2(a, 1, 7);
    CHECK(booki_atan_f16(&X, &Y) == 0, "atan rc");
    for (int i = 0; i < 7; ++i) {
        float want = atanf(xv[i]);
        CHECK(fabsf(read(&Y, i) - want) < 5e-3f,
              "atan[%d]=%.4f want %.4f", i, read(&Y, i), want);
    }
    return 0;
}

static int test_cumsum(booki_arena* a) {
    float xv[8] = { 1, 2, 3, 4,  10, 20, 30, 40 };  /* 2 rows */
    booki_tensor X = mk2(a, 2, 4); fill(&X, xv);
    booki_tensor Y = mk2(a, 2, 4);
    CHECK(booki_cumsum_f16(&X, &Y) == 0, "cumsum rc");
    float want[8] = { 1, 3, 6, 10,  10, 30, 60, 100 };
    for (int i = 0; i < 8; ++i)
        CHECK(fabsf(read(&Y, i) - want[i]) < 1e-2f,
              "cumsum[%d]=%.4f want %.2f", i, read(&Y, i), want[i]);
    return 0;
}

static int test_pad(booki_arena* a) {
    float xv[3] = { 1, 2, 3 };
    booki_tensor X = mk2(a, 1, 3); fill(&X, xv);
    booki_tensor Y = mk2(a, 1, 7);
    CHECK(booki_pad1d_f16(&X, /*before=*/2, /*after=*/2, /*value=*/-1.0f, &Y) == 0,
          "pad rc");
    float want[7] = { -1, -1, 1, 2, 3, -1, -1 };
    for (int i = 0; i < 7; ++i)
        CHECK(fabsf(read(&Y, i) - want[i]) < 1e-3f,
              "pad[%d]=%.4f want %.2f", i, read(&Y, i), want[i]);
    return 0;
}

static int test_scatter_nd(booki_arena* a) {
    /* data : [3, 2]  = [[1,2],[3,4],[5,6]]
     * idx  : [2, 1]  = [[0],[2]]
     * upd  : [2, 2]  = [[10,20],[50,60]]
     * out  : [3, 2]  = [[10,20],[3,4],[50,60]]
     */
    float dv[6] = { 1, 2, 3, 4, 5, 6 };
    int64_t ip[2] = { 0, 2 };
    float uv[4] = { 10, 20, 50, 60 };

    booki_tensor D = mk2(a, 3, 2); fill(&D, dv);
    booki_tensor I = mk2_i64(a, 2, 1);
    memcpy(I.data, ip, sizeof(ip));
    booki_tensor U = mk2(a, 2, 2); fill(&U, uv);
    booki_tensor O = mk2(a, 3, 2);

    CHECK(booki_scatter_nd_f16(&D, &I, &U, &O) == 0, "scatter rc");
    float want[6] = { 10, 20, 3, 4, 50, 60 };
    for (int i = 0; i < 6; ++i)
        CHECK(fabsf(read(&O, i) - want[i]) < 1e-2f,
              "scatter[%d]=%.4f want %.2f", i, read(&O, i), want[i]);
    return 0;
}

int main(void) {
    booki_arena* a = booki_arena_create(1 << 20);
    int rc = 0;
    rc |= test_exp(a);        booki_arena_reset(a);
    rc |= test_atan(a);       booki_arena_reset(a);
    rc |= test_cumsum(a);     booki_arena_reset(a);
    rc |= test_pad(a);        booki_arena_reset(a);
    rc |= test_scatter_nd(a);
    booki_arena_destroy(a);
    if (rc == 0) printf("ops_more ok\n");
    return rc;
}

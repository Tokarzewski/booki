/* Tests for the Kokoro-port additions: sub, leaky_relu, sin/cos, instance_norm. */

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
static booki_tensor mk3(booki_arena* a, int64_t a0, int64_t a1, int64_t a2) {
    int64_t s[3] = { a0, a1, a2 }; return booki_tensor_arena(a, BOOKI_DTYPE_F16, 3, s);
}
static booki_tensor mk1(booki_arena* a, int64_t n) {
    int64_t s[1] = { n }; return booki_tensor_arena(a, BOOKI_DTYPE_F16, 1, s);
}
static void fill(booki_tensor* t, const float* src) {
    uint16_t* p = (uint16_t*)t->data;
    int64_t n = booki_tensor_elements(t);
    for (int64_t i = 0; i < n; ++i) p[i] = booki_f32_to_f16(src[i]);
}
static float read(const booki_tensor* t, int64_t i) {
    return booki_f16_to_f32(((const uint16_t*)t->data)[i]);
}

static int test_sub(booki_arena* a) {
    float av[4] = { 5, 3, 7, 2 };
    float bv[4] = { 1, 4, 0, 9 };
    booki_tensor A = mk2(a, 1, 4); fill(&A, av);
    booki_tensor B = mk2(a, 1, 4); fill(&B, bv);
    booki_tensor S = mk2(a, 1, 4);
    CHECK(booki_sub_f16(&A, &B, &S) == 0, "sub rc");
    float want[4] = { 4, -1, 7, -7 };
    for (int i = 0; i < 4; ++i)
        CHECK(fabsf(read(&S, i) - want[i]) < 1e-2f, "sub[%d]=%.3f want %.3f",
              i, read(&S, i), want[i]);
    return 0;
}

static int test_leaky_relu(booki_arena* a) {
    float xv[6] = { -2.0f, -0.5f, 0.0f, 0.5f, 2.0f, -5.0f };
    booki_tensor X = mk2(a, 1, 6); fill(&X, xv);
    booki_tensor Y = mk2(a, 1, 6);
    CHECK(booki_leaky_relu_f16(&X, 0.1f, &Y) == 0, "leaky rc");
    for (int i = 0; i < 6; ++i) {
        float want = xv[i] > 0 ? xv[i] : 0.1f * xv[i];
        CHECK(fabsf(read(&Y, i) - want) < 5e-3f, "leaky[%d]=%.4f want %.4f",
              i, read(&Y, i), want);
    }
    return 0;
}

static int test_sin_cos(booki_arena* a) {
    float xs[8] = { 0.0f, 0.5f, 1.0f, 3.14159f, -1.5708f, 6.283185f, 10.0f, -10.0f };
    booki_tensor X = mk2(a, 1, 8); fill(&X, xs);
    booki_tensor S = mk2(a, 1, 8);
    booki_tensor C = mk2(a, 1, 8);
    CHECK(booki_sin_f16(&X, &S) == 0, "sin rc");
    CHECK(booki_cos_f16(&X, &C) == 0, "cos rc");
    for (int i = 0; i < 8; ++i) {
        float ws = sinf(xs[i]);
        float wc = cosf(xs[i]);
        CHECK(fabsf(read(&S, i) - ws) < 2e-2f,
              "sin[%d]=%.4f want %.4f", i, read(&S, i), ws);
        CHECK(fabsf(read(&C, i) - wc) < 2e-2f,
              "cos[%d]=%.4f want %.4f", i, read(&C, i), wc);
    }
    return 0;
}

static int test_instance_norm(booki_arena* a) {
    /* [N=1, C=2, T=4] — each (n,c) row normalised independently. */
    float xv[8] = {
        1, 2, 3, 4,        /* row 0: mean=2.5, var=1.25 */
        10, 20, 30, 40,    /* row 1: mean=25,  var=125  */
    };
    float scale[2] = { 1.0f, 2.0f };
    float bias[2]  = { 0.0f, 1.0f };

    booki_tensor X = mk3(a, 1, 2, 4); fill(&X, xv);
    booki_tensor S = mk1(a, 2);       fill(&S, scale);
    booki_tensor B = mk1(a, 2);       fill(&B, bias);
    booki_tensor Y = mk3(a, 1, 2, 4);

    CHECK(booki_instance_norm_f16(&X, &S, &B, 1e-6f, &Y) == 0, "instance_norm rc");

    /* Oracle in fp32. */
    for (int c = 0; c < 2; ++c) {
        float sum = 0, sumsq = 0;
        for (int t = 0; t < 4; ++t) {
            float v = xv[c * 4 + t];
            sum += v; sumsq += v * v;
        }
        float m = sum / 4.0f, v = sumsq / 4.0f - m * m;
        float inv = 1.0f / sqrtf(v + 1e-6f);
        for (int t = 0; t < 4; ++t) {
            float want = (xv[c * 4 + t] - m) * inv * scale[c] + bias[c];
            float got = read(&Y, c * 4 + t);
            CHECK(fabsf(got - want) < 2e-2f,
                  "inorm[%d,%d]=%.4f want %.4f", c, t, got, want);
        }
    }
    return 0;
}

int main(void) {
    booki_arena* a = booki_arena_create(1 << 20);
    int rc = 0;
    rc |= test_sub(a);          booki_arena_reset(a);
    rc |= test_leaky_relu(a);   booki_arena_reset(a);
    rc |= test_sin_cos(a);      booki_arena_reset(a);
    rc |= test_instance_norm(a);
    booki_arena_destroy(a);
    if (rc == 0) printf("ops_extra ok\n");
    return rc;
}

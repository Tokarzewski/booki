/* LSTM tests:
 *  - With all-zero weights + bias the output is identically zero.
 *  - With a known small configuration, output matches a hand-computed
 *    fp32 reference produced by the same recurrence in plain C.
 */

#include "booki_native.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } \
} while (0)

static booki_tensor mk2(booki_arena* a, int64_t M, int64_t N) {
    int64_t s[2] = { M, N }; return booki_tensor_arena(a, BOOKI_DTYPE_F16, 2, s);
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

static int test_zero_weights(booki_arena* a) {
    int64_t T = 3, D = 4, H = 2;
    float xv[12] = { 1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12 };
    float wv[32] = {0};   /* 4H × D = 8 × 4 */
    float rv[16] = {0};   /* 4H × H = 8 × 2 */

    booki_tensor X = mk2(a, T, D); fill(&X, xv);
    booki_tensor W = mk2(a, 4*H, D); fill(&W, wv);
    booki_tensor R = mk2(a, 4*H, H); fill(&R, rv);
    booki_tensor Y = mk2(a, T, H);

    booki_arena* scratch = booki_arena_create(1 << 16);
    int rc = booki_lstm_f16(&X, &W, &R, NULL, NULL, NULL, scratch, &Y);
    CHECK(rc == 0, "lstm zero rc=%d", rc);
    booki_arena_destroy(scratch);

    /* All zero weights → all-zero gate args → tanh(0) = 0 → h stays 0. */
    int64_t n = T * H;
    for (int64_t i = 0; i < n; ++i)
        CHECK(fabsf(read(&Y, i)) < 1e-2f, "y[%lld]=%.4f (should be 0)",
              (long long)i, read(&Y, i));
    return 0;
}

static float sig(float x) { return 1.0f / (1.0f + expf(-x)); }

/* Plain-C reference matching booki_lstm_f16's semantics exactly. */
static void lstm_ref(const float* x, const float* W, const float* R,
                     const float* b, float* y,
                     int T, int D, int H) {
    float* h = (float*)calloc(H, sizeof(float));
    float* c = (float*)calloc(H, sizeof(float));
    float* g = (float*)malloc(4 * H * sizeof(float));
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < 4 * H; ++i) {
            g[i] = b ? b[i] + b[4 * H + i] : 0.0f;
            for (int k = 0; k < D; ++k) g[i] += W[i * D + k] * x[t * D + k];
            for (int k = 0; k < H; ++k) g[i] += R[i * H + k] * h[k];
        }
        for (int i = 0; i < H; ++i) {
            float it = sig(g[i]);
            float ot = sig(g[H + i]);
            float ft = sig(g[2 * H + i]);
            float ct = tanhf(g[3 * H + i]);
            c[i] = ft * c[i] + it * ct;
            h[i] = ot * tanhf(c[i]);
            y[t * H + i] = h[i];
        }
    }
    free(h); free(c); free(g);
}

static int test_known(booki_arena* a) {
    int64_t T = 4, D = 3, H = 2;

    /* Small, deterministic values; fp16 precision should be ample. */
    float xv[12] = { 0.1f, 0.2f, 0.3f,
                     -0.1f, 0.4f, 0.0f,
                     0.5f, -0.2f, 0.1f,
                     0.0f, 0.0f, 0.5f };
    float wv[24];  /* 4H × D = 8 × 3 */
    float rv[16];  /* 4H × H = 8 × 2 */
    float bv[16];  /* 8H */
    for (int i = 0; i < 24; ++i) wv[i] = ((i * 7 + 3) % 11) * 0.05f - 0.25f;
    for (int i = 0; i < 16; ++i) rv[i] = ((i * 5 + 2) % 13) * 0.05f - 0.3f;
    for (int i = 0; i < 16; ++i) bv[i] = ((i * 3 + 1) % 7)  * 0.05f - 0.15f;

    booki_tensor X = mk2(a, T, D);   fill(&X, xv);
    booki_tensor W = mk2(a, 4*H, D); fill(&W, wv);
    booki_tensor R = mk2(a, 4*H, H); fill(&R, rv);
    booki_tensor B = mk1(a, 8*H);    fill(&B, bv);
    booki_tensor Y = mk2(a, T, H);

    booki_arena* scratch = booki_arena_create(1 << 16);
    int rc = booki_lstm_f16(&X, &W, &R, &B, NULL, NULL, scratch, &Y);
    CHECK(rc == 0, "lstm rc=%d", rc);
    booki_arena_destroy(scratch);

    float y_ref[8];
    lstm_ref(xv, wv, rv, bv, y_ref, (int)T, (int)D, (int)H);

    for (int i = 0; i < T * H; ++i) {
        float diff = fabsf(read(&Y, i) - y_ref[i]);
        CHECK(diff < 5e-3f, "y[%d]=%.4f ref=%.4f diff=%.4f", i, read(&Y, i), y_ref[i], diff);
    }
    return 0;
}

int main(void) {
    booki_arena* a = booki_arena_create(1 << 20);
    int rc = 0;
    rc |= test_zero_weights(a);  booki_arena_reset(a);
    rc |= test_known(a);
    booki_arena_destroy(a);
    if (rc == 0) printf("lstm ok\n");
    return rc;
}

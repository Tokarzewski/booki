/* Conv1d conformance tests:
 *  - identity convolution (K=1, weights=identity) reproduces input
 *  - moving-average kernel (K=3, weights=1/3) matches a hand-computed reference
 *  - depthwise (groups==in_channels) flag works
 *  - shape rejection
 */

#include "booki_native.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } \
} while (0)

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

static int identity_test(booki_arena* a) {
    /* Cin=Cout=2, K=1, identity weight. Out should equal input exactly. */
    float xv[8] = { 1, 2, 3, 4,   5, 6, 7, 8 };       /* [B=1, Cin=2, L=4] */
    float wv[4] = { 1, 0,   0, 1 };                    /* [Cout=2, Cin=2, K=1] */

    booki_tensor x = mk3(a, 1, 2, 4); fill(&x, xv);
    booki_tensor w = mk3(a, 2, 2, 1); fill(&w, wv);
    booki_tensor y = mk3(a, 1, 2, 4);

    int rc = booki_conv1d_f16(&x, &w, NULL, 1, 0, 1, 1, &y);
    CHECK(rc == 0, "conv1d returned %d", rc);
    for (int i = 0; i < 8; ++i)
        CHECK(fabsf(read(&y, i) - xv[i]) < 1e-2f, "y[%d]=%.4f vs %.2f", i, read(&y, i), xv[i]);
    return 0;
}

static int moving_avg_test(booki_arena* a) {
    /* Single channel, K=3, weights=1/3 each, padding=1.
     * Input: [1, 2, 3, 4, 5]. Expected with reflect-equivalent zero-pad:
     *   y[0] = (0 + 1 + 2) / 3
     *   y[1] = (1 + 2 + 3) / 3
     *   y[2] = (2 + 3 + 4) / 3
     *   y[3] = (3 + 4 + 5) / 3
     *   y[4] = (4 + 5 + 0) / 3
     */
    float xv[5] = { 1, 2, 3, 4, 5 };
    float wv[3] = { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f };

    booki_tensor x = mk3(a, 1, 1, 5); fill(&x, xv);
    booki_tensor w = mk3(a, 1, 1, 3); fill(&w, wv);
    booki_tensor y = mk3(a, 1, 1, 5);

    int rc = booki_conv1d_f16(&x, &w, NULL, 1, 1, 1, 1, &y);
    CHECK(rc == 0, "conv1d returned %d", rc);

    float want[5] = { (0 + 1 + 2) / 3.0f, (1 + 2 + 3) / 3.0f,
                      (2 + 3 + 4) / 3.0f, (3 + 4 + 5) / 3.0f,
                      (4 + 5 + 0) / 3.0f };
    for (int i = 0; i < 5; ++i)
        CHECK(fabsf(read(&y, i) - want[i]) < 5e-3f,
              "y[%d]=%.4f want %.4f", i, read(&y, i), want[i]);
    return 0;
}

static int bias_test(booki_arena* a) {
    float xv[4] = { 1, 2, 3, 4 };
    float wv[1] = { 2.0f };
    float bv[1] = { 10.0f };

    booki_tensor x = mk3(a, 1, 1, 4); fill(&x, xv);
    booki_tensor w = mk3(a, 1, 1, 1); fill(&w, wv);
    booki_tensor b = mk1(a, 1);       fill(&b, bv);
    booki_tensor y = mk3(a, 1, 1, 4);

    int rc = booki_conv1d_f16(&x, &w, &b, 1, 0, 1, 1, &y);
    CHECK(rc == 0, "conv1d returned %d", rc);
    for (int i = 0; i < 4; ++i)
        CHECK(fabsf(read(&y, i) - (xv[i] * 2 + 10)) < 5e-3f, "bias[%d]", i);
    return 0;
}

static int depthwise_test(booki_arena* a) {
    /* Groups = Cin = Cout = 2, K=1 weights=[3, 5] per channel.
     * Expected: channel-0 output = input * 3; channel-1 output = input * 5. */
    float xv[6] = { 1, 2, 3,   4, 5, 6 };          /* [B=1, C=2, L=3] */
    float wv[2] = { 3.0f, 5.0f };                  /* [Cout=2, Cin/g=1, K=1] */

    booki_tensor x = mk3(a, 1, 2, 3); fill(&x, xv);
    booki_tensor w = mk3(a, 2, 1, 1); fill(&w, wv);
    booki_tensor y = mk3(a, 1, 2, 3);

    int rc = booki_conv1d_f16(&x, &w, NULL, 1, 0, 1, /*groups=*/2, &y);
    CHECK(rc == 0, "conv1d depthwise returned %d", rc);

    float want[6] = { 3, 6, 9,   20, 25, 30 };
    for (int i = 0; i < 6; ++i)
        CHECK(fabsf(read(&y, i) - want[i]) < 5e-3f,
              "y[%d]=%.4f want %.2f", i, read(&y, i), want[i]);
    return 0;
}

static int shape_reject_test(booki_arena* a) {
    booki_tensor x = mk3(a, 1, 2, 4);
    booki_tensor w = mk3(a, 2, 2, 1);
    booki_tensor wrong_y = mk3(a, 1, 2, 5);   /* wrong out length */
    CHECK(booki_conv1d_f16(&x, &w, NULL, 1, 0, 1, 1, &wrong_y) != 0,
          "should reject wrong out length");
    return 0;
}

int main(void) {
    booki_arena* a = booki_arena_create(1 << 20);
    int rc = 0;
    rc |= identity_test(a);    booki_arena_reset(a);
    rc |= moving_avg_test(a);  booki_arena_reset(a);
    rc |= bias_test(a);        booki_arena_reset(a);
    rc |= depthwise_test(a);   booki_arena_reset(a);
    rc |= shape_reject_test(a);
    booki_arena_destroy(a);
    if (rc == 0) printf("conv1d ok\n");
    return rc;
}

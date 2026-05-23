/* Tests for ConvTranspose1d + Resize1d. */

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

static int test_convT_identity_upsample(booki_arena* a) {
    /* In=1, Out=1, K=2, stride=2, padding=0, dilation=1, groups=1.
     * Weight [1,1,2] = [1, 1]. Input [1,1,3] = [1, 2, 3].
     * L_out = (3-1)*2 - 0 + 1*(2-1) + 0 + 1 = 6.
     * Each input scatters into 2 output positions starting at n*stride.
     */
    float xv[3] = { 1, 2, 3 };
    float wv[2] = { 1, 1 };
    booki_tensor X = mk3(a, 1, 1, 3); fill(&X, xv);
    booki_tensor W = mk3(a, 1, 1, 2); fill(&W, wv);
    booki_tensor Y = mk3(a, 1, 1, 6);
    int rc = booki_conv_transpose1d_f16(&X, &W, NULL, /*stride=*/2,
                                        /*padding=*/0, /*output_padding=*/0,
                                        /*dilation=*/1, /*groups=*/1, &Y);
    CHECK(rc == 0, "convT rc=%d", rc);

    float want[6] = { 1, 1, 2, 2, 3, 3 };
    for (int i = 0; i < 6; ++i)
        CHECK(fabsf(read(&Y, i) - want[i]) < 5e-3f,
              "y[%d]=%.4f want %.2f", i, read(&Y, i), want[i]);
    return 0;
}

static int test_convT_with_bias(booki_arena* a) {
    float xv[4] = { 1, 2, 3, 4 };
    float wv[3] = { 0.5f, 0.5f, 0.5f };
    float bv[1] = { 10.0f };
    booki_tensor X = mk3(a, 1, 1, 4); fill(&X, xv);
    booki_tensor W = mk3(a, 1, 1, 3); fill(&W, wv);
    booki_tensor B = mk1(a, 1);       fill(&B, bv);
    /* L_out = (4-1)*1 - 0 + 2 + 0 + 1 = 6 */
    booki_tensor Y = mk3(a, 1, 1, 6);
    int rc = booki_conv_transpose1d_f16(&X, &W, &B, 1, 0, 0, 1, 1, &Y);
    CHECK(rc == 0, "convT bias rc=%d", rc);
    /* Hand-compute: each output position sums the input contributions plus 10. */
    float want[6] = {
        10 + 1 * 0.5f,
        10 + 1 * 0.5f + 2 * 0.5f,
        10 + 1 * 0.5f + 2 * 0.5f + 3 * 0.5f,
        10 + 2 * 0.5f + 3 * 0.5f + 4 * 0.5f,
        10 + 3 * 0.5f + 4 * 0.5f,
        10 + 4 * 0.5f,
    };
    for (int i = 0; i < 6; ++i)
        CHECK(fabsf(read(&Y, i) - want[i]) < 2e-2f,
              "convT bias[%d]=%.4f want %.4f", i, read(&Y, i), want[i]);
    return 0;
}

static int test_resize_nearest(booki_arena* a) {
    /* [1,1,4] → [1,1,8] nearest upsample of [1,2,3,4]. */
    float xv[4] = { 1, 2, 3, 4 };
    booki_tensor X = mk3(a, 1, 1, 4); fill(&X, xv);
    booki_tensor Y = mk3(a, 1, 1, 8);
    int rc = booki_resize1d_f16(&X, 8, BOOKI_RESIZE_NEAREST, &Y);
    CHECK(rc == 0, "resize nearest rc=%d", rc);
    /* half-pixel + floor: src(i) = (i+0.5)*0.5 - 0.5 ; nearest = round(src). */
    for (int i = 0; i < 8; ++i) {
        float src = ((float)i + 0.5f) * 0.5f - 0.5f;
        int s = (int)floorf(src + 0.5f);
        if (s < 0) s = 0;
        if (s > 3) s = 3;
        CHECK(fabsf(read(&Y, i) - xv[s]) < 1e-3f,
              "resize_n[%d]=%.2f src=%.2f want %.2f", i, read(&Y, i), src, xv[s]);
    }
    return 0;
}

static int test_resize_linear(booki_arena* a) {
    /* Simple sanity: upsample [10, 20] by 2× → endpoints preserved approximately. */
    float xv[2] = { 10.0f, 20.0f };
    booki_tensor X = mk3(a, 1, 1, 2); fill(&X, xv);
    booki_tensor Y = mk3(a, 1, 1, 4);
    int rc = booki_resize1d_f16(&X, 4, BOOKI_RESIZE_LINEAR, &Y);
    CHECK(rc == 0, "resize linear rc=%d", rc);
    /* With half-pixel alignment, output[0] and output[3] should land near
     * the original ends (not exactly because of half-pixel offset). */
    CHECK(read(&Y, 0) >= 9.0f && read(&Y, 0) <= 14.0f, "y[0]=%.4f", read(&Y, 0));
    CHECK(read(&Y, 3) >= 16.0f && read(&Y, 3) <= 21.0f, "y[3]=%.4f", read(&Y, 3));
    /* Monotonic increase. */
    for (int i = 1; i < 4; ++i)
        CHECK(read(&Y, i) >= read(&Y, i - 1) - 0.05f, "monotonic at %d", i);
    return 0;
}

int main(void) {
    booki_arena* a = booki_arena_create(1 << 20);
    int rc = 0;
    rc |= test_convT_identity_upsample(a); booki_arena_reset(a);
    rc |= test_convT_with_bias(a);          booki_arena_reset(a);
    rc |= test_resize_nearest(a);           booki_arena_reset(a);
    rc |= test_resize_linear(a);
    booki_arena_destroy(a);
    if (rc == 0) printf("conv_transpose1d ok\n");
    return rc;
}

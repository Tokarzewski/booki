/* Conformance tests for booki_matmul_f16:
 *  - empty / shape-mismatch errors
 *  - identity matrix
 *  - scalar vs NEON agreement (when NEON is available)
 *  - known-answer for a 3x4 * 4x2 case computed in fp32
 */

#include "booki_native.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The runtime stores fp16 as uint16_t everywhere; conversions go through
 * the library so tests and impl can never disagree about the encoding. */
typedef uint16_t fp16;
static inline float f16_to_f32_(fp16 h) { return booki_f16_to_f32(h); }
static inline fp16  f32_to_f16_(float f) { return booki_f32_to_f16(f); }

#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        return 1; \
    } \
} while (0)

static booki_tensor mk(booki_arena* a, int64_t M, int64_t N) {
    int64_t shape[2] = { M, N };
    return booki_tensor_arena(a, BOOKI_DTYPE_F16, 2, shape);
}

static void fill_from(booki_tensor* t, const float* src, int64_t n) {
    fp16* p = (fp16*)t->data;
    for (int64_t i = 0; i < n; ++i) p[i] = f32_to_f16_(src[i]);
}

static int known_answer_test(booki_arena* a) {
    /* A: 3x4, B: 4x2, C expected: 3x2 */
    float A_src[12] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
    };
    float B_src[8] = {
        1, 0,
        0, 1,
        1, 1,
       -1,-1,
    };
    /* C = A * B
     * row 0: 1+0+3-4=0, 0+2+3-4=1
     * row 1: 5+0+7-8=4, 0+6+7-8=5
     * row 2: 9+0+11-12=8, 0+10+11-12=9
     */
    float C_expected[6] = { 0, 1, 4, 5, 8, 9 };

    booki_tensor A = mk(a, 3, 4); fill_from(&A, A_src, 12);
    booki_tensor B = mk(a, 4, 2); fill_from(&B, B_src, 8);
    booki_tensor C = mk(a, 3, 2); memset(C.data, 0, C.nbytes);

    int rc = booki_matmul_f16(&A, &B, &C);
    CHECK(rc == 0, "matmul returned %d", rc);

    fp16* p = (fp16*)C.data;
    for (int i = 0; i < 6; ++i) {
        float got = f16_to_f32_(p[i]);
        CHECK(fabsf(got - C_expected[i]) < 1e-2f,
              "C[%d] expected %.2f got %.4f", i, C_expected[i], got);
    }
    return 0;
}

static int identity_test(booki_arena* a) {
    /* A * I = A for a square 8x8 fp16 matrix. */
    const int64_t M = 8;
    float buf[64];
    for (int i = 0; i < 64; ++i) buf[i] = (float)i * 0.5f - 7.0f;

    booki_tensor A = mk(a, M, M); fill_from(&A, buf, 64);
    booki_tensor I = mk(a, M, M);
    fp16* ip = (fp16*)I.data;
    memset(ip, 0, I.nbytes);
    for (int i = 0; i < M; ++i) ip[i * M + i] = f32_to_f16_(1.0f);

    booki_tensor C = mk(a, M, M); memset(C.data, 0, C.nbytes);

    int rc = booki_matmul_f16(&A, &I, &C);
    CHECK(rc == 0, "matmul returned %d", rc);

    fp16* a_p = (fp16*)A.data;
    fp16* c_p = (fp16*)C.data;
    for (int i = 0; i < 64; ++i) {
        float ad = f16_to_f32_(a_p[i]);
        float cd = f16_to_f32_(c_p[i]);
        CHECK(fabsf(ad - cd) < 1e-2f, "A*I != A at %d: %f vs %f", i, ad, cd);
    }
    return 0;
}

static int backends_agree_test(booki_arena* a) {
    /* Scalar and NEON should agree to within fp16 precision on a non-trivial
     * shape that exercises the 4x4 tile + tail paths. */
    const int64_t M = 13, K = 19, N = 7;
    int64_t elemsA = M * K, elemsB = K * N;

    float* abuf = (float*)malloc(sizeof(float) * elemsA);
    float* bbuf = (float*)malloc(sizeof(float) * elemsB);
    for (int64_t i = 0; i < elemsA; ++i) abuf[i] = (float)((i * 7 + 3) % 11) * 0.1f - 0.5f;
    for (int64_t i = 0; i < elemsB; ++i) bbuf[i] = (float)((i * 5 + 1) % 13) * 0.1f - 0.6f;

    booki_tensor A = mk(a, M, K); fill_from(&A, abuf, elemsA);
    booki_tensor B = mk(a, K, N); fill_from(&B, bbuf, elemsB);
    booki_tensor C_scalar = mk(a, M, N); memset(C_scalar.data, 0, C_scalar.nbytes);
    booki_tensor C_neon   = mk(a, M, N); memset(C_neon.data,   0, C_neon.nbytes);

    booki_set_backend(BOOKI_BACKEND_SCALAR);
    CHECK(booki_matmul_f16(&A, &B, &C_scalar) == 0, "scalar run failed");

    booki_set_backend(BOOKI_BACKEND_NEON);
    CHECK(booki_matmul_f16(&A, &B, &C_neon) == 0, "neon run failed");
    booki_set_backend(BOOKI_BACKEND_AUTO);

    fp16* sp = (fp16*)C_scalar.data;
    fp16* np = (fp16*)C_neon.data;
    float worst = 0.0f;
    for (int64_t i = 0; i < M * N; ++i) {
        float d = fabsf(f16_to_f32_(sp[i]) - f16_to_f32_(np[i]));
        if (d > worst) worst = d;
    }
    CHECK(worst < 1e-1f, "scalar/neon disagree by %.4f", worst);
    free(abuf); free(bbuf);
    return 0;
}

static int error_paths_test(booki_arena* a) {
    booki_tensor A = mk(a, 2, 3);
    booki_tensor B = mk(a, 4, 5);   /* K mismatch */
    booki_tensor C = mk(a, 2, 5);
    CHECK(booki_matmul_f16(&A, &B, &C) != 0, "should reject K mismatch");
    CHECK(booki_matmul_f16(NULL, &B, &C) != 0, "should reject NULL a");
    return 0;
}

static int sme_smoke(booki_arena* a) {
    /* Smoke-check the SME backend against the scalar oracle when available. */
    if (!booki_has_sme()) {
        printf("(SME not available — skipping)\n");
        return 0;
    }
    const int64_t M = 5, K = 6, N = 7;
    int64_t elemsA = M * K, elemsB = K * N;
    float* abuf = (float*)malloc(sizeof(float) * elemsA);
    float* bbuf = (float*)malloc(sizeof(float) * elemsB);
    for (int64_t i = 0; i < elemsA; ++i) abuf[i] = ((i * 7 + 1) % 9) * 0.1f - 0.4f;
    for (int64_t i = 0; i < elemsB; ++i) bbuf[i] = ((i * 5 + 3) % 11) * 0.1f - 0.5f;

    booki_tensor A = mk(a, M, K); fill_from(&A, abuf, elemsA);
    booki_tensor B = mk(a, K, N); fill_from(&B, bbuf, elemsB);
    booki_tensor Cs = mk(a, M, N); memset(Cs.data, 0, Cs.nbytes);
    booki_tensor Cm = mk(a, M, N); memset(Cm.data, 0, Cm.nbytes);

    booki_set_backend(BOOKI_BACKEND_SCALAR);
    int rc = booki_matmul_f16(&A, &B, &Cs);
    CHECK(rc == 0, "scalar matmul rc=%d", rc);

    booki_set_backend(BOOKI_BACKEND_SME);
    rc = booki_matmul_f16(&A, &B, &Cm);
    CHECK(rc == 0, "sme matmul rc=%d", rc);

    booki_set_backend(BOOKI_BACKEND_AUTO);

    fp16* sp = (fp16*)Cs.data;
    fp16* mp = (fp16*)Cm.data;
    float worst = 0;
    for (int64_t i = 0; i < M * N; ++i) {
        float d = fabsf(f16_to_f32_(sp[i]) - f16_to_f32_(mp[i]));
        if (d > worst) worst = d;
    }
    CHECK(worst < 1e-1f, "sme/scalar disagree by %.4f", worst);

    free(abuf); free(bbuf);
    return 0;
}

int main(void) {
    booki_arena* a = booki_arena_create(1 << 20);
    CHECK(a != NULL, "arena create");

    int rc = 0;
    rc |= known_answer_test(a); booki_arena_reset(a);
    rc |= identity_test(a);     booki_arena_reset(a);
#if defined(__ARM_NEON) || defined(__aarch64__)
    rc |= backends_agree_test(a); booki_arena_reset(a);
#endif
    rc |= error_paths_test(a);  booki_arena_reset(a);
    rc |= sme_smoke(a);

    booki_arena_destroy(a);
    if (rc == 0) printf("matmul ok (backend: %s, sme=%d)\n",
                        booki_backend_describe(booki_backend_active()),
                        booki_has_sme());
    return rc;
}

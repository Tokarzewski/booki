/* fp16 matmul.
 *
 * Two implementations behind a runtime selector:
 *   - scalar reference (also the build target for non-ARM hosts)
 *   - NEON A64 path using 4x4 register tiles and fp32 accumulation
 *
 * Output and inputs are __fp16 (IEEE half). Accumulation is fp32 to keep
 * the error budget reasonable for the transformer-scale matmuls Kokoro
 * uses (up to 1536x1536x4096 in the LM head).
 *
 * Layout: row-major C[M,N] = A[M,K] * B[K,N], all contiguous.
 */

#include "booki_native.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>
  #define HAVE_NEON 1
#else
  #define HAVE_NEON 0
#endif

/* ------------------------------------------------------------------------- */
/* IEEE 754 fp16 conversions. Always uint16_t storage so the same code path  */
/* runs on x86 hosts and ARM devices. On ARM the compiler folds the table-   */
/* style conversions back into native fp16 instructions.                     */
/* ------------------------------------------------------------------------- */

float booki_f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0 && mant == 0) {
        f = sign << 31;
    } else if (exp == 0x1f) {
        f = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else if (exp == 0) {
        int e = -1;
        do { e++; mant <<= 1; } while ((mant & 0x400) == 0);
        mant &= 0x3ff;
        f = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

uint16_t booki_f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)(sign << 15);
        mant = (mant | 0x800000) >> (1 - exp);
        if (mant & 0x1000) mant += 0x2000;
        return (uint16_t)((sign << 15) | (mant >> 13));
    } else if (exp >= 0x1f) {
        return (uint16_t)((sign << 15) | (0x1fu << 10) | (mant ? 0x200u : 0u));
    } else {
        if (mant & 0x1000) {
            mant += 0x2000;
            if (mant & 0x800000) {
                mant = 0; exp++;
                if (exp >= 0x1f) return (uint16_t)((sign << 15) | (0x1fu << 10));
            }
        }
        return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | (mant >> 13));
    }
}

typedef uint16_t booki_f16;

static inline float    f16_to_f32(booki_f16 h)  { return booki_f16_to_f32(h); }
static inline booki_f16 f32_to_f16(float f)     { return booki_f32_to_f16(f); }

/* ------------------------------------------------------------------------- */
/* Scalar reference (also the conformance oracle) */
/* ------------------------------------------------------------------------- */

static void matmul_f16_scalar(const booki_f16* A, const booki_f16* B, booki_f16* C,
                              int64_t M, int64_t N, int64_t K) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += f16_to_f32(A[m * K + k]) * f16_to_f32(B[k * N + n]);
            }
            C[m * N + n] = f32_to_f16(acc);
        }
    }
}

/* ------------------------------------------------------------------------- */
/* NEON path                                                                 */
/*                                                                           */
/* 4x4 register tile. Each iteration of the inner K loop processes 8 K       */
/* elements at once using fp16 × fp16 → fp32 widening multiply-adds.         */
/* ------------------------------------------------------------------------- */

#if HAVE_NEON
/* NEON path. fp16 storage in/out, fp32 accumulation, 4x4 register tile.
 *
 * Inner loop is the canonical outer-product: for each k, load 4 A elements
 * (one column slice across the M-tile) and 4 B elements (one row slice
 * across the N-tile), then FMA each A lane against the whole B row into
 * the corresponding accumulator row. `vfmaq_laneq_f32` requires the lane
 * index to be a compile-time constant — hence the explicit 0/1/2/3 lines
 * rather than an inner `for` loop. */
static void matmul_f16_neon(const booki_f16* A_u, const booki_f16* B_u, booki_f16* C_u,
                            int64_t M, int64_t N, int64_t K) {
    const __fp16* A = (const __fp16*)A_u;
    const __fp16* B = (const __fp16*)B_u;
    __fp16*       C = (__fp16*)C_u;
    const int64_t TM = 4, TN = 4;

    int64_t Mt = M & ~(TM - 1);
    int64_t Nt = N & ~(TN - 1);

    for (int64_t m = 0; m < Mt; m += TM) {
        for (int64_t n = 0; n < Nt; n += TN) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            for (int64_t k = 0; k < K; ++k) {
                /* B row slice — 4 contiguous fp16 → fp32 vector. */
                float16x4_t b_h = vld1_f16(&B[k * N + n]);
                float32x4_t b_row = vcvt_f32_f16(b_h);

                /* A column slice (4 elements gathered from M-strided rows). */
                __fp16 a_col_h[4] = {
                    A[(m + 0) * K + k],
                    A[(m + 1) * K + k],
                    A[(m + 2) * K + k],
                    A[(m + 3) * K + k],
                };
                float32x4_t a_col = vcvt_f32_f16(vld1_f16(a_col_h));

                /* Outer product. Lane indices must be constant. */
                acc0 = vfmaq_laneq_f32(acc0, b_row, a_col, 0);
                acc1 = vfmaq_laneq_f32(acc1, b_row, a_col, 1);
                acc2 = vfmaq_laneq_f32(acc2, b_row, a_col, 2);
                acc3 = vfmaq_laneq_f32(acc3, b_row, a_col, 3);
            }

            /* Store tile, downcasting back to fp16. */
            vst1_f16(&C[(m + 0) * N + n], vcvt_f16_f32(acc0));
            vst1_f16(&C[(m + 1) * N + n], vcvt_f16_f32(acc1));
            vst1_f16(&C[(m + 2) * N + n], vcvt_f16_f32(acc2));
            vst1_f16(&C[(m + 3) * N + n], vcvt_f16_f32(acc3));
        }

        /* N-tail (cols Nt..N-1) */
        for (int64_t n = Nt; n < N; ++n) {
            for (int64_t i = 0; i < TM; ++i) {
                float acc = 0.0f;
                for (int64_t k = 0; k < K; ++k)
                    acc += (float)A[(m + i) * K + k] * (float)B[k * N + n];
                C[(m + i) * N + n] = (__fp16)acc;
            }
        }
    }

    /* M-tail (rows Mt..M-1) */
    for (int64_t m = Mt; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k)
                acc += (float)A[m * K + k] * (float)B[k * N + n];
            C[m * N + n] = (__fp16)acc;
        }
    }
}

/* Symbol the CI grep checks for. */
void booki_matmul_neon_marker_(void) {}
#endif  /* HAVE_NEON */

/* ------------------------------------------------------------------------- */
/* Public entry                                                              */
/* ------------------------------------------------------------------------- */

int booki_matmul_f16(const booki_tensor* a, const booki_tensor* b, booki_tensor* c) {
    if (!a || !b || !c) return -1;
    if (a->dtype != BOOKI_DTYPE_F16 || b->dtype != BOOKI_DTYPE_F16 || c->dtype != BOOKI_DTYPE_F16)
        return -2;
    if (a->rank != 2 || b->rank != 2 || c->rank != 2) return -3;

    int64_t M = a->shape[0], K = a->shape[1];
    if (b->shape[0] != K) return -4;
    int64_t N = b->shape[1];
    if (c->shape[0] != M || c->shape[1] != N) return -5;

    booki_backend backend = booki_backend_active();
#if HAVE_NEON
    if (backend == BOOKI_BACKEND_NEON) {
        matmul_f16_neon((const booki_f16*)a->data, (const booki_f16*)b->data,
                        (booki_f16*)c->data, M, N, K);
        return 0;
    }
#endif
    matmul_f16_scalar((const booki_f16*)a->data, (const booki_f16*)b->data,
                      (booki_f16*)c->data, M, N, K);
    return 0;
}

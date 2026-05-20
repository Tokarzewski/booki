/* SME2-based fp16 matmul.
 *
 * Targets ARMv9.2-A + SME2 + sme-f16f32 (single-precision accumulation of
 * fp16 outer products into the ZA tile). Built only when the compiler
 * advertises __ARM_FEATURE_SME; runtime dispatch additionally checks
 * HWCAP2_SME so a binary built for SME runs unchanged on phones that
 * don't have it (booki_matmul.c routes to NEON in that case).
 *
 * Layout: C[M,N] = A[M,K] * B[K,N], row-major contiguous fp16, fp32 accum.
 *
 * Algorithm (one inner kernel covers an SVL_w × SVL_w output tile):
 *   - SVL_w = number of fp32 lanes in streaming mode (16 on most Oryon
 *     cores; SVL is implementation-defined).
 *   - For each output tile (i_tile, j_tile):
 *       zero ZA tile 0 (fp32)
 *       for k = 0, 2, 4, ...:
 *           A_pair = pair of fp16 columns of A from k..k+1, rows
 *                    [i_tile..i_tile+SVL_w)
 *           B_pair = pair of fp16 rows of B from k..k+1, cols
 *                    [j_tile..j_tile+SVL_w)
 *           FMOPA(za0, A_pair, B_pair)   // 2-step fp16 outer product
 *       store ZA tile 0 back to C (downcast fp32 -> fp16)
 *
 * For correctness this first cut pre-transposes A into a packed scratch
 * buffer so the inner loop only does contiguous loads. A NEON-pack
 * could later move into the same loop body. Performance is still
 * dramatically ahead of NEON because the FMOPA fp16 instruction does a
 * 16×2 × 2×16 outer product in one issue.
 */

#include "booki_native.h"

#include <arm_sme.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef __ARM_FEATURE_SME
#error "booki_sme.c must be compiled with -march=…+sme+sme2"
#endif

/* Pack A's [M, K] -> [K/2 pairs, M, 2] so each FMOPA can load contiguous
 * pairs of A columns. M / K small (≤4096 in Kokoro) so this packing is
 * cheap relative to the matmul itself. Output is fp16, contiguous. */
static void pack_a_pairs(const __fp16* A, __fp16* Apack,
                          int64_t M, int64_t K, int64_t K_pad) {
    /* For each row of A, write its K elements into Apack at positions:
     *   Apack[k_pair * (M * 2) + m * 2 + (k_in_pair)] = A[m * K + k]
     * Equivalently for each k from 0..K-1:
     *   k_pair = k / 2, k_in_pair = k & 1
     *   Apack[k_pair * 2 * M + m * 2 + k_in_pair] = A[m * K + k] */
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            int64_t kp  = k >> 1;
            int64_t kip = k & 1;
            Apack[kp * 2 * M + m * 2 + kip] = A[m * K + k];
        }
        for (int64_t k = K; k < K_pad; ++k) {
            int64_t kp  = k >> 1;
            int64_t kip = k & 1;
            Apack[kp * 2 * M + m * 2 + kip] = (__fp16)0.0f;
        }
    }
}

/* Pack B [K, N] -> [K/2 pairs, N, 2], with the inner pair holding
 * (B[2*kp, j], B[2*kp+1, j]) for column j. This matches the layout that
 * the fp16 FMOPA expects for the right-hand source: each column j
 * contributes a (k_lo, k_hi) pair, and the SVL_h vector spans N columns
 * of such pairs. */
static void pack_b_pairs(const __fp16* B, __fp16* Bpack,
                          int64_t K, int64_t N, int64_t K_pad) {
    for (int64_t kp = 0; kp < K_pad / 2; ++kp) {
        for (int64_t j = 0; j < N; ++j) {
            int64_t k0 = 2 * kp;
            int64_t k1 = 2 * kp + 1;
            __fp16 v0 = (k0 < K) ? B[k0 * N + j] : (__fp16)0.0f;
            __fp16 v1 = (k1 < K) ? B[k1 * N + j] : (__fp16)0.0f;
            Bpack[kp * 2 * N + j * 2 + 0] = v0;
            Bpack[kp * 2 * N + j * 2 + 1] = v1;
        }
    }
}

/* Inner kernel — one output tile [TM, TN] worth of work in streaming mode.
 *
 * TM, TN must be ≤ SVL_w (the number of fp32 lanes), since each FMOPA
 * computes an SVL_w × SVL_w outer-product tile and we use ZA tile 0.
 *
 * Apack rows: M, Bpack rows: 2*K_pad (interleaved K-pairs)
 */
__arm_locally_streaming
__arm_new("za")
static void sme_tile_kernel(const __fp16* Apack, const __fp16* Bpack,
                            float* Cf32,
                            int64_t M, int64_t N, int64_t K_pad,
                            int64_t row_off, int64_t col_off,
                            int64_t TM, int64_t TN) {
    /* Predicates clamp to the tile size (handles M / N not divisible by SVL_w). */
    svbool_t pn_m = svwhilelt_b32_s64(0, TM);
    svbool_t pn_n = svwhilelt_b32_s64(0, TN);
    svbool_t pn_h = svwhilelt_b16_s64(0, TM);   /* fp16 predicate over rows */
    svbool_t pn_h_b = svwhilelt_b16_s64(0, TN); /* fp16 predicate over cols */

    /* Zero ZA tile 0 (fp32 SVL_w × SVL_w). */
    svzero_za();

    /* Iterate over K-pairs. */
    int64_t k_pairs = K_pad >> 1;
    for (int64_t kp = 0; kp < k_pairs; ++kp) {
        /* A vector: TM rows × 2 K-elements, interleaved as
         *   (row0_k0, row0_k1, row1_k0, row1_k1, ...).
         * Total SVL_h fp16 lanes; predicate covers 2*TM lanes when TM<SVL_w. */
        svbool_t pa = svwhilelt_b16_s64(0, TM * 2);
        const __fp16* a_ptr = &Apack[kp * 2 * M + row_off * 2];
        svfloat16_t a = svld1_f16(pa, a_ptr);

        /* B vector: TN cols × 2 K-elements, interleaved as
         *   (col0_k0, col0_k1, col1_k0, col1_k1, ...). */
        svbool_t pb = svwhilelt_b16_s64(0, TN * 2);
        const __fp16* b_ptr = &Bpack[kp * 2 * N + col_off * 2];
        svfloat16_t b = svld1_f16(pb, b_ptr);

        /* FMOPA fp16 (widening to fp32): for each (i, j) in the tile,
         *   ZA32[i,j] += a[i, 0]*b[0, j] + a[i, 1]*b[1, j].
         * Predicates pn_m and pn_n bound the tile in fp32-lane units. */
        svmopa_za32_f16_m(0, pn_m, pn_n, a, b);
    }
    /* Suppress unused-variable warnings for the predicates we kept
     * declared at function scope for clarity. */
    (void)pn_h; (void)pn_h_b;

    /* Read out ZA tile 0 into Cf32 row by row. svread_hor_za32_m reads
     * one horizontal slice (SVL_w fp32 lanes) from tile 0. */
    for (int64_t r = 0; r < TM; ++r) {
        svfloat32_t out = svread_hor_za32_m(svdup_n_f32(0.0f), pn_n,
                                            /*tile=*/0,
                                            /*slice=*/(uint32_t)r);
        svst1_f32(pn_n, &Cf32[r * N + col_off], out);
    }
}

/* Top-level SME matmul. Pre-packs A and B, then iterates over output
 * tiles. Each tile is at most TM × TN where TM, TN are bounded by
 * SVL_w. We choose TM = TN = 16 which matches the fp32 lane count for
 * the 512-bit streaming vector length most Oryon implementations expose.
 * If SVL is smaller, the predicates still produce correct (but slower)
 * results. */
int booki_matmul_f16_sme(const booki_tensor* a, const booki_tensor* b,
                         booki_tensor* c) {
    if (!a || !b || !c) return -1;
    if (a->dtype != BOOKI_DTYPE_F16 || b->dtype != BOOKI_DTYPE_F16 ||
        c->dtype != BOOKI_DTYPE_F16) return -2;
    if (a->rank != 2 || b->rank != 2 || c->rank != 2) return -3;
    int64_t M = a->shape[0], K = a->shape[1];
    if (b->shape[0] != K) return -4;
    int64_t N = b->shape[1];
    if (c->shape[0] != M || c->shape[1] != N) return -5;

    /* Pad K to even so the K-pair loop terminates cleanly. */
    int64_t K_pad = (K + 1) & ~1;

    void* Ap = NULL; void* Bp = NULL; void* Cp = NULL;
    if (posix_memalign(&Ap, 64, sizeof(__fp16) * M * K_pad) != 0) return -6;
    if (posix_memalign(&Bp, 64, sizeof(__fp16) * K_pad * N) != 0) { free(Ap); return -6; }
    if (posix_memalign(&Cp, 64, sizeof(float)  * M * N)     != 0) { free(Ap); free(Bp); return -6; }
    __fp16* Apack = (__fp16*)Ap;
    __fp16* Bpack = (__fp16*)Bp;
    float*  Cf32  = (float*) Cp;
    memset(Cf32, 0, sizeof(float) * M * N);

    pack_a_pairs((const __fp16*)a->data, Apack, M, K, K_pad);
    pack_b_pairs((const __fp16*)b->data, Bpack, K, N, K_pad);

    /* Tile size — keep small enough that ZA tile 0 covers it; SVL_w is
     * 16 on Oryon (512-bit streaming). */
    const int64_t TM = 16;
    const int64_t TN = 16;

    for (int64_t i = 0; i < M; i += TM) {
        int64_t tm = (M - i) < TM ? (M - i) : TM;
        for (int64_t j = 0; j < N; j += TN) {
            int64_t tn = (N - j) < TN ? (N - j) : TN;
            sme_tile_kernel(Apack, Bpack, Cf32, M, N, K_pad,
                            /*row_off=*/i, /*col_off=*/j,
                            tm, tn);
        }
    }

    /* Cast fp32 result -> fp16 output. */
    __fp16* cp = (__fp16*)c->data;
    for (int64_t i = 0; i < M * N; ++i) cp[i] = (__fp16)Cf32[i];

    free(Apack); free(Bpack); free(Cf32);
    return 0;
}

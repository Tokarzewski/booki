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
 * Tile size is driven by the streaming vector length (SVL) the hardware
 * exposes, not assumed at compile time. SVL_w = svcntw() = number of fp32
 * lanes in streaming mode. On Snapdragon 8 Elite Gen 5 (Oryon Gen 3) SVL
 * is 128 bits so SVL_w = 4. ZA tile 0 is SVL_w × SVL_w fp32 elements.
 *
 * Multi-threading: A and B are packed once on the calling thread, then the
 * M dimension is split across booki_get_threads() workers via
 * booki_parallel_for. Each worker enters its own __arm_streaming scope, so
 * the ZA tile is private per call — no cross-thread state. See
 * booki_threading.h for why the default cap is 2.
 */

#include "booki_native.h"
#include "booki_threading.h"

#include <arm_sme.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef __ARM_FEATURE_SME
#error "booki_sme.c must be compiled with -march=…+sme+sme2"
#endif

/* Pack A's [M, K] -> [K/2 pairs, M, 2]: contiguous (k_lo, k_hi) pair per row,
 * stacked by row inside each k-pair block. */
static void pack_a_pairs(const __fp16* A, __fp16* Apack,
                          int64_t M, int64_t K, int64_t K_pad) {
    for (int64_t kp = 0; kp < K_pad / 2; ++kp) {
        for (int64_t m = 0; m < M; ++m) {
            int64_t k0 = 2 * kp;
            int64_t k1 = 2 * kp + 1;
            Apack[kp * 2 * M + m * 2 + 0] = (k0 < K) ? A[m * K + k0] : (__fp16)0.0f;
            Apack[kp * 2 * M + m * 2 + 1] = (k1 < K) ? A[m * K + k1] : (__fp16)0.0f;
        }
    }
}

/* Pack B's [K, N] -> [K/2 pairs, N, 2]: contiguous (k_lo, k_hi) pair per column. */
static void pack_b_pairs(const __fp16* B, __fp16* Bpack,
                          int64_t K, int64_t N, int64_t K_pad) {
    for (int64_t kp = 0; kp < K_pad / 2; ++kp) {
        for (int64_t j = 0; j < N; ++j) {
            int64_t k0 = 2 * kp;
            int64_t k1 = 2 * kp + 1;
            Bpack[kp * 2 * N + j * 2 + 0] = (k0 < K) ? B[k0 * N + j] : (__fp16)0.0f;
            Bpack[kp * 2 * N + j * 2 + 1] = (k1 < K) ? B[k1 * N + j] : (__fp16)0.0f;
        }
    }
}

/* Streaming kernel — runs a row-slice of the matmul inside one
 * __arm_streaming scope so SVL is consistent and ZA is owned end-to-end.
 *
 * Operates on rows [m_begin, m_end) of C. Apack is laid out as
 * [K_pad/2, M_total, 2] — i.e. the M stride within a k-pair block is
 * M_total, *not* (m_end - m_begin). That lets the caller pack A once and
 * dispatch row-slices to multiple threads without re-packing. */
__arm_locally_streaming
__arm_new("za")
static void sme_matmul_stream_slice(const __fp16* Apack, const __fp16* Bpack,
                                    float* Cf32,
                                    int64_t M_total, int64_t N, int64_t K_pad,
                                    int64_t m_begin, int64_t m_end) {
    /* SVL_w = number of fp32 lanes in streaming mode. ZA tile 0 is
     * SVL_w × SVL_w fp32 elements; one FMOPA computes one such tile's
     * worth of outer-product accumulation per K-pair. */
    uint64_t svl_w = svcntw();
    int64_t  tile  = (int64_t)svl_w;

    int64_t k_pairs = K_pad >> 1;

    for (int64_t i = m_begin; i < m_end; i += tile) {
        int64_t tm = (m_end - i) < tile ? (m_end - i) : tile;
        for (int64_t j = 0; j < N; j += tile) {
            int64_t tn = (N - j) < tile ? (N - j) : tile;

            svzero_za();

            /* Input loads use whilelt predicates that mask off out-of-
             * bounds lanes to zero. The FMOPA itself uses svptrue so the
             * full ZA tile is exercised; zero-padded input contributes
             * zero to the corresponding tile rows/cols, which we then
             * skip on readback. */
            svbool_t pa  = svwhilelt_b16_s64(0, tm * 2);
            svbool_t pb  = svwhilelt_b16_s64(0, tn * 2);
            svbool_t pn  = svptrue_b16();
            svbool_t pn_n_w = svwhilelt_b32_s64(0, tn);

            for (int64_t kp = 0; kp < k_pairs; ++kp) {
                svfloat16_t a = svld1_f16(pa, &Apack[kp * 2 * M_total + i * 2]);
                svfloat16_t b = svld1_f16(pb, &Bpack[kp * 2 * N + j * 2]);
                svmopa_za32_f16_m(0, pn, pn, a, b);
            }

            /* Store tile rows into Cf32, clipping cols by predicate and
             * rows by loop bound. */
            for (int64_t r = 0; r < tm; ++r) {
                svfloat32_t v = svread_hor_za32_m(svdup_n_f32(0.0f), pn_n_w,
                                                  /*tile=*/0,
                                                  /*slice=*/(uint32_t)r);
                svst1_f32(pn_n_w, &Cf32[(i + r) * N + j], v);
            }
        }
    }
}

/* Per-call context for the parallel_for worker. */
typedef struct {
    const __fp16* Apack;
    const __fp16* Bpack;
    float*        Cf32;
    int64_t       M;
    int64_t       N;
    int64_t       K_pad;
    int64_t       tile;   /* SVL_w; chunk boundaries are snapped to this. */
} sme_thread_ctx;

static void sme_worker(int64_t begin, int64_t end, void* ctx_v) {
    sme_thread_ctx* c = (sme_thread_ctx*)ctx_v;
    if (begin >= end) return;
    sme_matmul_stream_slice(c->Apack, c->Bpack, c->Cf32,
                            c->M, c->N, c->K_pad, begin, end);
}

/* Probe SVL_w on the calling thread inside a streaming scope. Used to
 * choose the parallel_for grain so worker boundaries land on tile edges.
 * Cheap to call (entering streaming mode is a few cycles on Oryon). */
__arm_locally_streaming
static int64_t probe_svl_w(void) {
    return (int64_t)svcntw();
}

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

    int64_t K_pad = (K + 1) & ~1;

    void *Ap = NULL, *Bp = NULL, *Cp = NULL;
    if (posix_memalign(&Ap, 64, sizeof(__fp16) * M * K_pad) != 0) return -6;
    if (posix_memalign(&Bp, 64, sizeof(__fp16) * K_pad * N) != 0) { free(Ap); return -6; }
    if (posix_memalign(&Cp, 64, sizeof(float)  * M * N)     != 0) { free(Ap); free(Bp); return -6; }

    __fp16* Apack = (__fp16*)Ap;
    __fp16* Bpack = (__fp16*)Bp;
    float*  Cf32  = (float*) Cp;
    memset(Cf32, 0, sizeof(float) * M * N);

    pack_a_pairs((const __fp16*)a->data, Apack, M, K, K_pad);
    pack_b_pairs((const __fp16*)b->data, Bpack, K, N, K_pad);

    sme_thread_ctx ctx = {
        .Apack = Apack, .Bpack = Bpack, .Cf32 = Cf32,
        .M = M, .N = N, .K_pad = K_pad,
        .tile = probe_svl_w(),
    };
    /* Snap worker boundaries to SVL_w-row chunks. The kernel itself handles
     * a ragged final tile, but giving the parallel_for the tile size as
     * grain keeps the work balanced and avoids splitting a single ZA tile
     * across two threads. */
    booki_parallel_for(M, ctx.tile, sme_worker, &ctx);

    __fp16* cp = (__fp16*)c->data;
    for (int64_t i = 0; i < M * N; ++i) cp[i] = (__fp16)Cf32[i];

    free(Apack); free(Bpack); free(Cf32);
    return 0;
}

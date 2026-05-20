/* Element-wise, reductions, and activations.
 *
 * All ops accept fp16 storage and accumulate in fp32 where it matters
 * (RMSNorm reduction, softmax exp/sum, attention scores). NEON paths exist
 * for the hot ones; scalar fallback uses booki_f16_to_f32 / booki_f32_to_f16
 * so the same code runs on x86 dev hosts.
 */

#include "booki_native.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>
  #define HAVE_NEON 1
#else
  #define HAVE_NEON 0
#endif

/* ------------------------------------------------------------------------- */
/* Helpers                                                                   */
/* ------------------------------------------------------------------------- */

static int same_shape(const booki_tensor* a, const booki_tensor* b) {
    if (a->rank != b->rank) return 0;
    for (int i = 0; i < a->rank; ++i) if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}

static int64_t last_axis(const booki_tensor* t) { return t->shape[t->rank - 1]; }

static int64_t rows_of(const booki_tensor* t) {
    int64_t r = 1;
    for (int i = 0; i < t->rank - 1; ++i) r *= t->shape[i];
    return r;
}

/* Convert a contiguous chunk of fp16 → fp32 into a caller buffer. */
static void f16_chunk_to_f32(const uint16_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if HAVE_NEON
    for (; i + 7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)&src[i]);
        vst1q_f32(&dst[i],     vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(&dst[i + 4], vcvt_f32_f16(vget_high_f16(h)));
    }
#endif
    for (; i < n; ++i) dst[i] = booki_f16_to_f32(src[i]);
}

static void f32_chunk_to_f16(const float* src, uint16_t* dst, int64_t n) {
    int64_t i = 0;
#if HAVE_NEON
    for (; i + 7 < n; i += 8) {
        float16x4_t lo = vcvt_f16_f32(vld1q_f32(&src[i]));
        float16x4_t hi = vcvt_f16_f32(vld1q_f32(&src[i + 4]));
        vst1_f16((__fp16*)&dst[i],     lo);
        vst1_f16((__fp16*)&dst[i + 4], hi);
    }
#endif
    for (; i < n; ++i) dst[i] = booki_f32_to_f16(src[i]);
}

/* ------------------------------------------------------------------------- */
/* RMSNorm                                                                   */
/* ------------------------------------------------------------------------- */

int booki_rmsnorm_f16(const booki_tensor* x, const booki_tensor* w,
                      float eps, booki_tensor* out) {
    if (!x || !w || !out) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || w->dtype != BOOKI_DTYPE_F16 ||
        out->dtype != BOOKI_DTYPE_F16) return -2;
    if (!same_shape(x, out)) return -3;
    if (w->rank != 1 || w->shape[0] != last_axis(x)) return -4;

    int64_t D = last_axis(x);
    int64_t R = rows_of(x);
    const uint16_t* xp = (const uint16_t*)x->data;
    const uint16_t* wp = (const uint16_t*)w->data;
    uint16_t*       op = (uint16_t*)out->data;

    for (int64_t r = 0; r < R; ++r) {
        const uint16_t* row = xp + r * D;
        uint16_t*       o   = op + r * D;

        /* mean(x^2) in fp32 */
        float sumsq = 0.0f;
        int64_t i = 0;
#if HAVE_NEON
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (; i + 7 < D; i += 8) {
            float16x8_t h = vld1q_f16((const __fp16*)&row[i]);
            float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
            acc = vfmaq_f32(acc, lo, lo);
            acc = vfmaq_f32(acc, hi, hi);
        }
        sumsq += vaddvq_f32(acc);
#endif
        for (; i < D; ++i) {
            float v = booki_f16_to_f32(row[i]);
            sumsq += v * v;
        }

        float inv_rms = 1.0f / sqrtf(sumsq / (float)D + eps);

        /* y = x * weight * inv_rms */
        i = 0;
#if HAVE_NEON
        float32x4_t inv = vdupq_n_f32(inv_rms);
        for (; i + 7 < D; i += 8) {
            float16x8_t xh = vld1q_f16((const __fp16*)&row[i]);
            float16x8_t wh = vld1q_f16((const __fp16*)&wp[i]);
            float32x4_t xlo = vcvt_f32_f16(vget_low_f16(xh));
            float32x4_t xhi = vcvt_f32_f16(vget_high_f16(xh));
            float32x4_t wlo = vcvt_f32_f16(vget_low_f16(wh));
            float32x4_t whi = vcvt_f32_f16(vget_high_f16(wh));
            float32x4_t lo = vmulq_f32(vmulq_f32(xlo, wlo), inv);
            float32x4_t hi = vmulq_f32(vmulq_f32(xhi, whi), inv);
            vst1_f16((__fp16*)&o[i],     vcvt_f16_f32(lo));
            vst1_f16((__fp16*)&o[i + 4], vcvt_f16_f32(hi));
        }
#endif
        for (; i < D; ++i) {
            float v = booki_f16_to_f32(row[i]) * booki_f16_to_f32(wp[i]) * inv_rms;
            o[i] = booki_f32_to_f16(v);
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* SiLU + GELU                                                               */
/* ------------------------------------------------------------------------- */

static inline float silu_f32(float x) {
    /* x * sigmoid(x) = x / (1 + exp(-x)) */
    return x / (1.0f + expf(-x));
}

static inline float gelu_tanh_f32(float x) {
    /* GELU (tanh approximation) used by Kokoro:
     *   0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) )) */
    const float k = 0.7978845608028654f;        /* sqrt(2/pi) */
    float inner = k * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static int unary_apply(const booki_tensor* x, booki_tensor* out,
                       float (*fn)(float)) {
    if (!x || !out) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || out->dtype != BOOKI_DTYPE_F16) return -2;
    if (!same_shape(x, out)) return -3;
    int64_t n = booki_tensor_elements(x);
    const uint16_t* xp = (const uint16_t*)x->data;
    uint16_t*       op = (uint16_t*)out->data;
    for (int64_t i = 0; i < n; ++i) {
        op[i] = booki_f32_to_f16(fn(booki_f16_to_f32(xp[i])));
    }
    return 0;
}

int booki_silu_f16(const booki_tensor* x, booki_tensor* out) {
    return unary_apply(x, out, silu_f32);
}
int booki_gelu_f16(const booki_tensor* x, booki_tensor* out) {
    return unary_apply(x, out, gelu_tanh_f32);
}

/* ------------------------------------------------------------------------- */
/* Element-wise add / mul                                                    */
/* ------------------------------------------------------------------------- */

static int binary_apply(const booki_tensor* a, const booki_tensor* b,
                        booki_tensor* out, int is_mul) {
    if (!a || !b || !out) return -1;
    if (a->dtype != BOOKI_DTYPE_F16 || b->dtype != BOOKI_DTYPE_F16 ||
        out->dtype != BOOKI_DTYPE_F16) return -2;
    if (!same_shape(a, b) || !same_shape(a, out)) return -3;

    int64_t n = booki_tensor_elements(a);
    const uint16_t* ap = (const uint16_t*)a->data;
    const uint16_t* bp = (const uint16_t*)b->data;
    uint16_t*       op = (uint16_t*)out->data;

    int64_t i = 0;
#if HAVE_NEON
    for (; i + 7 < n; i += 8) {
        float16x8_t av = vld1q_f16((const __fp16*)&ap[i]);
        float16x8_t bv = vld1q_f16((const __fp16*)&bp[i]);
        float16x8_t cv = is_mul ? vmulq_f16(av, bv) : vaddq_f16(av, bv);
        vst1q_f16((__fp16*)&op[i], cv);
    }
#endif
    for (; i < n; ++i) {
        float av = booki_f16_to_f32(ap[i]);
        float bv = booki_f16_to_f32(bp[i]);
        op[i] = booki_f32_to_f16(is_mul ? av * bv : av + bv);
    }
    return 0;
}

int booki_add_f16(const booki_tensor* a, const booki_tensor* b, booki_tensor* out) {
    return binary_apply(a, b, out, /*is_mul=*/0);
}
int booki_mul_f16(const booki_tensor* a, const booki_tensor* b, booki_tensor* out) {
    return binary_apply(a, b, out, /*is_mul=*/1);
}

/* ------------------------------------------------------------------------- */
/* Softmax                                                                   */
/* ------------------------------------------------------------------------- */

int booki_softmax_f16(const booki_tensor* x, booki_tensor* out) {
    if (!x || !out) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || out->dtype != BOOKI_DTYPE_F16) return -2;
    if (!same_shape(x, out)) return -3;

    int64_t D = last_axis(x);
    int64_t R = rows_of(x);
    const uint16_t* xp = (const uint16_t*)x->data;
    uint16_t*       op = (uint16_t*)out->data;

    /* Heap scratch per row keeps the inner loop allocation-free and small
     * enough that Kokoro attention's D (typically ≤4096) fits in L1. */
    float* tmp = (float*)malloc(sizeof(float) * D);
    if (!tmp) return -4;

    for (int64_t r = 0; r < R; ++r) {
        const uint16_t* row = xp + r * D;
        uint16_t*       o   = op + r * D;

        f16_chunk_to_f32(row, tmp, D);

        /* max-shift for numerical stability */
        float mx = tmp[0];
        for (int64_t i = 1; i < D; ++i) if (tmp[i] > mx) mx = tmp[i];

        float sum = 0.0f;
        for (int64_t i = 0; i < D; ++i) { tmp[i] = expf(tmp[i] - mx); sum += tmp[i]; }
        float inv_sum = 1.0f / sum;
        for (int64_t i = 0; i < D; ++i) tmp[i] *= inv_sum;

        f32_chunk_to_f16(tmp, o, D);
    }
    free(tmp);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Embedding lookup                                                          */
/* ------------------------------------------------------------------------- */

int booki_embedding_f16(const booki_tensor* ids, const booki_tensor* w,
                        booki_tensor* out) {
    if (!ids || !w || !out) return -1;
    if (ids->dtype != BOOKI_DTYPE_I64) return -2;
    if (w->dtype != BOOKI_DTYPE_F16 || out->dtype != BOOKI_DTYPE_F16) return -3;
    if (w->rank != 2) return -4;

    int64_t D = w->shape[1];
    int64_t V = w->shape[0];
    int64_t T = booki_tensor_elements(ids);
    /* out is [T, D] flat; we don't enforce rank here so 1-D and 2-D ids both work. */
    if (booki_tensor_elements(out) != T * D) return -5;

    const int64_t*  ip = (const int64_t*)ids->data;
    const uint16_t* wp = (const uint16_t*)w->data;
    uint16_t*       op = (uint16_t*)out->data;
    size_t row_bytes = (size_t)D * sizeof(uint16_t);

    for (int64_t t = 0; t < T; ++t) {
        int64_t id = ip[t];
        if (id < 0 || id >= V) return -6;
        memcpy(op + t * D, wp + id * D, row_bytes);
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Scaled-dot-product attention (single head, no mask, no KV cache).         */
/* attn = softmax(Q @ K^T / sqrt(d_k)) @ V                                   */
/* ------------------------------------------------------------------------- */

int booki_attention_f16(const booki_tensor* q, const booki_tensor* k,
                        const booki_tensor* v, booki_arena* arena,
                        booki_tensor* out) {
    if (!q || !k || !v || !arena || !out) return -1;
    if (q->dtype != BOOKI_DTYPE_F16 || k->dtype != BOOKI_DTYPE_F16 ||
        v->dtype != BOOKI_DTYPE_F16 || out->dtype != BOOKI_DTYPE_F16) return -2;
    if (q->rank != 2 || k->rank != 2 || v->rank != 2 || out->rank != 2) return -3;
    int64_t M = q->shape[0], D = q->shape[1];
    int64_t N = k->shape[0];
    if (k->shape[1] != D || v->shape[0] != N || v->shape[1] != D) return -4;
    if (out->shape[0] != M || out->shape[1] != D) return -5;

    /* K_T: [D, N] — transpose K into the arena. */
    int64_t kt_shape[2] = { D, N };
    booki_tensor KT = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, kt_shape);
    if (!KT.data) return -6;
    const uint16_t* kp = (const uint16_t*)k->data;
    uint16_t*       ktp = (uint16_t*)KT.data;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t d = 0; d < D; ++d)
            ktp[d * N + n] = kp[n * D + d];

    /* scores = Q @ K^T   →  [M, N] */
    int64_t s_shape[2] = { M, N };
    booki_tensor S = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_shape);
    if (!S.data) return -7;
    int rc = booki_matmul_f16(q, &KT, &S);
    if (rc) return rc;

    /* scale by 1/sqrt(D), then softmax row-wise */
    float scale = 1.0f / sqrtf((float)D);
    int64_t total = M * N;
    uint16_t* sp = (uint16_t*)S.data;
    for (int64_t i = 0; i < total; ++i) {
        sp[i] = booki_f32_to_f16(booki_f16_to_f32(sp[i]) * scale);
    }
    rc = booki_softmax_f16(&S, &S);
    if (rc) return rc;

    /* out = scores @ V */
    rc = booki_matmul_f16(&S, v, out);
    return rc;
}

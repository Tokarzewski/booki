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

static inline float silu_f32(float x) { return x / (1.0f + expf(-x)); }
static inline float gelu_tanh_f32(float x) {
    const float k = 0.7978845608028654f;
    float inner = k * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

#if HAVE_NEON
/* Fast vectorised expf approximation (range-reduced, polynomial degree-5).
 * Accurate to <2e-3 over the range silu/gelu hit in practice (-20..20).
 * Saturates for very negative inputs to keep the inner sigmoid stable. */
static inline float32x4_t exp_ps_neon(float32x4_t x) {
    const float32x4_t log2e   = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t ln2_hi  = vdupq_n_f32(0.6931471805599453f);
    const float32x4_t one     = vdupq_n_f32(1.0f);

    /* Clamp to avoid overflow / denorm pile-up. */
    x = vminq_f32(x, vdupq_n_f32(88.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));

    /* n = round(x * log2(e)); f = x - n*ln(2). */
    float32x4_t y = vmulq_f32(x, log2e);
    int32x4_t   n = vcvtnq_s32_f32(y);
    float32x4_t fn = vcvtq_f32_s32(n);
    float32x4_t f = vsubq_f32(x, vmulq_f32(fn, ln2_hi));

    /* exp(f) ≈ 1 + f + f^2/2 + f^3/6 + f^4/24 + f^5/120 (Horner) */
    float32x4_t p = vdupq_n_f32(1.0f / 120.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f),  p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f),   p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 2.0f),   p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f),          p, f);
    p = vfmaq_f32(one,                        p, f);

    /* Apply 2^n via bit-reinterpret. */
    int32x4_t expo = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(expo));
}

static inline float32x4_t silu_ps_neon(float32x4_t x) {
    /* x * sigmoid(x) = x / (1 + exp(-x)) */
    float32x4_t e = exp_ps_neon(vnegq_f32(x));
    float32x4_t denom = vaddq_f32(vdupq_n_f32(1.0f), e);
    return vdivq_f32(x, denom);
}

/* tanh(x) = 1 - 2/(exp(2x) + 1) */
static inline float32x4_t tanh_ps_neon(float32x4_t x) {
    float32x4_t e2x = exp_ps_neon(vmulq_f32(x, vdupq_n_f32(2.0f)));
    float32x4_t denom = vaddq_f32(e2x, vdupq_n_f32(1.0f));
    return vsubq_f32(vdupq_n_f32(1.0f), vdivq_f32(vdupq_n_f32(2.0f), denom));
}

static inline float32x4_t gelu_ps_neon(float32x4_t x) {
    const float32x4_t k       = vdupq_n_f32(0.7978845608028654f);
    const float32x4_t c       = vdupq_n_f32(0.044715f);
    const float32x4_t half    = vdupq_n_f32(0.5f);
    const float32x4_t one     = vdupq_n_f32(1.0f);
    float32x4_t x3    = vmulq_f32(x, vmulq_f32(x, x));
    float32x4_t inner = vmulq_f32(k, vaddq_f32(x, vmulq_f32(c, x3)));
    return vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, tanh_ps_neon(inner))));
}
#endif

int booki_silu_f16(const booki_tensor* x, booki_tensor* out) {
    if (!x || !out) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || out->dtype != BOOKI_DTYPE_F16) return -2;
    if (!same_shape(x, out)) return -3;

    int64_t n = booki_tensor_elements(x);
    const uint16_t* xp = (const uint16_t*)x->data;
    uint16_t*       op = (uint16_t*)out->data;
    int64_t i = 0;
#if HAVE_NEON
    for (; i + 7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)&xp[i]);
        float32x4_t lo = silu_ps_neon(vcvt_f32_f16(vget_low_f16(h)));
        float32x4_t hi = silu_ps_neon(vcvt_f32_f16(vget_high_f16(h)));
        vst1_f16((__fp16*)&op[i],     vcvt_f16_f32(lo));
        vst1_f16((__fp16*)&op[i + 4], vcvt_f16_f32(hi));
    }
#endif
    for (; i < n; ++i) op[i] = booki_f32_to_f16(silu_f32(booki_f16_to_f32(xp[i])));
    return 0;
}

int booki_gelu_f16(const booki_tensor* x, booki_tensor* out) {
    if (!x || !out) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || out->dtype != BOOKI_DTYPE_F16) return -2;
    if (!same_shape(x, out)) return -3;

    int64_t n = booki_tensor_elements(x);
    const uint16_t* xp = (const uint16_t*)x->data;
    uint16_t*       op = (uint16_t*)out->data;
    int64_t i = 0;
#if HAVE_NEON
    for (; i + 7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)&xp[i]);
        float32x4_t lo = gelu_ps_neon(vcvt_f32_f16(vget_low_f16(h)));
        float32x4_t hi = gelu_ps_neon(vcvt_f32_f16(vget_high_f16(h)));
        vst1_f16((__fp16*)&op[i],     vcvt_f16_f32(lo));
        vst1_f16((__fp16*)&op[i + 4], vcvt_f16_f32(hi));
    }
#endif
    for (; i < n; ++i) op[i] = booki_f32_to_f16(gelu_tanh_f32(booki_f16_to_f32(xp[i])));
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Element-wise add / sub / mul                                              */
/* ------------------------------------------------------------------------- */

enum { OP_BIN_ADD = 0, OP_BIN_SUB = 1, OP_BIN_MUL = 2 };

static int binary_apply(const booki_tensor* a, const booki_tensor* b,
                        booki_tensor* out, int kind) {
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
        float16x8_t cv;
        switch (kind) {
            case OP_BIN_SUB: cv = vsubq_f16(av, bv); break;
            case OP_BIN_MUL: cv = vmulq_f16(av, bv); break;
            default:         cv = vaddq_f16(av, bv); break;
        }
        vst1q_f16((__fp16*)&op[i], cv);
    }
#endif
    for (; i < n; ++i) {
        float av = booki_f16_to_f32(ap[i]);
        float bv = booki_f16_to_f32(bp[i]);
        float r;
        switch (kind) {
            case OP_BIN_SUB: r = av - bv; break;
            case OP_BIN_MUL: r = av * bv; break;
            default:         r = av + bv; break;
        }
        op[i] = booki_f32_to_f16(r);
    }
    return 0;
}

int booki_add_f16(const booki_tensor* a, const booki_tensor* b, booki_tensor* out) {
    return binary_apply(a, b, out, OP_BIN_ADD);
}
int booki_sub_f16(const booki_tensor* a, const booki_tensor* b, booki_tensor* out) {
    return binary_apply(a, b, out, OP_BIN_SUB);
}
int booki_mul_f16(const booki_tensor* a, const booki_tensor* b, booki_tensor* out) {
    return binary_apply(a, b, out, OP_BIN_MUL);
}

/* ------------------------------------------------------------------------- */
/* LeakyReLU                                                                 */
/* ------------------------------------------------------------------------- */

int booki_leaky_relu_f16(const booki_tensor* x, float alpha, booki_tensor* out) {
    if (!x || !out) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || out->dtype != BOOKI_DTYPE_F16) return -2;
    if (!same_shape(x, out)) return -3;

    int64_t n = booki_tensor_elements(x);
    const uint16_t* xp = (const uint16_t*)x->data;
    uint16_t*       op = (uint16_t*)out->data;

    int64_t i = 0;
#if HAVE_NEON
    float16x8_t va = vdupq_n_f16((__fp16)alpha);
    float16x8_t vz = vdupq_n_f16(0.0f);
    for (; i + 7 < n; i += 8) {
        float16x8_t v = vld1q_f16((const __fp16*)&xp[i]);
        uint16x8_t  pos = vcgtq_f16(v, vz);
        float16x8_t neg = vmulq_f16(v, va);
        float16x8_t r = vbslq_f16(pos, v, neg);
        vst1q_f16((__fp16*)&op[i], r);
    }
#endif
    for (; i < n; ++i) {
        float v = booki_f16_to_f32(xp[i]);
        op[i] = booki_f32_to_f16(v > 0.0f ? v : alpha * v);
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* sin / cos                                                                 */
/*                                                                           */
/* Range-reduce x to [-pi, pi] using a high-precision multiplier, then       */
/* evaluate a degree-7 odd polynomial for sin. cos(x) = sin(x + pi/2).      */
/* Accuracy: < 1e-3 max-abs over the literary range Kokoro feeds in.        */
/* ------------------------------------------------------------------------- */

static inline float sin_scalar(float x) {
    /* Range-reduce to [-pi, pi]. */
    const float TWO_PI    = 6.283185307179586f;
    const float INV_TWO_PI= 0.15915494309189535f;
    float k = floorf(x * INV_TWO_PI + 0.5f);
    float y = x - k * TWO_PI;
    /* Map y in [-pi, pi] to [-pi/2, pi/2] using sin symmetry. */
    if (y >  1.5707963267948966f) y =  3.141592653589793f - y;
    if (y < -1.5707963267948966f) y = -3.141592653589793f - y;
    /* Horner of x - x^3/6 + x^5/120 - x^7/5040 (good to ~1e-7 on [-pi/2,pi/2]). */
    float y2 = y * y;
    float p = -1.984126984e-4f;     /* -1/5040 */
    p = p * y2 +  8.333333333e-3f;  /*  1/120  */
    p = p * y2 + -1.666666667e-1f;  /* -1/6    */
    p = p * y2 +  1.0f;
    return y * p;
}

int booki_sin_f16(const booki_tensor* x, booki_tensor* out) {
    if (!x || !out) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || out->dtype != BOOKI_DTYPE_F16) return -2;
    if (!same_shape(x, out)) return -3;
    int64_t n = booki_tensor_elements(x);
    const uint16_t* xp = (const uint16_t*)x->data;
    uint16_t*       op = (uint16_t*)out->data;
    for (int64_t i = 0; i < n; ++i)
        op[i] = booki_f32_to_f16(sin_scalar(booki_f16_to_f32(xp[i])));
    return 0;
}

int booki_cos_f16(const booki_tensor* x, booki_tensor* out) {
    if (!x || !out) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || out->dtype != BOOKI_DTYPE_F16) return -2;
    if (!same_shape(x, out)) return -3;
    int64_t n = booki_tensor_elements(x);
    const uint16_t* xp = (const uint16_t*)x->data;
    uint16_t*       op = (uint16_t*)out->data;
    for (int64_t i = 0; i < n; ++i) {
        float v = booki_f16_to_f32(xp[i]);
        op[i] = booki_f32_to_f16(sin_scalar(v + 1.5707963267948966f));
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* InstanceNormalization                                                     */
/*                                                                           */
/* Input  : [..., C, T]    (last two axes are channel + spatial)             */
/* scale  : [C]                                                              */
/* bias   : [C]   (NULL allowed)                                             */
/* Output : same shape as input                                              */
/*                                                                           */
/* For each (sample, channel) row of length T, compute mean and variance     */
/* in fp32, then normalise + scale + bias. fp32 accumulation keeps the       */
/* error budget reasonable for the channel sizes Kokoro uses (up to 512).    */
/* ------------------------------------------------------------------------- */

int booki_instance_norm_f16(const booki_tensor* x,
                            const booki_tensor* scale,
                            const booki_tensor* bias,
                            float eps, booki_tensor* out) {
    if (!x || !scale || !out) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || scale->dtype != BOOKI_DTYPE_F16 ||
        out->dtype != BOOKI_DTYPE_F16) return -2;
    if (bias && bias->dtype != BOOKI_DTYPE_F16) return -2;
    if (!same_shape(x, out)) return -3;
    if (x->rank < 2) return -4;

    int64_t T = x->shape[x->rank - 1];
    int64_t C = x->shape[x->rank - 2];
    if (scale->rank != 1 || scale->shape[0] != C) return -5;
    if (bias && (bias->rank != 1 || bias->shape[0] != C)) return -5;

    int64_t batches = 1;
    for (int i = 0; i < x->rank - 2; ++i) batches *= x->shape[i];

    const uint16_t* xp = (const uint16_t*)x->data;
    const uint16_t* sp = (const uint16_t*)scale->data;
    const uint16_t* bp = bias ? (const uint16_t*)bias->data : NULL;
    uint16_t*       op = (uint16_t*)out->data;

    for (int64_t b = 0; b < batches; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            const uint16_t* row = xp + (b * C + c) * T;
            uint16_t*       orow = op + (b * C + c) * T;

            /* Mean + variance in fp32. */
            float sum = 0.0f, sumsq = 0.0f;
            int64_t t = 0;
#if HAVE_NEON
            float32x4_t vsum = vdupq_n_f32(0.0f);
            float32x4_t vssq = vdupq_n_f32(0.0f);
            for (; t + 7 < T; t += 8) {
                float16x8_t h = vld1q_f16((const __fp16*)&row[t]);
                float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
                float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
                vsum = vaddq_f32(vsum, vaddq_f32(lo, hi));
                vssq = vfmaq_f32(vssq, lo, lo);
                vssq = vfmaq_f32(vssq, hi, hi);
            }
            sum   += vaddvq_f32(vsum);
            sumsq += vaddvq_f32(vssq);
#endif
            for (; t < T; ++t) {
                float v = booki_f16_to_f32(row[t]);
                sum += v; sumsq += v * v;
            }
            float mean = sum / (float)T;
            float var  = sumsq / (float)T - mean * mean;
            float inv  = 1.0f / sqrtf(var + eps);

            float s = booki_f16_to_f32(sp[c]);
            float bv = bp ? booki_f16_to_f32(bp[c]) : 0.0f;

            t = 0;
#if HAVE_NEON
            float32x4_t vmean = vdupq_n_f32(mean);
            float32x4_t vinv  = vdupq_n_f32(inv);
            float32x4_t vs    = vdupq_n_f32(s);
            float32x4_t vb    = vdupq_n_f32(bv);
            for (; t + 7 < T; t += 8) {
                float16x8_t h = vld1q_f16((const __fp16*)&row[t]);
                float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
                float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
                lo = vmulq_f32(vsubq_f32(lo, vmean), vinv);
                hi = vmulq_f32(vsubq_f32(hi, vmean), vinv);
                lo = vfmaq_f32(vb, lo, vs);
                hi = vfmaq_f32(vb, hi, vs);
                vst1_f16((__fp16*)&orow[t],     vcvt_f16_f32(lo));
                vst1_f16((__fp16*)&orow[t + 4], vcvt_f16_f32(hi));
            }
#endif
            for (; t < T; ++t) {
                float v = booki_f16_to_f32(row[t]);
                v = (v - mean) * inv * s + bv;
                orow[t] = booki_f32_to_f16(v);
            }
        }
    }
    return 0;
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

/* ------------------------------------------------------------------------- */
/* Multi-head attention                                                      */
/*                                                                           */
/* Q [M, H*D], K [N, H*D], V [N, H*D]  →  out [M, H*D]                       */
/*                                                                           */
/* Splits the embedding dimension into H heads, runs single-head attention   */
/* on each independently, concatenates the results. KV cache + causal mask  */
/* arrive with the graph executor.                                           */
/* ------------------------------------------------------------------------- */

int booki_multihead_attention_f16(const booki_tensor* q, const booki_tensor* k,
                                  const booki_tensor* v, int num_heads,
                                  booki_arena* arena, booki_tensor* out) {
    if (!q || !k || !v || !arena || !out) return -1;
    if (q->dtype != BOOKI_DTYPE_F16 || k->dtype != BOOKI_DTYPE_F16 ||
        v->dtype != BOOKI_DTYPE_F16 || out->dtype != BOOKI_DTYPE_F16) return -2;
    if (q->rank != 2 || k->rank != 2 || v->rank != 2 || out->rank != 2) return -3;
    if (num_heads <= 0) return -4;

    int64_t M = q->shape[0], E = q->shape[1];
    int64_t N = k->shape[0];
    if (k->shape[1] != E || v->shape[0] != N || v->shape[1] != E) return -5;
    if (out->shape[0] != M || out->shape[1] != E) return -6;
    if (E % num_heads != 0) return -7;
    int64_t D = E / num_heads;

    /* Per-head slabs. Each head reads a contiguous D-element column of the
     * input rows and writes a contiguous D-element column of the output.
     * We materialise each head's Q/K/V as a fresh contiguous tensor in the
     * arena so booki_attention_f16 (which assumes contiguous rows) just
     * works. Avoids implementing strided matmul.  */
    int64_t s_q[2] = { M, D };
    int64_t s_kv[2]= { N, D };

    for (int h = 0; h < num_heads; ++h) {
        booki_tensor Qh = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_q);
        booki_tensor Kh = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_kv);
        booki_tensor Vh = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_kv);
        booki_tensor Yh = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_q);
        if (!Qh.data || !Kh.data || !Vh.data || !Yh.data) return -8;

        const uint16_t* qp = (const uint16_t*)q->data;
        const uint16_t* kp = (const uint16_t*)k->data;
        const uint16_t* vp = (const uint16_t*)v->data;
        uint16_t* qhp = (uint16_t*)Qh.data;
        uint16_t* khp = (uint16_t*)Kh.data;
        uint16_t* vhp = (uint16_t*)Vh.data;

        int64_t off = (int64_t)h * D;
        for (int64_t m = 0; m < M; ++m)
            memcpy(qhp + m * D, qp + m * E + off, D * sizeof(uint16_t));
        for (int64_t n = 0; n < N; ++n) {
            memcpy(khp + n * D, kp + n * E + off, D * sizeof(uint16_t));
            memcpy(vhp + n * D, vp + n * E + off, D * sizeof(uint16_t));
        }

        int rc = booki_attention_f16(&Qh, &Kh, &Vh, arena, &Yh);
        if (rc) return rc;

        uint16_t* op  = (uint16_t*)out->data;
        uint16_t* yhp = (uint16_t*)Yh.data;
        for (int64_t m = 0; m < M; ++m)
            memcpy(op + m * E + off, yhp + m * D, D * sizeof(uint16_t));
    }
    return 0;
}

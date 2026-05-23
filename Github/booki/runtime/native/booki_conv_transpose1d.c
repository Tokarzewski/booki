/* 1-D transposed convolution + 1-D resize.
 *
 * ConvTranspose1d is the dual of Conv1d: for each input sample at index n,
 * scatter it across the output at positions [n*stride - padding + k*dilation]
 * for k in [0, K). Same weight layout as PyTorch:
 *
 *   weight : [in_channels, out_channels / groups, kernel]
 *
 * Output length:
 *   L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
 *
 * Used by HiFi-GAN-family vocoders for upsampling. We do the direct
 * scatter implementation here — fast enough for Kokoro's small vocoder
 * spatial sizes, and easy to verify.
 */

#include "booki_native.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int booki_conv_transpose1d_f16(const booki_tensor* x, const booki_tensor* w,
                               const booki_tensor* bias,
                               int64_t stride, int64_t padding,
                               int64_t output_padding, int64_t dilation,
                               int64_t groups, booki_tensor* out)
{
    if (!x || !w || !out) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || w->dtype != BOOKI_DTYPE_F16 ||
        out->dtype != BOOKI_DTYPE_F16) return -2;
    if (bias && bias->dtype != BOOKI_DTYPE_F16) return -2;
    if (x->rank != 3 || w->rank != 3 || out->rank != 3) return -3;

    int64_t B    = x->shape[0];
    int64_t Cin  = x->shape[1];
    int64_t Lin  = x->shape[2];
    int64_t W_Cin = w->shape[0];
    int64_t Cout_per_g = w->shape[1];
    int64_t K   = w->shape[2];

    if (groups <= 0)              return -4;
    if (Cin != W_Cin)             return -5;
    if (Cin % groups != 0)        return -5;
    int64_t Cout = Cout_per_g * groups;
    if (bias && (bias->rank != 1 || bias->shape[0] != Cout)) return -6;

    int64_t Lout = (Lin - 1) * stride - 2 * padding + dilation * (K - 1)
                 + output_padding + 1;
    if (out->shape[0] != B || out->shape[1] != Cout || out->shape[2] != Lout)
        return -7;

    int64_t Cin_per_g = Cin / groups;

    const uint16_t* xp = (const uint16_t*)x->data;
    const uint16_t* wp = (const uint16_t*)w->data;
    const uint16_t* bp = bias ? (const uint16_t*)bias->data : NULL;
    uint16_t*       op = (uint16_t*)out->data;

    /* Accumulate in fp32 to keep the kernel's many partial sums sane. */
    float* acc = (float*)malloc(sizeof(float) * Cout * Lout);
    if (!acc) return -8;

    for (int64_t b = 0; b < B; ++b) {
        /* Initialise from bias (broadcast across L_out) or zero. */
        for (int64_t oc = 0; oc < Cout; ++oc) {
            float bv = bp ? booki_f16_to_f32(bp[oc]) : 0.0f;
            for (int64_t l = 0; l < Lout; ++l) acc[oc * Lout + l] = bv;
        }

        /* Scatter every input × kernel contribution into the right output bin. */
        for (int64_t g = 0; g < groups; ++g) {
            for (int64_t ic_in_g = 0; ic_in_g < Cin_per_g; ++ic_in_g) {
                int64_t ic = g * Cin_per_g + ic_in_g;
                const uint16_t* xrow = xp + (b * Cin + ic) * Lin;

                for (int64_t oc_in_g = 0; oc_in_g < Cout_per_g; ++oc_in_g) {
                    int64_t oc = g * Cout_per_g + oc_in_g;
                    const uint16_t* wrow = wp + (ic * Cout_per_g + oc_in_g) * K;
                    float* aout = acc + oc * Lout;

                    for (int64_t n = 0; n < Lin; ++n) {
                        float v = booki_f16_to_f32(xrow[n]);
                        if (v == 0.0f) continue;
                        for (int64_t k = 0; k < K; ++k) {
                            int64_t l = n * stride - padding + k * dilation;
                            if (l < 0 || l >= Lout) continue;
                            aout[l] += v * booki_f16_to_f32(wrow[k]);
                        }
                    }
                }
            }
        }

        /* Cast fp32 accumulator into the fp16 output. */
        for (int64_t oc = 0; oc < Cout; ++oc) {
            for (int64_t l = 0; l < Lout; ++l) {
                op[(b * Cout + oc) * Lout + l] =
                    booki_f32_to_f16(acc[oc * Lout + l]);
            }
        }
    }

    free(acc);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* 1-D resize                                                                 */
/*                                                                            */
/* Treats the last axis of x as the spatial dimension. All other axes are     */
/* preserved (batched). For nearest mode we use ONNX's default                 */
/* coordinate_transformation_mode = "half_pixel" with nearest_mode = "floor".  */
/* For linear mode we use bilinear interpolation, again with half-pixel       */
/* alignment.                                                                  */
/* ------------------------------------------------------------------------- */

int booki_resize1d_f16(const booki_tensor* x, int64_t out_size,
                       booki_resize_mode mode, booki_tensor* out) {
    if (!x || !out) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || out->dtype != BOOKI_DTYPE_F16) return -2;
    if (x->rank < 1 || out->rank != x->rank) return -3;
    int64_t Lin = x->shape[x->rank - 1];
    if (out_size <= 0 || out->shape[out->rank - 1] != out_size) return -4;
    for (int i = 0; i < x->rank - 1; ++i)
        if (out->shape[i] != x->shape[i]) return -5;

    int64_t rows = 1;
    for (int i = 0; i < x->rank - 1; ++i) rows *= x->shape[i];

    const uint16_t* xp = (const uint16_t*)x->data;
    uint16_t*       op = (uint16_t*)out->data;
    float scale = (float)Lin / (float)out_size;

    for (int64_t r = 0; r < rows; ++r) {
        const uint16_t* in_row = xp + r * Lin;
        uint16_t*       out_row = op + r * out_size;
        for (int64_t i = 0; i < out_size; ++i) {
            float src = ((float)i + 0.5f) * scale - 0.5f;
            if (mode == BOOKI_RESIZE_NEAREST) {
                int64_t s = (int64_t)floorf(src + 0.5f);
                if (s < 0) s = 0;
                if (s >= Lin) s = Lin - 1;
                out_row[i] = in_row[s];
            } else {
                int64_t s0 = (int64_t)floorf(src);
                int64_t s1 = s0 + 1;
                float t = src - (float)s0;
                if (s0 < 0)    { s0 = 0; t = 0.0f; }
                if (s1 < 0)    { s1 = 0; }
                if (s0 >= Lin) { s0 = Lin - 1; }
                if (s1 >= Lin) { s1 = Lin - 1; }
                float v0 = booki_f16_to_f32(in_row[s0]);
                float v1 = booki_f16_to_f32(in_row[s1]);
                out_row[i] = booki_f32_to_f16(v0 * (1.0f - t) + v1 * t);
            }
        }
    }
    return 0;
}

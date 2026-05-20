/* 1-D convolution (PyTorch / ONNX layout).
 *
 *   input  : [batch, in_channels,  in_length]
 *   weight : [out_channels, in_channels / groups, kernel]
 *   bias   : [out_channels]      (optional, NULL allowed)
 *   output : [batch, out_channels, out_length]
 *
 * Supports stride, padding (symmetric, zero-padded), and dilation.
 * Groups == in_channels covers depthwise conv (used by Kokoro's duration
 * predictor). Groups == 1 is the default dense convolution.
 *
 * Implementation: direct (no im2col + matmul). For Kokoro's tiny conv1d
 * tensors this is faster than im2col anyway. NEON path coming once we hit
 * a model where it's the bottleneck.
 */

#include "booki_native.h"

#include <stdint.h>
#include <string.h>

int booki_conv1d_f16(const booki_tensor* x,
                     const booki_tensor* w,
                     const booki_tensor* bias,
                     int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                     booki_tensor* out)
{
    if (!x || !w || !out) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || w->dtype != BOOKI_DTYPE_F16 ||
        out->dtype != BOOKI_DTYPE_F16) return -2;
    if (bias && bias->dtype != BOOKI_DTYPE_F16) return -2;
    if (x->rank != 3 || w->rank != 3 || out->rank != 3) return -3;

    int64_t B  = x->shape[0];
    int64_t Cin = x->shape[1];
    int64_t L  = x->shape[2];
    int64_t Cout = w->shape[0];
    int64_t W_Cin = w->shape[1];
    int64_t K  = w->shape[2];

    if (groups <= 0) return -4;
    if (Cin % groups != 0) return -5;
    if (Cout % groups != 0) return -5;
    if (W_Cin * groups != Cin) return -5;
    if (bias && (bias->rank != 1 || bias->shape[0] != Cout)) return -6;

    int64_t out_L = (L + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    if (out->shape[0] != B || out->shape[1] != Cout || out->shape[2] != out_L) return -7;

    int64_t Cin_per_g  = Cin  / groups;
    int64_t Cout_per_g = Cout / groups;

    const uint16_t* xp = (const uint16_t*)x->data;
    const uint16_t* wp = (const uint16_t*)w->data;
    const uint16_t* bp = bias ? (const uint16_t*)bias->data : NULL;
    uint16_t*       op = (uint16_t*)out->data;

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t g = 0; g < groups; ++g) {
            for (int64_t oc_in_g = 0; oc_in_g < Cout_per_g; ++oc_in_g) {
                int64_t oc = g * Cout_per_g + oc_in_g;
                float bias_v = bp ? booki_f16_to_f32(bp[oc]) : 0.0f;

                for (int64_t ol = 0; ol < out_L; ++ol) {
                    float acc = bias_v;
                    int64_t il_base = ol * stride - padding;

                    for (int64_t ic_in_g = 0; ic_in_g < Cin_per_g; ++ic_in_g) {
                        int64_t ic = g * Cin_per_g + ic_in_g;
                        const uint16_t* xrow = xp + (b * Cin + ic) * L;
                        const uint16_t* wrow = wp + (oc * Cin_per_g + ic_in_g) * K;

                        for (int64_t k = 0; k < K; ++k) {
                            int64_t il = il_base + k * dilation;
                            if (il < 0 || il >= L) continue;
                            acc += booki_f16_to_f32(xrow[il]) * booki_f16_to_f32(wrow[k]);
                        }
                    }
                    op[(b * Cout + oc) * out_L + ol] = booki_f32_to_f16(acc);
                }
            }
        }
    }
    return 0;
}

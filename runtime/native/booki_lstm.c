/* Vanilla LSTM forward pass.
 *
 * Implements the LLaMA / ONNX gate ordering: i, o, f, c (input, output,
 * forget, cell-candidate) packed contiguously in the gate matrix rows.
 * That's a different order from PyTorch's i, f, g, o — when converting
 * weights from PyTorch we reshuffle in the converter so the runtime
 * here doesn't have to branch on convention.
 *
 * The recurrence (matching ONNX LSTM "noPeepholes" form, no projection):
 *
 *   gates_t = W @ x_t + R @ h_{t-1} + b_W + b_R          (4*hidden)
 *   i_t = sigmoid(gates_t[0:H])
 *   o_t = sigmoid(gates_t[H:2H])
 *   f_t = sigmoid(gates_t[2H:3H])
 *   c̃_t= tanh(gates_t[3H:4H])
 *   c_t = f_t * c_{t-1} + i_t * c̃_t
 *   h_t = o_t * tanh(c_t)
 *
 * fp32 accumulation throughout to keep recurrent stability.
 */

#include "booki_native.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline float sigmoid_f32(float x) { return 1.0f / (1.0f + expf(-x)); }

/* Compute  out += M @ v  where M is [4H, D] fp16, v is [D] fp16, out is [4H] fp32. */
static void gemv_accum(const uint16_t* M, const uint16_t* v,
                       float* out, int64_t H4, int64_t D) {
    for (int64_t i = 0; i < H4; ++i) {
        float acc = 0.0f;
        const uint16_t* row = M + i * D;
        for (int64_t k = 0; k < D; ++k)
            acc += booki_f16_to_f32(row[k]) * booki_f16_to_f32(v[k]);
        out[i] += acc;
    }
}

int booki_lstm_f16(const booki_tensor* x,
                   const booki_tensor* W, const booki_tensor* R,
                   const booki_tensor* b,
                   const booki_tensor* h0, const booki_tensor* c0,
                   booki_arena* arena, booki_tensor* out)
{
    if (!x || !W || !R || !out || !arena) return -1;
    if (x->dtype != BOOKI_DTYPE_F16 || W->dtype != BOOKI_DTYPE_F16 ||
        R->dtype != BOOKI_DTYPE_F16 || out->dtype != BOOKI_DTYPE_F16) return -2;
    if (b && b->dtype != BOOKI_DTYPE_F16) return -2;
    if (x->rank != 2 || W->rank != 2 || R->rank != 2 || out->rank != 2) return -3;

    int64_t T = x->shape[0];
    int64_t D = x->shape[1];
    if (W->shape[1] != D) return -4;
    if (W->shape[0] % 4 != 0) return -5;
    int64_t H4 = W->shape[0];
    int64_t H  = H4 / 4;
    if (R->shape[0] != H4 || R->shape[1] != H) return -6;
    if (b && b->shape[0] != 8 * H) return -7;
    if (h0 && h0->shape[0] != H) return -8;
    if (c0 && c0->shape[0] != H) return -8;
    if (out->shape[0] != T || out->shape[1] != H) return -9;

    /* Persistent state across timesteps. */
    int64_t shape_state[1] = { H };
    booki_tensor h = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 1, shape_state);
    booki_tensor c_state = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 1, shape_state);
    if (!h.data || !c_state.data) return -10;

    if (h0) memcpy(h.data,        h0->data, H * sizeof(uint16_t));
    else    memset(h.data, 0,                H * sizeof(uint16_t));
    if (c0) memcpy(c_state.data,  c0->data, H * sizeof(uint16_t));
    else    memset(c_state.data, 0,         H * sizeof(uint16_t));

    /* Per-step scratch. */
    float* gates = (float*)malloc(sizeof(float) * H4);
    if (!gates) return -11;

    const uint16_t* Wp = (const uint16_t*)W->data;
    const uint16_t* Rp = (const uint16_t*)R->data;
    const uint16_t* bp = b ? (const uint16_t*)b->data : NULL;
    const uint16_t* xp = (const uint16_t*)x->data;
    uint16_t*       op = (uint16_t*)out->data;
    uint16_t*       hp = (uint16_t*)h.data;
    uint16_t*       cp = (uint16_t*)c_state.data;

    for (int64_t t = 0; t < T; ++t) {
        /* gates = b_W + b_R   (split into the two halves of the 8H bias).  */
        if (bp) {
            for (int64_t i = 0; i < H4; ++i) {
                gates[i] = booki_f16_to_f32(bp[i]) + booki_f16_to_f32(bp[H4 + i]);
            }
        } else {
            memset(gates, 0, sizeof(float) * H4);
        }
        gemv_accum(Wp, xp + t * D, gates, H4, D);
        gemv_accum(Rp, hp,         gates, H4, H);

        /* Split + non-linear gates. */
        const float* g_i = gates;
        const float* g_o = gates + H;
        const float* g_f = gates + 2 * H;
        const float* g_c = gates + 3 * H;

        for (int64_t i = 0; i < H; ++i) {
            float it = sigmoid_f32(g_i[i]);
            float ot = sigmoid_f32(g_o[i]);
            float ft = sigmoid_f32(g_f[i]);
            float ct = tanhf      (g_c[i]);
            float c_prev = booki_f16_to_f32(cp[i]);
            float c_new  = ft * c_prev + it * ct;
            float h_new  = ot * tanhf(c_new);
            cp[i] = booki_f32_to_f16(c_new);
            hp[i] = booki_f32_to_f16(h_new);
            op[t * H + i] = hp[i];
        }
    }

    free(gates);
    return 0;
}

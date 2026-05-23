/* Composed transformer-block bench.
 *
 * Runs one PreNorm transformer-block forward pass per iteration with the
 * runtime's existing ops:
 *
 *   y = x
 *   y = y + Attn( RMSNorm(y) @ Wqkv ) @ Wo
 *   y = y + (SiLU( RMSNorm(y) @ W_gate ) * (RMSNorm(y) @ W_up)) @ W_down
 *
 * The numbers are the first end-to-end latency the native runtime produces
 * that's comparable to what the eventual graph executor will measure
 * per-layer. Outputs the same JSON shape as booki_bench so the CI gate
 * tracks it over time.
 */

#include "booki_native.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1e6;
}

static void fill_random(uint16_t* p, int64_t n, uint64_t seed) {
    uint64_t s = seed;
    for (int64_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        /* Map to a small fp16 ish range so matmul accumulation stays sane. */
        float f = ((int64_t)(s >> 32) & 0xffff) / 65535.0f * 0.2f - 0.1f;
        p[i] = booki_f32_to_f16(f);
    }
}

typedef struct {
    int64_t seq;
    int64_t hidden;
    int     heads;
    int64_t ffn_dim;
} block_shape;

static int run_block(const block_shape* sh, int iters, int json, int* first) {
    /* Arena needs to hold:
     *   x, residual, norm, qkv (3*hidden), q/k/v slices (auto via MHA),
     *   attn_out, w_o output, gate/up/down activations, ffn output.
     * Generous ceiling: 64 MB covers up to ~seq=512 hidden=2048 ffn=8192. */
    booki_arena* arena = booki_arena_create(64ull * 1024 * 1024);
    if (!arena) return 1;

    int64_t s_x[2]   = { sh->seq, sh->hidden };
    int64_t s_w[2]   = { sh->hidden, sh->hidden };
    int64_t s_wff[2] = { sh->hidden, sh->ffn_dim };
    int64_t s_wfd[2] = { sh->ffn_dim, sh->hidden };
    int64_t s_n[1]   = { sh->hidden };
    int64_t s_ffn[2] = { sh->seq, sh->ffn_dim };

    booki_tensor x     = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_x);
    booki_tensor norm  = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_x);
    booki_tensor q     = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_x);
    booki_tensor k     = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_x);
    booki_tensor v     = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_x);
    booki_tensor attn  = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_x);
    booki_tensor wo_o  = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_x);
    booki_tensor gate  = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_ffn);
    booki_tensor up    = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_ffn);
    booki_tensor swi   = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_ffn);
    booki_tensor down  = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_x);

    booki_tensor Wq    = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_w);
    booki_tensor Wk    = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_w);
    booki_tensor Wv    = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_w);
    booki_tensor Wo    = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_w);
    booki_tensor W_g   = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_wff);
    booki_tensor W_u   = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_wff);
    booki_tensor W_d   = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s_wfd);
    booki_tensor norm_w = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 1, s_n);

    fill_random((uint16_t*)x.data,     sh->seq * sh->hidden,         1);
    fill_random((uint16_t*)Wq.data,    sh->hidden * sh->hidden,      2);
    fill_random((uint16_t*)Wk.data,    sh->hidden * sh->hidden,      3);
    fill_random((uint16_t*)Wv.data,    sh->hidden * sh->hidden,      4);
    fill_random((uint16_t*)Wo.data,    sh->hidden * sh->hidden,      5);
    fill_random((uint16_t*)W_g.data,   sh->hidden * sh->ffn_dim,     6);
    fill_random((uint16_t*)W_u.data,   sh->hidden * sh->ffn_dim,     7);
    fill_random((uint16_t*)W_d.data,   sh->ffn_dim * sh->hidden,     8);
    fill_random((uint16_t*)norm_w.data, sh->hidden,                  9);

    /* Per-iter scratch arena so each run is reset cleanly. */
    booki_arena* iter_arena = booki_arena_create(32ull * 1024 * 1024);
    if (!iter_arena) { booki_arena_destroy(arena); return 1; }

    /* Warmup */
    booki_rmsnorm_f16(&x, &norm_w, 1e-6f, &norm);
    booki_matmul_f16(&norm, &Wq, &q);
    booki_matmul_f16(&norm, &Wk, &k);
    booki_matmul_f16(&norm, &Wv, &v);
    booki_multihead_attention_f16(&q, &k, &v, sh->heads, iter_arena, &attn);
    booki_arena_reset(iter_arena);
    booki_matmul_f16(&attn, &Wo, &wo_o);
    booki_add_f16(&x, &wo_o, &x);
    booki_rmsnorm_f16(&x, &norm_w, 1e-6f, &norm);
    booki_matmul_f16(&norm, &W_g, &gate);
    booki_silu_f16(&gate, &gate);
    booki_matmul_f16(&norm, &W_u, &up);
    booki_mul_f16(&gate, &up, &swi);
    booki_matmul_f16(&swi, &W_d, &down);
    booki_add_f16(&x, &down, &x);

    double t0 = now_ms();
    for (int it = 0; it < iters; ++it) {
        booki_rmsnorm_f16(&x, &norm_w, 1e-6f, &norm);
        booki_matmul_f16(&norm, &Wq, &q);
        booki_matmul_f16(&norm, &Wk, &k);
        booki_matmul_f16(&norm, &Wv, &v);
        booki_multihead_attention_f16(&q, &k, &v, sh->heads, iter_arena, &attn);
        booki_arena_reset(iter_arena);
        booki_matmul_f16(&attn, &Wo, &wo_o);
        booki_add_f16(&x, &wo_o, &x);
        booki_rmsnorm_f16(&x, &norm_w, 1e-6f, &norm);
        booki_matmul_f16(&norm, &W_g, &gate);
        booki_silu_f16(&gate, &gate);
        booki_matmul_f16(&norm, &W_u, &up);
        booki_mul_f16(&gate, &up, &swi);
        booki_matmul_f16(&swi, &W_d, &down);
        booki_add_f16(&x, &down, &x);
    }
    double dt = (now_ms() - t0) / iters;

    char name[64];
    snprintf(name, sizeof(name), "block_s%lld_h%lld_ff%lld",
             (long long)sh->seq, (long long)sh->hidden, (long long)sh->ffn_dim);
    double tokens_per_sec = 1000.0 / dt;
    if (json) {
        printf("%s    { \"name\": \"%s\", \"ms_per_token\": %.4f, "
               "\"tokens_per_sec\": %.2f }",
               *first ? "" : ",\n", name, dt, tokens_per_sec);
        *first = 0;
    } else {
        printf("%-30s  %.3f ms   %.1f blocks/s\n", name, dt, tokens_per_sec);
    }

    booki_arena_destroy(iter_arena);
    booki_arena_destroy(arena);
    return 0;
}

int main(int argc, char** argv) {
    int iters = 3;
    int json = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json")) json = 1;
    }

    if (json) {
        printf("{\n");
        printf("  \"backend\": \"native-%s\",\n", booki_backend_describe(booki_backend_active()));
        printf("  \"runtime\": \"%s\",\n", booki_native_version());
        printf("  \"benchmarks\": [\n");
    } else {
        printf("backend          native-%s\n", booki_backend_describe(booki_backend_active()));
    }

    /* Three Kokoro-ish shapes. */
    block_shape shapes[] = {
        { /*seq=*/ 64,  /*hidden=*/ 256, /*heads=*/ 4, /*ffn=*/ 1024 },
        { /*seq=*/ 128, /*hidden=*/ 512, /*heads=*/ 8, /*ffn=*/ 2048 },
    };
    int first = 1;
    for (int i = 0; i < (int)(sizeof(shapes)/sizeof(shapes[0])); ++i) {
        if (run_block(&shapes[i], iters, json, &first)) return 1;
    }

    if (json) printf("\n  ]\n}\n");
    return 0;
}

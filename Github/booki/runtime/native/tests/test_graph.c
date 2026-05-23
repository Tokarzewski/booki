/* Graph executor tests.
 *
 * 1. A trivial one-op graph (matmul) gives the same result as a direct
 *    booki_matmul_f16 call.
 * 2. A composed pre-norm transformer block built via the graph API
 *    matches the same block computed directly with the kernels. This is
 *    the test that proves the executor can serve as the running engine
 *    for a real model once the GGUF model-to-graph compiler lands.
 */

#include "booki_graph.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } \
} while (0)

static booki_tensor mk2(booki_arena* a, int64_t M, int64_t N) {
    int64_t s[2] = { M, N }; return booki_tensor_arena(a, BOOKI_DTYPE_F16, 2, s);
}
static booki_tensor mk1(booki_arena* a, int64_t N) {
    int64_t s[1] = { N }; return booki_tensor_arena(a, BOOKI_DTYPE_F16, 1, s);
}

static void fill_random(booki_tensor* t, uint64_t seed) {
    uint16_t* p = (uint16_t*)t->data;
    int64_t n = booki_tensor_elements(t);
    uint64_t s = seed;
    for (int64_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        float f = ((s >> 32) & 0xffff) / 65535.0f * 0.2f - 0.1f;
        p[i] = booki_f32_to_f16(f);
    }
}

static int test_one_matmul(void) {
    booki_arena* ws = booki_arena_create(2 << 20);
    CHECK(ws, "ws");

    booki_tensor A = mk2(ws, 3, 4); fill_random(&A, 1);
    booki_tensor B = mk2(ws, 4, 5); fill_random(&B, 2);

    /* Direct call as the oracle. */
    booki_tensor C_direct = mk2(ws, 3, 5);
    CHECK(booki_matmul_f16(&A, &B, &C_direct) == 0, "direct matmul");

    /* Graph version. */
    booki_arena* gws = booki_arena_create(2 << 20);
    booki_graph* g = booki_graph_create(gws, NULL);
    CHECK(g, "graph");

    int64_t s_a[2] = { 3, 4 };
    int64_t s_b[2] = { 4, 5 };
    int a_id = booki_graph_input(g, "A", BOOKI_DTYPE_F16, 2, s_a);
    int b_id = booki_graph_input(g, "B", BOOKI_DTYPE_F16, 2, s_b);
    int m_id = booki_graph_matmul(g, a_id, b_id);
    booki_graph_set_output(g, m_id);
    booki_graph_bind(g, a_id, &A);
    booki_graph_bind(g, b_id, &B);

    booki_tensor C_graph;
    CHECK(booki_graph_run(g, &C_graph) == 0,
          "graph_run: %s", booki_graph_last_error(g));

    uint16_t* dp = (uint16_t*)C_direct.data;
    uint16_t* gp = (uint16_t*)C_graph.data;
    for (int i = 0; i < 15; ++i)
        CHECK(fabsf(booki_f16_to_f32(dp[i]) - booki_f16_to_f32(gp[i])) < 5e-3f,
              "matmul diverges at %d", i);

    booki_graph_destroy(g);
    booki_arena_destroy(gws);
    booki_arena_destroy(ws);
    return 0;
}

/* Compute one pre-norm transformer block by direct kernel calls. */
static void compose_block_direct(
    const booki_tensor* x_in,
    const booki_tensor* nw,    /* norm weight */
    const booki_tensor* Wq, const booki_tensor* Wk, const booki_tensor* Wv,
    const booki_tensor* Wo,
    const booki_tensor* Wg, const booki_tensor* Wu, const booki_tensor* Wd,
    int heads, booki_arena* ws, booki_tensor* out)
{
    int64_t S = x_in->shape[0], H = x_in->shape[1];
    int64_t FF = Wg->shape[1];
    int64_t sX[2] = { S, H };
    int64_t sF[2] = { S, FF };

    booki_tensor x = mk2(ws, S, H);
    memcpy(x.data, x_in->data, x_in->nbytes);

    booki_tensor norm  = booki_tensor_arena(ws, BOOKI_DTYPE_F16, 2, sX);
    booki_tensor q     = booki_tensor_arena(ws, BOOKI_DTYPE_F16, 2, sX);
    booki_tensor k     = booki_tensor_arena(ws, BOOKI_DTYPE_F16, 2, sX);
    booki_tensor v     = booki_tensor_arena(ws, BOOKI_DTYPE_F16, 2, sX);
    booki_tensor attn  = booki_tensor_arena(ws, BOOKI_DTYPE_F16, 2, sX);
    booki_tensor wo_o  = booki_tensor_arena(ws, BOOKI_DTYPE_F16, 2, sX);
    booki_tensor gate  = booki_tensor_arena(ws, BOOKI_DTYPE_F16, 2, sF);
    booki_tensor up    = booki_tensor_arena(ws, BOOKI_DTYPE_F16, 2, sF);
    booki_tensor swi   = booki_tensor_arena(ws, BOOKI_DTYPE_F16, 2, sF);
    booki_tensor down  = booki_tensor_arena(ws, BOOKI_DTYPE_F16, 2, sX);

    booki_arena* scratch = booki_arena_create(4 << 20);

    booki_rmsnorm_f16(&x, nw, 1e-6f, &norm);
    booki_matmul_f16(&norm, Wq, &q);
    booki_matmul_f16(&norm, Wk, &k);
    booki_matmul_f16(&norm, Wv, &v);
    booki_multihead_attention_f16(&q, &k, &v, heads, scratch, &attn);
    booki_matmul_f16(&attn, Wo, &wo_o);
    booki_add_f16(&x, &wo_o, &x);
    booki_rmsnorm_f16(&x, nw, 1e-6f, &norm);
    booki_matmul_f16(&norm, Wg, &gate);
    booki_silu_f16(&gate, &gate);
    booki_matmul_f16(&norm, Wu, &up);
    booki_mul_f16(&gate, &up, &swi);
    booki_matmul_f16(&swi, Wd, &down);
    booki_add_f16(&x, &down, &x);

    booki_arena_destroy(scratch);
    memcpy(out->data, x.data, x.nbytes);
}

static int test_compose_block(void) {
    const int64_t S = 8, H = 16, FF = 32;
    const int heads = 2;

    booki_arena* setup = booki_arena_create(8 << 20);
    booki_tensor x_in = mk2(setup, S, H);             fill_random(&x_in, 1);
    booki_tensor nw   = mk1(setup, H);                fill_random(&nw,   2);
    booki_tensor Wq   = mk2(setup, H, H);             fill_random(&Wq,   3);
    booki_tensor Wk   = mk2(setup, H, H);             fill_random(&Wk,   4);
    booki_tensor Wv   = mk2(setup, H, H);             fill_random(&Wv,   5);
    booki_tensor Wo   = mk2(setup, H, H);             fill_random(&Wo,   6);
    booki_tensor Wg   = mk2(setup, H, FF);            fill_random(&Wg,   7);
    booki_tensor Wu   = mk2(setup, H, FF);            fill_random(&Wu,   8);
    booki_tensor Wd   = mk2(setup, FF, H);            fill_random(&Wd,   9);

    /* Oracle. */
    booki_arena* ows = booki_arena_create(8 << 20);
    booki_tensor oracle = mk2(ows, S, H);
    compose_block_direct(&x_in, &nw, &Wq, &Wk, &Wv, &Wo, &Wg, &Wu, &Wd,
                         heads, ows, &oracle);

    /* Graph. */
    booki_arena* gws = booki_arena_create(8 << 20);
    booki_graph* g = booki_graph_create(gws, NULL);

    int64_t sX[2] = { S, H };
    int x_id  = booki_graph_input(g, "x", BOOKI_DTYPE_F16, 2, sX);
    int nw_id = booki_graph_const(g, &nw);
    int Wq_id = booki_graph_const(g, &Wq);
    int Wk_id = booki_graph_const(g, &Wk);
    int Wv_id = booki_graph_const(g, &Wv);
    int Wo_id = booki_graph_const(g, &Wo);
    int Wg_id = booki_graph_const(g, &Wg);
    int Wu_id = booki_graph_const(g, &Wu);
    int Wd_id = booki_graph_const(g, &Wd);

    int norm1 = booki_graph_rmsnorm(g, x_id, nw_id, 1e-6f);
    int q     = booki_graph_matmul (g, norm1, Wq_id);
    int k     = booki_graph_matmul (g, norm1, Wk_id);
    int v     = booki_graph_matmul (g, norm1, Wv_id);
    int attn  = booki_graph_attention(g, q, k, v, heads);
    int wo    = booki_graph_matmul (g, attn, Wo_id);
    int x2    = booki_graph_add    (g, x_id, wo);
    int norm2 = booki_graph_rmsnorm(g, x2, nw_id, 1e-6f);
    int gate  = booki_graph_matmul (g, norm2, Wg_id);
    int gact  = booki_graph_silu   (g, gate);
    int up    = booki_graph_matmul (g, norm2, Wu_id);
    int swi   = booki_graph_mul    (g, gact, up);
    int down  = booki_graph_matmul (g, swi, Wd_id);
    int out_n = booki_graph_add    (g, x2, down);

    booki_graph_set_output(g, out_n);
    booki_graph_bind(g, x_id, &x_in);

    booki_tensor result;
    CHECK(booki_graph_run(g, &result) == 0,
          "graph_run: %s", booki_graph_last_error(g));

    uint16_t* op = (uint16_t*)oracle.data;
    uint16_t* rp = (uint16_t*)result.data;
    int64_t n = booki_tensor_elements(&oracle);
    float worst = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float d = fabsf(booki_f16_to_f32(op[i]) - booki_f16_to_f32(rp[i]));
        if (d > worst) worst = d;
    }
    CHECK(worst < 5e-2f, "graph block diverges by %.4f", worst);

    booki_graph_destroy(g);
    booki_arena_destroy(setup);
    booki_arena_destroy(ows);
    booki_arena_destroy(gws);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_one_matmul();
    rc |= test_compose_block();
    if (rc == 0) printf("graph ok (%d nodes built)\n", booki_graph_node_count(NULL));
    return rc;
}

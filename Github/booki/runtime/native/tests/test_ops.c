/* Conformance for RMSNorm, SiLU, GELU, add/mul, softmax, embedding, attention.
 * Each test computes the expected output in scalar fp32 and compares against
 * the runtime's fp16 result with a generous tolerance (fp16 quantization
 * limits how tight we can pull this without per-op golden tables). */

#include "booki_native.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
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
static booki_tensor mk1_i64(booki_arena* a, int64_t N) {
    int64_t s[1] = { N }; return booki_tensor_arena(a, BOOKI_DTYPE_I64, 1, s);
}

static void fill(booki_tensor* t, const float* src) {
    uint16_t* p = (uint16_t*)t->data;
    int64_t n = booki_tensor_elements(t);
    for (int64_t i = 0; i < n; ++i) p[i] = booki_f32_to_f16(src[i]);
}
static float read(booki_tensor* t, int64_t i) {
    return booki_f16_to_f32(((uint16_t*)t->data)[i]);
}

static int test_rmsnorm(booki_arena* a) {
    /* x = [1, 2, 3, 4] with weight = [1, 1, 1, 1] and eps = 1e-6 */
    float xv[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float wv[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    booki_tensor x = mk2(a, 1, 4); fill(&x, xv);
    booki_tensor w = mk1(a, 4);    fill(&w, wv);
    booki_tensor y = mk2(a, 1, 4);

    CHECK(booki_rmsnorm_f16(&x, &w, 1e-6f, &y) == 0, "rmsnorm");
    /* mean(x^2) = (1+4+9+16)/4 = 7.5 → rms = sqrt(7.5) ≈ 2.7386 */
    float rms = sqrtf(7.5f);
    for (int i = 0; i < 4; ++i) {
        float want = xv[i] / rms;
        float got = read(&y, i);
        CHECK(fabsf(got - want) < 5e-3f, "rmsnorm[%d] want %.4f got %.4f", i, want, got);
    }
    return 0;
}

static int test_silu(booki_arena* a) {
    float xv[6] = { -2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 5.0f };
    booki_tensor x = mk2(a, 1, 6); fill(&x, xv);
    booki_tensor y = mk2(a, 1, 6);
    CHECK(booki_silu_f16(&x, &y) == 0, "silu");
    for (int i = 0; i < 6; ++i) {
        float want = xv[i] / (1.0f + expf(-xv[i]));
        float got = read(&y, i);
        CHECK(fabsf(got - want) < 5e-3f, "silu[%d] want %.4f got %.4f", i, want, got);
    }
    return 0;
}

static int test_gelu(booki_arena* a) {
    float xv[4] = { -2.0f, 0.0f, 1.0f, 3.0f };
    booki_tensor x = mk2(a, 1, 4); fill(&x, xv);
    booki_tensor y = mk2(a, 1, 4);
    CHECK(booki_gelu_f16(&x, &y) == 0, "gelu");
    const float k = 0.7978845608028654f;
    for (int i = 0; i < 4; ++i) {
        float ix = k * (xv[i] + 0.044715f * xv[i] * xv[i] * xv[i]);
        float want = 0.5f * xv[i] * (1.0f + tanhf(ix));
        float got = read(&y, i);
        CHECK(fabsf(got - want) < 5e-3f, "gelu[%d] want %.4f got %.4f", i, want, got);
    }
    return 0;
}

static int test_add_mul(booki_arena* a) {
    float av[4] = { 1, 2, 3, 4 };
    float bv[4] = { 5, 6, 7, 8 };
    booki_tensor A = mk2(a, 1, 4); fill(&A, av);
    booki_tensor B = mk2(a, 1, 4); fill(&B, bv);
    booki_tensor S = mk2(a, 1, 4);
    booki_tensor P = mk2(a, 1, 4);
    CHECK(booki_add_f16(&A, &B, &S) == 0, "add");
    CHECK(booki_mul_f16(&A, &B, &P) == 0, "mul");
    for (int i = 0; i < 4; ++i) {
        CHECK(fabsf(read(&S, i) - (av[i] + bv[i])) < 1e-2f, "add[%d]", i);
        CHECK(fabsf(read(&P, i) - (av[i] * bv[i])) < 1e-2f, "mul[%d]", i);
    }
    return 0;
}

static int test_softmax(booki_arena* a) {
    /* Two rows: [1, 2, 3], [0, 0, 0]. */
    float xv[6] = { 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f };
    booki_tensor x = mk2(a, 2, 3); fill(&x, xv);
    booki_tensor y = mk2(a, 2, 3);
    CHECK(booki_softmax_f16(&x, &y) == 0, "softmax");

    /* Row 1 sums to 1, uniform 1/3. */
    float row1_sum = read(&y, 3) + read(&y, 4) + read(&y, 5);
    CHECK(fabsf(row1_sum - 1.0f) < 1e-2f, "row1 sum %.4f", row1_sum);
    for (int i = 3; i < 6; ++i)
        CHECK(fabsf(read(&y, i) - (1.0f / 3.0f)) < 1e-2f, "row1 uniform");

    /* Row 0: classic softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]. */
    float want[3] = { 0.09003057f, 0.24472847f, 0.66524096f };
    for (int i = 0; i < 3; ++i)
        CHECK(fabsf(read(&y, i) - want[i]) < 1e-2f, "row0[%d]", i);
    return 0;
}

static int test_embedding(booki_arena* a) {
    /* Vocabulary [3 ids × 2 dims], pick ids [2, 0]. */
    float wv[6] = { 10, 20, 30, 40, 50, 60 };
    int64_t ids_src[2] = { 2, 0 };

    booki_tensor w = mk2(a, 3, 2); fill(&w, wv);
    booki_tensor ids = mk1_i64(a, 2);
    memcpy(ids.data, ids_src, sizeof(ids_src));
    booki_tensor out = mk2(a, 2, 2);

    CHECK(booki_embedding_f16(&ids, &w, &out) == 0, "embedding");
    /* out[0] = w[2] = [50, 60], out[1] = w[0] = [10, 20] */
    CHECK(fabsf(read(&out, 0) - 50.0f) < 1e-1f, "out[0,0]");
    CHECK(fabsf(read(&out, 1) - 60.0f) < 1e-1f, "out[0,1]");
    CHECK(fabsf(read(&out, 2) - 10.0f) < 1e-1f, "out[1,0]");
    CHECK(fabsf(read(&out, 3) - 20.0f) < 1e-1f, "out[1,1]");
    return 0;
}

static int test_attention(booki_arena* a) {
    /* Tiny attention: M=2 queries, N=3 keys, D=4. Identity-ish setup so
     * results stay numerically clean: Q == K means softmax(QK^T) is sharp
     * along the diagonal, so out ≈ V[corresponding rows]. */
    float qv[8]  = { 1,0,0,0,  0,1,0,0 };
    float kv[12] = { 1,0,0,0,  0,1,0,0,  0,0,1,0 };
    float vv[12] = { 10,20,30,40,  50,60,70,80,  90,100,110,120 };

    booki_tensor Q = mk2(a, 2, 4); fill(&Q, qv);
    booki_tensor K = mk2(a, 3, 4); fill(&K, kv);
    booki_tensor V = mk2(a, 3, 4); fill(&V, vv);
    booki_tensor Y = mk2(a, 2, 4);

    /* Sub-arena for the kernel's scratch (KT, scores). */
    booki_arena* scratch = booki_arena_create(64 * 1024);
    CHECK(scratch, "scratch");

    CHECK(booki_attention_f16(&Q, &K, &V, scratch, &Y) == 0, "attention");

    /* Row 0 should be ~closest to V[0], row 1 ~closest to V[1].
     * Softmax with d_k=4 scale isn't fully peaked so we use loose tolerance. */
    float r0_first = read(&Y, 0);
    float r1_first = read(&Y, 4);
    CHECK(r0_first > 0.0f && r0_first < 50.0f,
          "row0[0] sanity %.4f", r0_first);
    CHECK(r1_first > 20.0f && r1_first < 80.0f,
          "row1[0] sanity %.4f", r1_first);

    booki_arena_destroy(scratch);
    return 0;
}

int main(void) {
    booki_arena* a = booki_arena_create(2 << 20);
    if (!a) { fprintf(stderr, "arena\n"); return 1; }

    int rc = 0;
    rc |= test_rmsnorm(a);    booki_arena_reset(a);
    rc |= test_silu(a);       booki_arena_reset(a);
    rc |= test_gelu(a);       booki_arena_reset(a);
    rc |= test_add_mul(a);    booki_arena_reset(a);
    rc |= test_softmax(a);    booki_arena_reset(a);
    rc |= test_embedding(a);  booki_arena_reset(a);
    rc |= test_attention(a);  booki_arena_reset(a);

    booki_arena_destroy(a);
    if (rc == 0) printf("ops ok (backend: %s)\n",
                        booki_backend_describe(booki_backend_active()));
    return rc;
}

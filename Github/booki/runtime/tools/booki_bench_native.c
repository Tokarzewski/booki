/* Micro-benchmark for the native runtime's ops.
 *
 * Same JSON output shape as booki_bench so the runtime-ci.yml regression
 * gate can compare runs over time. Each op gets a dedicated benchmark with
 * a Kokoro-shaped tensor size so the numbers translate to model-scale
 * inference once the full executor lands.
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

typedef struct {
    const char* name;
    int64_t M, N, K;
} matmul_case;

/* Sizes from Kokoro 1.1 (attention QKV proj, FFN gate, LM head). */
static const matmul_case CASES[] = {
    { "matmul_f16_256x256x256",   256, 256,  256 },
    { "matmul_f16_512x512x512",   512, 512,  512 },
    { "matmul_f16_1024x1024x1024",1024,1024, 1024 },
};

static void bench_unary(const char* name, booki_arena* arena,
                        int (*fn)(const booki_tensor*, booki_tensor*),
                        int64_t rows, int64_t cols, int iters,
                        int json, int* first) {
    int64_t s[2] = { rows, cols };
    booki_tensor x = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s);
    booki_tensor y = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s);
    int64_t n = rows * cols;
    for (int64_t i = 0; i < n; ++i) ((uint16_t*)x.data)[i] = (uint16_t)((i * 17) & 0x3fff);
    fn(&x, &y);  /* warmup */
    double t0 = now_ms();
    for (int i = 0; i < iters; ++i) fn(&x, &y);
    double dt = (now_ms() - t0) / iters;
    double tokens_per_sec = 1000.0 / dt;
    if (json) {
        printf("%s    { \"name\": \"%s\", \"ms_per_token\": %.4f, "
               "\"tokens_per_sec\": %.2f }",
               *first ? "" : ",\n", name, dt, tokens_per_sec);
        *first = 0;
    } else {
        printf("%-30s  %.4f ms   %.2f ops/s\n", name, dt, tokens_per_sec);
    }
}

static void bench_binary(const char* name, booki_arena* arena,
                         int (*fn)(const booki_tensor*, const booki_tensor*, booki_tensor*),
                         int64_t rows, int64_t cols, int iters,
                         int json, int* first) {
    int64_t s[2] = { rows, cols };
    booki_tensor a = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s);
    booki_tensor b = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s);
    booki_tensor c = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s);
    int64_t n = rows * cols;
    for (int64_t i = 0; i < n; ++i) {
        ((uint16_t*)a.data)[i] = (uint16_t)((i * 13) & 0x3fff);
        ((uint16_t*)b.data)[i] = (uint16_t)((i * 19) & 0x3fff);
    }
    fn(&a, &b, &c);
    double t0 = now_ms();
    for (int i = 0; i < iters; ++i) fn(&a, &b, &c);
    double dt = (now_ms() - t0) / iters;
    double tokens_per_sec = 1000.0 / dt;
    if (json) {
        printf("%s    { \"name\": \"%s\", \"ms_per_token\": %.4f, "
               "\"tokens_per_sec\": %.2f }",
               *first ? "" : ",\n", name, dt, tokens_per_sec);
        *first = 0;
    } else {
        printf("%-30s  %.4f ms   %.2f ops/s\n", name, dt, tokens_per_sec);
    }
}

int main(int argc, char** argv) {
    int iters = 3;
    int emit_json = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json")) emit_json = 1;
    }

    /* Arena big enough for the largest case (M*K + K*N + M*N halves) */
    size_t cap = 4ull * 1024 * 1024 * sizeof(uint16_t) * 3;
    booki_arena* arena = booki_arena_create(cap);
    if (!arena) { fprintf(stderr, "arena alloc failed\n"); return 1; }

    /* Thread count is read from BOOKI_THREADS by the threading layer on
     * first use; calling booki_get_threads() here pins it and lets us
     * surface the value in the bench output. */
    int threads = booki_get_threads();

    if (emit_json) {
        printf("{\n");
        printf("  \"backend\": \"native-%s\",\n", booki_backend_describe(booki_backend_active()));
        printf("  \"runtime\": \"%s\",\n", booki_native_version());
        printf("  \"threads\": %d,\n", threads);
        printf("  \"benchmarks\": [\n");
    } else {
        printf("backend          native-%s\n", booki_backend_describe(booki_backend_active()));
        printf("runtime          %s\n", booki_native_version());
        printf("threads          %d\n", threads);
    }

    int n_cases = (int)(sizeof(CASES) / sizeof(CASES[0]));
    int first = 1;
    for (int i = 0; i < n_cases; ++i) {
        const matmul_case* c = &CASES[i];
        booki_arena_reset(arena);

        int64_t sa[2] = { c->M, c->K };
        int64_t sb[2] = { c->K, c->N };
        int64_t sc[2] = { c->M, c->N };
        booki_tensor A = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, sa);
        booki_tensor B = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, sb);
        booki_tensor C = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, sc);
        if (!A.data || !B.data || !C.data) { fprintf(stderr, "tensor alloc\n"); return 1; }

        /* Deterministic non-zero fill, no allocation. */
        for (int64_t k = 0; k < c->M * c->K; ++k) ((uint16_t*)A.data)[k] = (uint16_t)((k * 37 + 11) & 0x7fff);
        for (int64_t k = 0; k < c->K * c->N; ++k) ((uint16_t*)B.data)[k] = (uint16_t)((k * 23 + 5)  & 0x7fff);

        /* Warm up */
        booki_matmul_f16(&A, &B, &C);

        double t0 = now_ms();
        for (int j = 0; j < iters; ++j) booki_matmul_f16(&A, &B, &C);
        double dt = (now_ms() - t0) / iters;

        double flops_per_op = 2.0 * (double)c->M * (double)c->N * (double)c->K;
        double gflops = (flops_per_op / 1e9) / (dt / 1000.0);
        double ms_per_token = dt;   /* one op = one "token" for the gate */
        double tokens_per_sec = 1000.0 / dt;

        if (emit_json) {
            printf("%s    { \"name\": \"%s\", \"ms_per_token\": %.4f, "
                   "\"tokens_per_sec\": %.2f, \"gflops\": %.2f }",
                   first ? "" : ",\n", c->name, ms_per_token, tokens_per_sec, gflops);
            first = 0;
        } else {
            printf("%-30s  %.3f ms   %.2f GFLOPS\n", c->name, dt, gflops);
        }
    }

    /* Element-wise / activation / norm ops at Kokoro-scale tensors. */
    booki_arena_reset(arena);
    bench_unary  ("silu_f16_512x1024",  arena, booki_silu_f16,  512, 1024, iters, emit_json, &first);
    bench_unary  ("gelu_f16_512x1024",  arena, booki_gelu_f16,  512, 1024, iters, emit_json, &first);
    bench_binary ("add_f16_512x1024",   arena, booki_add_f16,   512, 1024, iters, emit_json, &first);
    bench_binary ("mul_f16_512x1024",   arena, booki_mul_f16,   512, 1024, iters, emit_json, &first);
    bench_unary  ("softmax_f16_64x4096",arena, booki_softmax_f16, 64, 4096, iters, emit_json, &first);

    /* RMSNorm needs a weight tensor so it doesn't fit bench_unary. */
    {
        booki_arena_reset(arena);
        int64_t s[2] = { 512, 1024 };
        int64_t ws[1] = { 1024 };
        booki_tensor x = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s);
        booki_tensor w = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 1, ws);
        booki_tensor y = booki_tensor_arena(arena, BOOKI_DTYPE_F16, 2, s);
        for (int64_t i = 0; i < 512 * 1024; ++i) ((uint16_t*)x.data)[i] = (uint16_t)((i * 11) & 0x3fff);
        for (int64_t i = 0; i < 1024; ++i) ((uint16_t*)w.data)[i] = (uint16_t)((i * 7) & 0x3fff);
        booki_rmsnorm_f16(&x, &w, 1e-6f, &y);
        double t0 = now_ms();
        for (int i = 0; i < iters; ++i) booki_rmsnorm_f16(&x, &w, 1e-6f, &y);
        double dt = (now_ms() - t0) / iters;
        if (emit_json) {
            printf("%s    { \"name\": \"rmsnorm_f16_512x1024\", \"ms_per_token\": %.4f, "
                   "\"tokens_per_sec\": %.2f }",
                   first ? "" : ",\n", dt, 1000.0 / dt);
            first = 0;
        } else {
            printf("%-30s  %.4f ms\n", "rmsnorm_f16_512x1024", dt);
        }
    }

    if (emit_json) {
        printf("\n  ]\n}\n");
    }

    booki_arena_destroy(arena);
    return 0;
}

/* booki_bench — micro-benchmark harness. Emits the JSON shape
 * runtime-ci.yml expects: { "benchmarks": [ { "name", "ms_per_token",
 * "tokens_per_sec" }, ... ] }.
 *
 * Inputs are fabricated (deterministic token sequence, zero style) so the
 * bench is hermetic and cross-comparable between runs. Real text /
 * phonemization happens in the Android app, not here.
 */

#include "booki_baseline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BENCH_DEFAULT_ITERS
#define BENCH_DEFAULT_ITERS 5
#endif

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1e6;
}

static void usage(void) {
    fprintf(stderr,
        "usage: booki_bench --model <path> [--iters N] [--tokens N] [--json]\n");
}

int main(int argc, char** argv) {
    const char* model_path = getenv("BOOKI_MODEL");
    int  iters  = BENCH_DEFAULT_ITERS;
    int  tokens = 128;
    int  emit_json = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json")) emit_json = 1;
        else { usage(); return 2; }
    }

    if (!model_path) {
        fprintf(stderr, "error: --model or BOOKI_MODEL required\n");
        return 2;
    }

    booki_session* s = booki_open(model_path);
    if (!s) { fprintf(stderr, "open failed: %s\n", booki_last_error()); return 1; }

    int64_t* token_ids = malloc(tokens * sizeof(int64_t));
    float*   style     = calloc(256, sizeof(float));
    if (!token_ids || !style) { fprintf(stderr, "oom\n"); return 1; }
    for (int i = 0; i < tokens; ++i) token_ids[i] = (i % 100) + 1;

    booki_input in = {
        .tokens      = token_ids,
        .token_count = (size_t)tokens,
        .style       = style,
        .style_dim   = 256,
        .speed       = 1.0f,
    };

    /* Warm-up — first iter triggers lazy graph optimization. */
    booki_output warm = booki_run(s, &in);
    if (warm.sample_count == 0) {
        fprintf(stderr, "warmup failed: %s\n", booki_last_error());
        return 1;
    }
    booki_free(warm.samples);

    double total_ms = 0.0;
    size_t total_samples = 0;
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        booki_output out = booki_run(s, &in);
        double dt = now_ms() - t0;
        if (out.sample_count == 0) {
            fprintf(stderr, "iter %d failed: %s\n", i, booki_last_error());
            return 1;
        }
        total_ms += dt;
        total_samples += out.sample_count;
        booki_free(out.samples);
    }

    double ms_per_token  = total_ms / (double)(iters * tokens);
    double tokens_per_s  = (double)(iters * tokens) / (total_ms / 1000.0);
    double realtime_x    = ((double)total_samples / 24000.0) / (total_ms / 1000.0);

    if (emit_json) {
        printf(
            "{\n"
            "  \"backend\": \"%s\",\n"
            "  \"runtime\": \"%s\",\n"
            "  \"model\": \"%s\",\n"
            "  \"benchmarks\": [\n"
            "    { \"name\": \"synth_%dtok\", \"ms_per_token\": %.4f, "
            "\"tokens_per_sec\": %.2f, \"realtime_factor\": %.2f }\n"
            "  ]\n"
            "}\n",
            booki_backend_name(), booki_runtime_version(), model_path,
            tokens, ms_per_token, tokens_per_s, realtime_x);
    } else {
        printf("backend          %s\n", booki_backend_name());
        printf("model            %s\n", model_path);
        printf("iters            %d\n", iters);
        printf("tokens/iter      %d\n", tokens);
        printf("ms per token     %.4f\n", ms_per_token);
        printf("tokens/sec       %.2f\n", tokens_per_s);
        printf("realtime factor  %.2f\n", realtime_x);
    }

    free(token_ids);
    free(style);
    booki_close(s);
    return 0;
}

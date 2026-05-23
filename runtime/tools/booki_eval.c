/* booki_eval — golden-output diff harness.
 *
 * Runs one canonical inference and either:
 *   --record  : writes the first N output samples to <golden> (use to
 *               regenerate the fixture after a known-good change)
 *   default   : diffs against <golden>; exits non-zero if any sample
 *               differs by more than --tolerance.
 *
 * Goldens are committed under runtime/golden/. Only the first 1024 samples
 * are checked — enough to catch regressions, small enough to commit (4 KB).
 */

#include "booki_baseline.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GOLDEN_SAMPLES 1024

static void usage(void) {
    fprintf(stderr,
        "usage: booki_eval --model <path> --golden <path> [--tolerance F] "
        "[--record] [--report <json>]\n");
}

int main(int argc, char** argv) {
    const char* model_path  = NULL;
    const char* golden_path = NULL;
    const char* report_path = NULL;
    float       tol         = 1e-3f;
    int         record      = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--golden") && i + 1 < argc) golden_path = argv[++i];
        else if (!strcmp(argv[i], "--report") && i + 1 < argc) report_path = argv[++i];
        else if (!strcmp(argv[i], "--tolerance") && i + 1 < argc) tol = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--record")) record = 1;
        else { usage(); return 2; }
    }
    if (!model_path || !golden_path) { usage(); return 2; }

    booki_session* s = booki_open(model_path);
    if (!s) { fprintf(stderr, "open: %s\n", booki_last_error()); return 1; }

    /* Canonical input — must stay byte-identical across runtimes for the
     * diff to be meaningful. */
    int64_t tokens[64];
    for (int i = 0; i < 64; ++i) tokens[i] = (i % 50) + 1;
    float style[256] = {0};
    booki_input in = { tokens, 64, style, 256, 1.0f };

    booki_output out = booki_run(s, &in);
    if (out.sample_count == 0) {
        fprintf(stderr, "run: %s\n", booki_last_error());
        return 1;
    }
    size_t n = out.sample_count < GOLDEN_SAMPLES ? out.sample_count : GOLDEN_SAMPLES;

    if (record) {
        FILE* f = fopen(golden_path, "wb");
        if (!f) { perror("fopen"); return 1; }
        fwrite(out.samples, sizeof(float), n, f);
        fclose(f);
        printf("wrote %zu samples to %s\n", n, golden_path);
        booki_free(out.samples);
        booki_close(s);
        return 0;
    }

    FILE* f = fopen(golden_path, "rb");
    if (!f) { fprintf(stderr, "open golden: %s\n", golden_path); return 1; }
    float* golden = malloc(GOLDEN_SAMPLES * sizeof(float));
    size_t read_n = fread(golden, sizeof(float), GOLDEN_SAMPLES, f);
    fclose(f);
    if (read_n < n) n = read_n;

    float max_abs = 0.0f, mean_abs = 0.0f;
    size_t worst_idx = 0, fails = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(out.samples[i] - golden[i]);
        if (d > max_abs) { max_abs = d; worst_idx = i; }
        mean_abs += d;
        if (d > tol) ++fails;
    }
    mean_abs /= (float)n;

    printf("samples_compared %zu\n", n);
    printf("max_abs_diff     %.6e (idx %zu)\n", max_abs, worst_idx);
    printf("mean_abs_diff    %.6e\n", mean_abs);
    printf("tolerance        %.6e\n", tol);
    printf("samples_over_tol %zu\n", fails);
    printf("verdict          %s\n", fails == 0 ? "PASS" : "FAIL");

    if (report_path) {
        FILE* r = fopen(report_path, "w");
        if (r) {
            fprintf(r,
                "{\n"
                "  \"summary\": { \"verdict\": \"%s\", \"max_abs_diff\": %.6e, "
                "\"mean_abs_diff\": %.6e, \"samples_over_tol\": %zu, "
                "\"tolerance\": %.6e, \"samples_compared\": %zu }\n"
                "}\n",
                fails == 0 ? "PASS" : "FAIL", max_abs, mean_abs, fails, tol, n);
            fclose(r);
        }
    }

    free(golden);
    booki_free(out.samples);
    booki_close(s);
    return fails == 0 ? 0 : 1;
}

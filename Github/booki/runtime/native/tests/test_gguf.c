/* GGUF reader tests.
 *
 * Builds a tiny synthetic GGUF v3 file in a temp path, runs the reader on
 * it, and verifies metadata + tensor data come back byte-identical. Avoids
 * needing a real Kokoro GGUF on disk — this is purely the format contract.
 */

#include "booki_native.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } \
} while (0)

#define GGUF_TYPE_UINT32 4
#define GGUF_TYPE_STRING 8
#define GGML_TYPE_F32 0
#define GGML_TYPE_F16 1
#define ALIGN_TO 32

typedef struct {
    FILE* f;
    size_t pos;
} writer;

static void w_bytes(writer* w, const void* p, size_t n) {
    fwrite(p, 1, n, w->f); w->pos += n;
}
static void w_u32(writer* w, uint32_t v) { w_bytes(w, &v, 4); }
static void w_u64(writer* w, uint64_t v) { w_bytes(w, &v, 8); }
static void w_str(writer* w, const char* s) {
    uint64_t n = strlen(s);
    w_u64(w, n);
    w_bytes(w, s, n);
}
static void w_pad(writer* w) {
    size_t pad = (ALIGN_TO - (w->pos % ALIGN_TO)) % ALIGN_TO;
    static const char zeros[ALIGN_TO] = {0};
    if (pad) w_bytes(w, zeros, pad);
}

static int build_file(const char* path) {
    writer w;
    w.f = fopen(path, "wb");
    if (!w.f) return 1;
    w.pos = 0;

    /* Header */
    w_bytes(&w, "GGUF", 4);
    w_u32(&w, 3);                       /* version */
    w_u64(&w, 2);                       /* tensor_count */
    w_u64(&w, 1);                       /* meta_count */

    /* One metadata kv: arch=test-model */
    w_str(&w, "general.architecture");
    w_u32(&w, GGUF_TYPE_STRING);
    w_str(&w, "test-model");

    /* Tensor 1: "weight" — F32 [2, 3] */
    w_str(&w, "weight");
    w_u32(&w, 2);                       /* rank */
    /* GGUF stores shape innermost-first. PyTorch/booki present [2,3] meaning
     * 2 rows, 3 cols. Innermost is cols → 3, outer is rows → 2. */
    w_u64(&w, 3); w_u64(&w, 2);
    w_u32(&w, GGML_TYPE_F32);
    w_u64(&w, 0);                       /* offset */

    /* Tensor 2: "bias" — F16 [4] */
    w_str(&w, "bias");
    w_u32(&w, 1);
    w_u64(&w, 4);
    w_u32(&w, GGML_TYPE_F16);
    w_u64(&w, 64);                      /* offset (after weight + padding) */

    /* Pad to ALIGN_TO before data section. */
    w_pad(&w);

    /* Tensor data — first 6 floats for "weight", then 8 bytes alignment,
     * then 4 fp16 halves for "bias". */
    float w_data[6] = { 1, 2, 3, 4, 5, 6 };
    w_bytes(&w, w_data, sizeof(w_data));
    /* Pad up to offset 64 in the data section. We've written 24 bytes so
     * far; we need 64-24 = 40 zero bytes. */
    static const char zeros[64] = {0};
    w_bytes(&w, zeros, 40);

    uint16_t bias_data[4] = {
        booki_f32_to_f16(-1.0f),
        booki_f32_to_f16(0.0f),
        booki_f32_to_f16(0.5f),
        booki_f32_to_f16(2.5f),
    };
    w_bytes(&w, bias_data, sizeof(bias_data));

    fclose(w.f);
    return 0;
}

int main(void) {
    char path[] = "/tmp/booki_test_gguf_XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp failed");
    close(fd);

    int rc = build_file(path);
    CHECK(rc == 0, "build_file");

    char err[256] = {0};
    booki_model* m = booki_model_open(path, err, sizeof(err));
    CHECK(m != NULL, "model_open: %s", err);
    CHECK(booki_model_tensor_count(m) == 2, "tensor count");
    CHECK(strcmp(booki_model_tensor_name(m, 0), "weight") == 0, "name 0");
    CHECK(strcmp(booki_model_tensor_name(m, 1), "bias") == 0,   "name 1");

    const char* arch = booki_model_meta_string(m, "general.architecture");
    CHECK(arch && strcmp(arch, "test-model") == 0,
          "arch metadata: %s", arch ? arch : "(null)");

    booki_tensor weight = {0};
    CHECK(booki_model_tensor(m, "weight", &weight) == 0, "find weight");
    CHECK(weight.dtype == BOOKI_DTYPE_F32, "weight dtype");
    CHECK(weight.rank == 2, "weight rank");
    CHECK(weight.shape[0] == 2 && weight.shape[1] == 3,
          "weight shape: [%lld, %lld]",
          (long long)weight.shape[0], (long long)weight.shape[1]);

    float* wp = (float*)weight.data;
    for (int i = 0; i < 6; ++i)
        CHECK(fabsf(wp[i] - (float)(i + 1)) < 1e-6f,
              "weight[%d] = %f", i, wp[i]);

    booki_tensor bias = {0};
    CHECK(booki_model_tensor(m, "bias", &bias) == 0, "find bias");
    CHECK(bias.dtype == BOOKI_DTYPE_F16, "bias dtype");
    CHECK(bias.shape[0] == 4, "bias shape");
    uint16_t* bp = (uint16_t*)bias.data;
    float wants[4] = { -1.0f, 0.0f, 0.5f, 2.5f };
    for (int i = 0; i < 4; ++i) {
        float v = booki_f16_to_f32(bp[i]);
        CHECK(fabsf(v - wants[i]) < 1e-2f, "bias[%d] = %f vs %f", i, v, wants[i]);
    }

    /* Lookup miss should return error and not crash. */
    booki_tensor missing = {0};
    CHECK(booki_model_tensor(m, "nope", &missing) != 0, "miss should fail");

    booki_model_close(m);

    /* Reject non-GGUF file. */
    FILE* bad = fopen(path, "wb");
    fputs("NOPE garbage", bad); fclose(bad);
    booki_model* bm = booki_model_open(path, err, sizeof(err));
    CHECK(bm == NULL, "should reject non-gguf");

    unlink(path);
    printf("gguf ok\n");
    return 0;
}

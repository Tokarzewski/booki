/* Booki runtime — public C surface.
 *
 * Today this header is backed by ONNX Runtime (see booki_baseline.c). When
 * #13 lands, the same surface is reimplemented by a custom engine. Anything
 * a consumer (booki_bench, booki_eval, the Android JNI) needs lives here;
 * the implementation .c file is replaceable.
 */
#ifndef BOOKI_RUNTIME_H
#define BOOKI_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct booki_session booki_session;

/* Synthesis inputs. Mirrors the Kokoro ONNX signature so the eventual
 * custom kernel keeps the same shape contract. */
typedef struct {
    const int64_t* tokens;     /* shape [T] */
    size_t         token_count;
    const float*   style;      /* shape [256] */
    size_t         style_dim;
    float          speed;      /* scalar */
} booki_input;

typedef struct {
    float*   samples;          /* malloc'd; caller frees with booki_free */
    size_t   sample_count;
    uint32_t sample_rate_hz;
} booki_output;

/* Returns NULL on failure; check booki_last_error() for the message. */
booki_session* booki_open(const char* model_path);
void           booki_close(booki_session* s);

/* Runs inference; on failure returns a zeroed output and last_error is set. */
booki_output   booki_run(booki_session* s, const booki_input* in);

void           booki_free(void* p);
const char*    booki_last_error(void);

/* Build provenance — useful for the bench JSON header. */
const char*    booki_backend_name(void);   /* "onnxruntime-1.20.1" / "booki-native-..." */
const char*    booki_runtime_version(void);

#ifdef __cplusplus
}
#endif
#endif

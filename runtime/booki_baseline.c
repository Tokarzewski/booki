/* Booki runtime — ONNX Runtime baseline.
 *
 * The whole engine in one file (mirroring antirez/ds4's shape). When #13
 * starts, this file gets replaced by a NEON-tuned implementation; the
 * booki_runtime.h surface and the CMake target stay put.
 */

#include "booki_baseline.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnxruntime_c_api.h"

#define BOOKI_VERSION "0.1.0-baseline"

/* ------------------------------------------------------------------------- */
/* Error reporting                                                           */
/* ------------------------------------------------------------------------- */

static _Thread_local char g_last_error[512] = "";

static void set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, ap);
    va_end(ap);
}

const char* booki_last_error(void) { return g_last_error; }

const char* booki_backend_name(void)    { return "onnxruntime-1.20.1"; }
const char* booki_runtime_version(void) { return BOOKI_VERSION; }

void booki_free(void* p) { free(p); }

/* ------------------------------------------------------------------------- */
/* Session                                                                   */
/* ------------------------------------------------------------------------- */

struct booki_session {
    const OrtApi*   api;
    OrtEnv*         env;
    OrtSession*     session;
    OrtMemoryInfo*  memory_info;
    OrtAllocator*   allocator;
    char**          input_names;
    size_t          input_count;
    char**          output_names;
    size_t          output_count;
};

static void session_free_names(char** names, size_t n) {
    if (!names) return;
    for (size_t i = 0; i < n; ++i) free(names[i]);
    free(names);
}

#define CHECK(expr) do { \
    OrtStatus* _s = (expr); \
    if (_s != NULL) { \
        set_error("%s", session->api->GetErrorMessage(_s)); \
        session->api->ReleaseStatus(_s); \
        booki_close(session); \
        return NULL; \
    } \
} while (0)

booki_session* booki_open(const char* model_path) {
    booki_session* session = calloc(1, sizeof(*session));
    if (!session) { set_error("oom"); return NULL; }

    session->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!session->api) { set_error("ORT API unavailable"); free(session); return NULL; }

    CHECK(session->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "booki", &session->env));

    OrtSessionOptions* opts = NULL;
    CHECK(session->api->CreateSessionOptions(&opts));
    session->api->SetIntraOpNumThreads(opts, 2);
    session->api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_BASIC);

    CHECK(session->api->CreateSession(session->env, model_path, opts, &session->session));
    session->api->ReleaseSessionOptions(opts);

    CHECK(session->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                            &session->memory_info));
    CHECK(session->api->GetAllocatorWithDefaultOptions(&session->allocator));

    CHECK(session->api->SessionGetInputCount(session->session, &session->input_count));
    CHECK(session->api->SessionGetOutputCount(session->session, &session->output_count));

    session->input_names  = calloc(session->input_count,  sizeof(char*));
    session->output_names = calloc(session->output_count, sizeof(char*));
    for (size_t i = 0; i < session->input_count; ++i) {
        char* name = NULL;
        CHECK(session->api->SessionGetInputName(session->session, i, session->allocator, &name));
        session->input_names[i] = strdup(name);
        session->api->AllocatorFree(session->allocator, name);
    }
    for (size_t i = 0; i < session->output_count; ++i) {
        char* name = NULL;
        CHECK(session->api->SessionGetOutputName(session->session, i, session->allocator, &name));
        session->output_names[i] = strdup(name);
        session->api->AllocatorFree(session->allocator, name);
    }
    return session;
}
#undef CHECK

void booki_close(booki_session* s) {
    if (!s) return;
    session_free_names(s->input_names,  s->input_count);
    session_free_names(s->output_names, s->output_count);
    if (s->memory_info) s->api->ReleaseMemoryInfo(s->memory_info);
    if (s->session)     s->api->ReleaseSession(s->session);
    if (s->env)         s->api->ReleaseEnv(s->env);
    free(s);
}

/* ------------------------------------------------------------------------- */
/* Inference                                                                 */
/* ------------------------------------------------------------------------- */

static booki_output empty_output(void) { booki_output o = {0}; return o; }

booki_output booki_run(booki_session* s, const booki_input* in) {
    if (!s || !in) { set_error("null arg"); return empty_output(); }

    const OrtApi* api = s->api;
    OrtValue* tokens_t = NULL;
    OrtValue* style_t  = NULL;
    OrtValue* speed_t  = NULL;
    OrtValue* outputs[1] = { NULL };

    int64_t tokens_shape[2] = { 1, (int64_t)in->token_count };
    OrtStatus* st = api->CreateTensorWithDataAsOrtValue(
        s->memory_info, (void*)in->tokens, in->token_count * sizeof(int64_t),
        tokens_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &tokens_t);
    if (st) { set_error("%s", api->GetErrorMessage(st)); api->ReleaseStatus(st); return empty_output(); }

    int64_t style_shape[2] = { 1, (int64_t)in->style_dim };
    st = api->CreateTensorWithDataAsOrtValue(
        s->memory_info, (void*)in->style, in->style_dim * sizeof(float),
        style_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &style_t);
    if (st) { set_error("%s", api->GetErrorMessage(st)); api->ReleaseStatus(st); goto fail; }

    int64_t speed_shape[1] = { 1 };
    st = api->CreateTensorWithDataAsOrtValue(
        s->memory_info, (void*)&in->speed, sizeof(float),
        speed_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &speed_t);
    if (st) { set_error("%s", api->GetErrorMessage(st)); api->ReleaseStatus(st); goto fail; }

    const OrtValue* input_vals[3] = { tokens_t, style_t, speed_t };
    st = api->Run(s->session, NULL,
        (const char* const*)s->input_names, input_vals, 3,
        (const char* const*)s->output_names, 1, outputs);
    if (st) { set_error("%s", api->GetErrorMessage(st)); api->ReleaseStatus(st); goto fail; }

    /* Extract output samples. Kokoro returns float32 [N]. */
    OrtTensorTypeAndShapeInfo* info = NULL;
    st = api->GetTensorTypeAndShape(outputs[0], &info);
    if (st) { set_error("%s", api->GetErrorMessage(st)); api->ReleaseStatus(st); goto fail; }
    size_t out_count = 0;
    st = api->GetTensorShapeElementCount(info, &out_count);
    api->ReleaseTensorTypeAndShapeInfo(info);
    if (st) { set_error("%s", api->GetErrorMessage(st)); api->ReleaseStatus(st); goto fail; }

    float* src = NULL;
    st = api->GetTensorMutableData(outputs[0], (void**)&src);
    if (st) { set_error("%s", api->GetErrorMessage(st)); api->ReleaseStatus(st); goto fail; }

    booki_output out = {
        .samples       = malloc(out_count * sizeof(float)),
        .sample_count  = out_count,
        .sample_rate_hz = 24000,  /* Kokoro convention */
    };
    if (out.samples) memcpy(out.samples, src, out_count * sizeof(float));

    api->ReleaseValue(outputs[0]);
    api->ReleaseValue(tokens_t);
    api->ReleaseValue(style_t);
    api->ReleaseValue(speed_t);
    return out;

fail:
    if (tokens_t) api->ReleaseValue(tokens_t);
    if (style_t)  api->ReleaseValue(style_t);
    if (speed_t)  api->ReleaseValue(speed_t);
    if (outputs[0]) api->ReleaseValue(outputs[0]);
    return empty_output();
}

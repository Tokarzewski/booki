/* Booki native runtime — public surface.
 *
 * This is the home of the hand-rolled engine that #13 plans to grow into a
 * full Kokoro / Orpheus / SNAC executor. Today it ships the foundation:
 * tensor + arena types, plus one real op (fp16 matmul) with scalar and
 * NEON backends. Each new op merges as its own PR, gated by the bench
 * regression check in runtime-ci.yml.
 *
 * The booki_baseline.h surface from the ORT-backed engine lives separately —
 * we run both side-by-side in CI until coverage is complete enough that the
 * native runtime can replace it.
 */
#ifndef BOOKI_NATIVE_H
#define BOOKI_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/* Tensors                                                                   */
/* ------------------------------------------------------------------------- */

typedef enum {
    BOOKI_DTYPE_F32 = 0,
    BOOKI_DTYPE_F16 = 1,   /* __fp16 / IEEE half */
    BOOKI_DTYPE_I64 = 2,
    BOOKI_DTYPE_I8  = 3,
} booki_dtype;

/* Max tensor rank covers everything Kokoro / Orpheus need (BTCH). */
#define BOOKI_MAX_RANK 4

typedef struct {
    booki_dtype dtype;
    int         rank;
    int64_t     shape[BOOKI_MAX_RANK];
    int64_t     strides[BOOKI_MAX_RANK];  /* element-strides, not byte-strides */
    void*       data;
    size_t      nbytes;
} booki_tensor;

size_t booki_dtype_size(booki_dtype dt);
int64_t booki_tensor_elements(const booki_tensor* t);

/* ------------------------------------------------------------------------- */
/* Arena                                                                     */
/*                                                                           */
/* Bump allocator with a single backing buffer. All scratch tensors for a    */
/* single inference live in one arena; reset() rewinds without freeing so    */
/* successive runs reuse the same allocation. No per-token malloc.           */
/* ------------------------------------------------------------------------- */

typedef struct booki_arena {
    uint8_t* base;
    size_t   capacity;
    size_t   used;
    int      owns;        /* 1 if base was malloc'd by booki_arena_create */
} booki_arena;

booki_arena* booki_arena_create(size_t capacity);
void         booki_arena_destroy(booki_arena* a);
void         booki_arena_reset(booki_arena* a);
size_t       booki_arena_high_water(const booki_arena* a);

/* Allocate aligned space inside the arena. Returns NULL when full. */
void* booki_arena_alloc(booki_arena* a, size_t nbytes, size_t align);

/* Allocate a contiguous tensor of the given shape inside the arena. */
booki_tensor booki_tensor_arena(booki_arena* a, booki_dtype dt, int rank, const int64_t* shape);

/* ------------------------------------------------------------------------- */
/* Backend selection                                                         */
/* ------------------------------------------------------------------------- */

typedef enum {
    BOOKI_BACKEND_AUTO = 0,
    BOOKI_BACKEND_SCALAR,
    BOOKI_BACKEND_NEON,
} booki_backend;

/* Selects which kernels run. AUTO uses NEON when compiled with __ARM_NEON. */
void          booki_set_backend(booki_backend b);
booki_backend booki_backend_active(void);
const char*   booki_backend_describe(booki_backend b);

/* ------------------------------------------------------------------------- */
/* Operators                                                                 */
/* ------------------------------------------------------------------------- */

/* fp16 matmul: C[M,N] = A[M,K] * B[K,N] (row-major, contiguous).
 * Returns 0 on success, non-zero on shape mismatch / dtype error. */
int booki_matmul_f16(const booki_tensor* a, const booki_tensor* b, booki_tensor* c);

/* Root Mean Square LayerNorm (LLaMA/Kokoro convention):
 *   y[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
 * Operates on the last axis of [..., D]. */
int booki_rmsnorm_f16(const booki_tensor* x, const booki_tensor* weight,
                      float eps, booki_tensor* out);

/* SiLU (a.k.a. swish): y = x * sigmoid(x). Element-wise. */
int booki_silu_f16(const booki_tensor* x, booki_tensor* out);

/* GELU (tanh approximation, used by Kokoro's FFN). */
int booki_gelu_f16(const booki_tensor* x, booki_tensor* out);

/* Element-wise binary ops. Same shape, in-place (out = a) allowed. */
int booki_add_f16(const booki_tensor* a, const booki_tensor* b, booki_tensor* out);
int booki_mul_f16(const booki_tensor* a, const booki_tensor* b, booki_tensor* out);

/* Softmax along the last axis with fp32 accumulation + max-shift for
 * numerical stability. Operates on [..., D]; rows are independent. */
int booki_softmax_f16(const booki_tensor* x, booki_tensor* out);

/* Embedding lookup: rows = ids.shape, output rows = weight rows.
 *   out[i, :] = weight[ids[i], :]
 * ids is int64; weight + out are fp16. */
int booki_embedding_f16(const booki_tensor* ids, const booki_tensor* weight,
                        booki_tensor* out);

/* Scaled-dot-product attention. Single head, no mask, no KV cache yet
 * (those land alongside the graph executor):
 *
 *   attn = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * Q [M, D], K [N, D], V [N, D], out [M, D]. Uses scratch from `arena`. */
int booki_attention_f16(const booki_tensor* q, const booki_tensor* k,
                        const booki_tensor* v, booki_arena* arena,
                        booki_tensor* out);

/* Multi-head SDPA. Q/K/V are packed [M, H*D_h] / [N, H*D_h] / [N, H*D_h],
 * out is [M, H*D_h]. Internally reshapes to H independent heads and runs
 * booki_attention_f16 on each. Mask + KV cache come with the executor. */
int booki_multihead_attention_f16(const booki_tensor* q, const booki_tensor* k,
                                  const booki_tensor* v, int num_heads,
                                  booki_arena* arena, booki_tensor* out);

/* 1-D convolution (PyTorch / ONNX layout). See booki_conv1d.c for the full
 * contract. stride, padding, dilation, groups follow PyTorch semantics. */
int booki_conv1d_f16(const booki_tensor* x, const booki_tensor* weight,
                     const booki_tensor* bias,
                     int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                     booki_tensor* out);

/* ------------------------------------------------------------------------- */
/* fp16 conversion helpers — provided for tests and tools that need to
 * cross the boundary without hard-coding ARM intrinsics. */
/* ------------------------------------------------------------------------- */

float    booki_f16_to_f32(uint16_t h);
uint16_t booki_f32_to_f16(float f);

/* ------------------------------------------------------------------------- */
/* Model loader (GGUF v3)                                                    */
/*                                                                           */
/* GGUF is a flat binary container designed for fast mmap-style loading      */
/* of LLM weights. The loader gives us a name -> [booki_tensor] view into    */
/* the file's data without copying. Used as the on-disk format for Booki    */
/* models (converted offline from .onnx / .pt). Spec:                       */
/* https://github.com/ggml-org/ggml/blob/master/docs/gguf.md                 */
/* ------------------------------------------------------------------------- */

typedef struct booki_model booki_model;

booki_model* booki_model_open(const char* path, char* err_out, size_t err_cap);
void         booki_model_close(booki_model* m);

/* Returns the count of tensors in the model. */
int booki_model_tensor_count(const booki_model* m);

/* Returns 0 + populates *out on success, non-zero on missing name or dtype
 * mismatch. The returned tensor's data pointer is owned by [m] and remains
 * valid until booki_model_close. */
int booki_model_tensor(const booki_model* m, const char* name, booki_tensor* out);

/* Returns the i-th tensor's name (interned in [m]), or NULL if out of range. */
const char* booki_model_tensor_name(const booki_model* m, int i);

/* Returns a metadata string (e.g. "general.architecture") or NULL. */
const char* booki_model_meta_string(const booki_model* m, const char* key);

/* ------------------------------------------------------------------------- */
/* Version                                                                   */
/* ------------------------------------------------------------------------- */

const char* booki_native_version(void);

#ifdef __cplusplus
}
#endif
#endif

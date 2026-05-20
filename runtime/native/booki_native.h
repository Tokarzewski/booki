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
 *
 * A, B, C must all be BOOKI_DTYPE_F16, rank 2. The result is accumulated
 * in fp32 internally and stored as fp16.
 *
 * Returns 0 on success, non-zero on shape mismatch / dtype error. */
int booki_matmul_f16(const booki_tensor* a, const booki_tensor* b, booki_tensor* c);

/* ------------------------------------------------------------------------- */
/* fp16 conversion helpers — provided for tests and tools that need to
 * cross the boundary without hard-coding ARM intrinsics. */
/* ------------------------------------------------------------------------- */

float    booki_f16_to_f32(uint16_t h);
uint16_t booki_f32_to_f16(float f);

/* ------------------------------------------------------------------------- */
/* Version                                                                   */
/* ------------------------------------------------------------------------- */

const char* booki_native_version(void);

#ifdef __cplusplus
}
#endif
#endif

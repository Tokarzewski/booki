/* Tensor + arena implementation. Pure C, no SIMD, no third-party deps. */

#include "booki_native.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BOOKI_NATIVE_VERSION "0.1.0-native"

size_t booki_dtype_size(booki_dtype dt) {
    switch (dt) {
        case BOOKI_DTYPE_F32: return 4;
        case BOOKI_DTYPE_F16: return 2;
        case BOOKI_DTYPE_I64: return 8;
        case BOOKI_DTYPE_I8:  return 1;
    }
    return 0;
}

int64_t booki_tensor_elements(const booki_tensor* t) {
    if (!t || t->rank <= 0) return 0;
    int64_t n = 1;
    for (int i = 0; i < t->rank; ++i) n *= t->shape[i];
    return n;
}

/* ------------------------------------------------------------------------- */
/* Arena                                                                     */
/* ------------------------------------------------------------------------- */

static size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

booki_arena* booki_arena_create(size_t capacity) {
    booki_arena* a = (booki_arena*)calloc(1, sizeof(*a));
    if (!a) return NULL;
    /* 64-byte aligned base so SIMD loads never straddle the boundary. */
    if (posix_memalign((void**)&a->base, 64, capacity) != 0) {
        free(a);
        return NULL;
    }
    a->capacity = capacity;
    a->used     = 0;
    a->owns     = 1;
    return a;
}

void booki_arena_destroy(booki_arena* a) {
    if (!a) return;
    if (a->owns && a->base) free(a->base);
    free(a);
}

void booki_arena_reset(booki_arena* a) {
    if (a) a->used = 0;
}

size_t booki_arena_high_water(const booki_arena* a) {
    return a ? a->used : 0;
}

void* booki_arena_alloc(booki_arena* a, size_t nbytes, size_t align) {
    if (!a || nbytes == 0) return NULL;
    size_t start = align_up(a->used, align);
    if (start + nbytes > a->capacity) {
        fprintf(stderr, "booki: arena oom (need %zu at offset %zu, cap %zu)\n",
                nbytes, start, a->capacity);
        return NULL;
    }
    a->used = start + nbytes;
    return a->base + start;
}

booki_tensor booki_tensor_arena(booki_arena* a, booki_dtype dt, int rank, const int64_t* shape) {
    booki_tensor t = {0};
    if (!a || rank <= 0 || rank > BOOKI_MAX_RANK || !shape) return t;

    int64_t elems = 1;
    for (int i = 0; i < rank; ++i) { t.shape[i] = shape[i]; elems *= shape[i]; }
    /* Contiguous C-order strides. */
    int64_t s = 1;
    for (int i = rank - 1; i >= 0; --i) { t.strides[i] = s; s *= shape[i]; }

    t.dtype  = dt;
    t.rank   = rank;
    t.nbytes = (size_t)elems * booki_dtype_size(dt);
    t.data   = booki_arena_alloc(a, t.nbytes, 64);
    if (!t.data) memset(&t, 0, sizeof(t));
    return t;
}

/* ------------------------------------------------------------------------- */
/* Version                                                                   */
/* ------------------------------------------------------------------------- */

const char* booki_native_version(void) { return BOOKI_NATIVE_VERSION; }

/* ------------------------------------------------------------------------- */
/* Backend selection                                                         */
/* ------------------------------------------------------------------------- */

static booki_backend g_backend = BOOKI_BACKEND_AUTO;

void booki_set_backend(booki_backend b) { g_backend = b; }

booki_backend booki_backend_active(void) {
    if (g_backend != BOOKI_BACKEND_AUTO) return g_backend;
#if defined(__ARM_NEON) || defined(__aarch64__)
    return BOOKI_BACKEND_NEON;
#else
    return BOOKI_BACKEND_SCALAR;
#endif
}

const char* booki_backend_describe(booki_backend b) {
    if (b == BOOKI_BACKEND_AUTO) b = booki_backend_active();
    switch (b) {
        case BOOKI_BACKEND_SCALAR: return "scalar";
        case BOOKI_BACKEND_NEON:   return "neon";
        default:                   return "auto";
    }
}

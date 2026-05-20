/* Smoke tests for tensor + arena. */

#include "booki_native.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return 1; } \
} while (0)

int main(void) {
    booki_arena* a = booki_arena_create(64 * 1024);
    CHECK(a != NULL, "create");
    CHECK(booki_arena_high_water(a) == 0, "starts empty");

    int64_t shape[2] = { 8, 16 };
    booki_tensor t = booki_tensor_arena(a, BOOKI_DTYPE_F32, 2, shape);
    CHECK(t.data != NULL,                 "tensor data");
    CHECK(t.nbytes == 8 * 16 * 4,         "tensor nbytes");
    CHECK(booki_tensor_elements(&t) == 128, "elements");
    CHECK(t.strides[0] == 16 && t.strides[1] == 1, "C-order strides");

    /* All writes inside our buffer? */
    memset(t.data, 0x42, t.nbytes);

    /* Allocating past capacity should return NULL without trashing the arena */
    booki_tensor too_big = booki_tensor_arena(a, BOOKI_DTYPE_F32, 2,
        (int64_t[]){ 1000, 1000 });
    CHECK(too_big.data == NULL, "oversize alloc rejected");

    /* Reset reclaims the whole arena. */
    booki_arena_reset(a);
    CHECK(booki_arena_high_water(a) == 0, "reset");

    /* Backend describes itself sanely. */
    CHECK(booki_backend_describe(booki_backend_active()) != NULL, "backend name");
    CHECK(booki_native_version() != NULL, "version");

    booki_arena_destroy(a);
    printf("arena ok\n");
    return 0;
}

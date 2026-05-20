/* Tests for booki_sequence + SplitToSequence / ConcatFromSequence. */

#include "booki_native.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } \
} while (0)

static booki_tensor mk(booki_arena* a, int rank, const int64_t* shape) {
    return booki_tensor_arena(a, BOOKI_DTYPE_F16, rank, shape);
}

static void fill(booki_tensor* t, const float* src) {
    uint16_t* p = (uint16_t*)t->data;
    int64_t n = booki_tensor_elements(t);
    for (int64_t i = 0; i < n; ++i) p[i] = booki_f32_to_f16(src[i]);
}
static float read(const booki_tensor* t, int64_t i) {
    return booki_f16_to_f32(((const uint16_t*)t->data)[i]);
}

static int test_basic_push_pop(booki_arena* a) {
    booki_sequence* s = booki_sequence_new(a);
    CHECK(s, "seq_new");
    CHECK(booki_sequence_len(s) == 0, "empty len");

    int64_t shp[1] = { 3 };
    booki_tensor t1 = mk(a, 1, shp); fill(&t1, (float[]){1, 2, 3});
    booki_tensor t2 = mk(a, 1, shp); fill(&t2, (float[]){10, 20, 30});
    CHECK(booki_sequence_push(s, &t1) == 0, "push 1");
    CHECK(booki_sequence_push(s, &t2) == 0, "push 2");
    CHECK(booki_sequence_len(s) == 2, "len=2");

    booki_tensor read1, read2;
    CHECK(booki_sequence_at(s, 0, &read1) == 0, "at 0");
    CHECK(booki_sequence_at(s, 1, &read2) == 0, "at 1");
    CHECK(fabsf(read(&read1, 0) - 1.0f) < 1e-3f, "t1[0]");
    CHECK(fabsf(read(&read2, 2) - 30.0f) < 1e-3f, "t2[2]");
    return 0;
}

static int test_split_to_sequence(booki_arena* a) {
    /* x: shape [6], split axis=0, chunk=2 → 3 tensors of shape [2] each. */
    int64_t shp[1] = { 6 };
    booki_tensor x = mk(a, 1, shp); fill(&x, (float[]){1, 2, 3, 4, 5, 6});
    booki_sequence* s = booki_sequence_new(a);
    CHECK(booki_split_to_sequence_f16(&x, 0, /*chunk=*/2, s, a) == 0, "split");
    CHECK(booki_sequence_len(s) == 3, "split len");

    float want[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    for (int c = 0; c < 3; ++c) {
        booki_tensor chunk;
        booki_sequence_at(s, c, &chunk);
        for (int i = 0; i < 2; ++i)
            CHECK(fabsf(read(&chunk, i) - want[c][i]) < 1e-3f,
                  "chunk[%d][%d]=%.2f want %.2f", c, i, read(&chunk, i), want[c][i]);
    }
    return 0;
}

static int test_split_2d_axis_1(booki_arena* a) {
    /* x: shape [2, 6], split axis=1, chunk=3 → 2 tensors of shape [2, 3]. */
    int64_t shp[2] = { 2, 6 };
    booki_tensor x = mk(a, 2, shp);
    fill(&x, (float[]){1, 2, 3, 4, 5, 6,
                        10, 20, 30, 40, 50, 60});
    booki_sequence* s = booki_sequence_new(a);
    CHECK(booki_split_to_sequence_f16(&x, 1, /*chunk=*/3, s, a) == 0, "split");
    CHECK(booki_sequence_len(s) == 2, "len");

    /* First chunk should be [[1,2,3],[10,20,30]]; second [[4,5,6],[40,50,60]]. */
    booki_tensor c0, c1;
    booki_sequence_at(s, 0, &c0);
    booki_sequence_at(s, 1, &c1);
    CHECK(fabsf(read(&c0, 0) - 1.0f) < 1e-3f, "c0[0]");
    CHECK(fabsf(read(&c0, 5) - 30.0f) < 1e-3f, "c0[5]");
    CHECK(fabsf(read(&c1, 0) - 4.0f) < 1e-3f, "c1[0]");
    CHECK(fabsf(read(&c1, 5) - 60.0f) < 1e-3f, "c1[5]");
    return 0;
}

static int test_concat_from_sequence(booki_arena* a) {
    /* Split then concat should round-trip. */
    int64_t shp[2] = { 2, 6 };
    booki_tensor x = mk(a, 2, shp);
    fill(&x, (float[]){1, 2, 3, 4, 5, 6,
                        10, 20, 30, 40, 50, 60});
    booki_sequence* s = booki_sequence_new(a);
    CHECK(booki_split_to_sequence_f16(&x, 1, /*chunk=*/2, s, a) == 0, "split");

    booki_tensor y = {0};
    CHECK(booki_concat_from_sequence_f16(s, /*axis=*/1, a, &y) == 0, "concat");
    CHECK(y.rank == 2 && y.shape[0] == 2 && y.shape[1] == 6, "y shape");

    /* Round-trip exactness. */
    for (int64_t i = 0; i < 12; ++i)
        CHECK(fabsf(read(&y, i) - read(&x, i)) < 1e-3f,
              "round-trip diverges at %lld", (long long)i);
    return 0;
}

int main(void) {
    booki_arena* a = booki_arena_create(1 << 20);
    int rc = 0;
    rc |= test_basic_push_pop(a);   booki_arena_reset(a);
    rc |= test_split_to_sequence(a); booki_arena_reset(a);
    rc |= test_split_2d_axis_1(a);   booki_arena_reset(a);
    rc |= test_concat_from_sequence(a);
    booki_arena_destroy(a);
    if (rc == 0) printf("sequence ok\n");
    return rc;
}

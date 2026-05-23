/* Implementation of the parallel_for helper. See booki_threading.h. */

#include "booki_threading.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Hard ceiling — Snapdragon 8 Elite Gen 5 has 2 Prime cores. Going higher
 * would buy nothing for matmul shaped like Kokoro's (the L2 is shared
 * between the two Prime cores; the Performance cluster has different L2 and
 * is thermally throttled first). Keep the ceiling here so callers can't
 * accidentally oversubscribe. */
#define BOOKI_MAX_THREADS 8

static int g_threads = 0;  /* 0 = not yet probed */

static int clamp_threads(int n) {
    if (n < 1) n = 1;
    if (n > BOOKI_MAX_THREADS) n = BOOKI_MAX_THREADS;
    return n;
}

static int probe_default(void) {
    const char* env = getenv("BOOKI_THREADS");
    if (env && *env) {
        int v = atoi(env);
        if (v > 0) return clamp_threads(v);
    }
#if defined(_SC_NPROCESSORS_CONF)
    long n = sysconf(_SC_NPROCESSORS_CONF);
    if (n < 1) n = 1;
    /* Default is 1 — phone validation showed that fork-join overhead
     * per matmul makes the composed block bench 2× slower with 2
     * threads, even though isolated 1024³ matmul gains ~8%. Opt into
     * threading explicitly via BOOKI_THREADS=N when running batch
     * workloads dominated by one big matmul (e.g. LM-head decode). */
    (void)n;
    return 1;
#else
    return 1;
#endif
}

int booki_set_threads(int n) {
    if (n == 0) {
        g_threads = probe_default();
    } else {
        g_threads = clamp_threads(n);
    }
    return g_threads;
}

int booki_get_threads(void) {
    if (g_threads == 0) g_threads = probe_default();
    return g_threads;
}

typedef struct {
    booki_parallel_fn fn;
    void* ctx;
    int64_t begin;
    int64_t end;
} worker_arg;

static void* worker_entry(void* p) {
    worker_arg* w = (worker_arg*)p;
    w->fn(w->begin, w->end, w->ctx);
    return NULL;
}

int booki_parallel_for(int64_t n_outer, int64_t grain,
                       booki_parallel_fn fn, void* ctx) {
    if (n_outer <= 0) return 0;
    if (grain < 1) grain = 1;

    int nthreads = booki_get_threads();

    /* Don't spin up workers when there's not enough work to amortize the
     * pthread_create cost — falls back to in-thread serial. */
    int64_t max_useful = n_outer / grain;
    if (max_useful < 1) max_useful = 1;
    if ((int64_t)nthreads > max_useful) nthreads = (int)max_useful;
    if (nthreads <= 1) {
        fn(0, n_outer, ctx);
        return 0;
    }

    /* Static block partition. For matmul this is fine because each output
     * row costs the same; if a kernel ever has imbalanced rows it can use a
     * different grain to opt into smaller chunks. */
    pthread_t threads[BOOKI_MAX_THREADS];
    worker_arg args[BOOKI_MAX_THREADS];
    int spawned = 0;

    int64_t chunk = (n_outer + nthreads - 1) / nthreads;
    /* Round chunk up to a multiple of grain so tile boundaries line up
     * with what the caller asked for (e.g. SVL_w on SME). */
    chunk = ((chunk + grain - 1) / grain) * grain;
    if (chunk < grain) chunk = grain;

    for (int t = 0; t < nthreads; ++t) {
        int64_t b = (int64_t)t * chunk;
        if (b >= n_outer) break;
        int64_t e = b + chunk;
        if (e > n_outer) e = n_outer;
        if (t == nthreads - 1) e = n_outer;  /* last picks up remainder */
        args[t].fn = fn;
        args[t].ctx = ctx;
        args[t].begin = b;
        args[t].end = e;
    }

    /* Spawn N-1, run the last slice on the calling thread. Saves one
     * create/join pair on the hot path. */
    for (int t = 0; t < nthreads - 1; ++t) {
        if (args[t].begin >= n_outer) break;
        if (pthread_create(&threads[t], NULL, worker_entry, &args[t]) != 0) {
            /* On failure, finish remaining work serially. */
            for (int s = t; s < nthreads; ++s) {
                if (args[s].begin >= n_outer) break;
                fn(args[s].begin, args[s].end, ctx);
            }
            for (int s = 0; s < t; ++s) pthread_join(threads[s], NULL);
            return 0;
        }
        spawned++;
    }

    /* In-thread tail. */
    if (args[nthreads - 1].begin < n_outer) {
        fn(args[nthreads - 1].begin, args[nthreads - 1].end, ctx);
    }

    for (int t = 0; t < spawned; ++t) pthread_join(threads[t], NULL);
    return 0;
}

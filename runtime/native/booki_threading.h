/* Tiny pthread-based fork-join helper for the native runtime.
 *
 * Designed for the matmul M-dimension split on Snapdragon 8 Elite Gen 5
 * (2 Prime "L" cores share an L2; threading past 2 buys nothing and costs
 * thermal headroom). The helper is intentionally minimal:
 *
 *   - No persistent pool. Each parallel_for spawns its workers, joins, exits.
 *     A matmul takes O(ms); pthread_create on Linux/Android is O(us), well
 *     under 1% of the kernel time at the sizes we care about. A pool can
 *     come later if profiling shows it matters.
 *   - No global state. Workers receive a per-call context pointer.
 *   - Thread count is read from BOOKI_THREADS env var (set once per process),
 *     defaults to min(2, sysconf(_SC_NPROCESSORS_CONF)). Clamped to 2 because
 *     more threads on a heterogeneous big.LITTLE chip causes the slowest
 *     core to gate forward progress.
 *
 * The helper degrades to a plain in-thread loop when num_threads == 1 or
 * when the range is too small to amortize spawn overhead.
 */
#ifndef BOOKI_THREADING_H
#define BOOKI_THREADING_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Worker function. Operates on the half-open interval [begin, end) of the
 * outer index. Each worker is single-threaded internally; the helper is
 * responsible for the slicing. */
typedef void (*booki_parallel_fn)(int64_t begin, int64_t end, void* ctx);

/* Slice [0, n_outer) across booki_get_threads() workers and run [fn] on
 * each slice in a separate pthread. Joins before returning.
 *
 * [grain] is a hint: when (n_outer / grain) < num_threads the helper
 * shrinks the worker count so each thread sees at least [grain] work units.
 * Pass 1 to always use the full thread count.
 *
 * Returns 0 on success. On pthread_create failure falls back to serial
 * execution (still returns 0). */
int booki_parallel_for(int64_t n_outer, int64_t grain,
                       booki_parallel_fn fn, void* ctx);

/* Override the thread count for this process. 0 means "re-read BOOKI_THREADS
 * and re-clamp". Returns the value now in effect. */
int booki_set_threads(int n);
int booki_get_threads(void);

#ifdef __cplusplus
}
#endif
#endif

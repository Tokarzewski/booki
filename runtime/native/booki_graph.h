/* Graph executor — turns the ops into a composable forward pass driven
 * by a node list rather than hand-written C. Together with the GGUF
 * loader this is what lets us load a real Kokoro model and run it.
 *
 * Today the graph is built imperatively from C (see test_graph.c). A
 * model-loader pass that materialises a graph from a GGUF file's op
 * metadata is the next step; the executor interface stays the same.
 *
 * Usage:
 *   booki_arena*  workspace = booki_arena_create(MB);
 *   booki_graph*  g         = booki_graph_create(workspace, NULL);
 *   int x       = booki_graph_input(g, "x", BOOKI_DTYPE_F16, 2, (int64_t[]){M, D});
 *   int weight  = booki_graph_const(g, &W_tensor);
 *   int normed  = booki_graph_rmsnorm(g, x, weight, 1e-6f);
 *   booki_graph_set_output(g, normed);
 *
 *   booki_graph_bind(g, x, &input_tensor);
 *   booki_tensor out;
 *   booki_graph_run(g, &out);
 */
#ifndef BOOKI_GRAPH_H
#define BOOKI_GRAPH_H

#include "booki_native.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct booki_graph booki_graph;

booki_graph* booki_graph_create(booki_arena* workspace, booki_model* weights);
void         booki_graph_destroy(booki_graph* g);

/* Construct nodes. Each call returns a node id (>= 0) used to reference
 * the tensor produced by that node in subsequent calls. */
int booki_graph_input(booki_graph* g, const char* name,
                      booki_dtype dt, int rank, const int64_t* shape);
int booki_graph_const(booki_graph* g, const booki_tensor* t);
int booki_graph_weight(booki_graph* g, const char* model_tensor_name);

int booki_graph_matmul   (booki_graph* g, int a, int b);
int booki_graph_rmsnorm  (booki_graph* g, int x, int weight, float eps);
int booki_graph_silu     (booki_graph* g, int x);
int booki_graph_gelu     (booki_graph* g, int x);
int booki_graph_add      (booki_graph* g, int a, int b);
int booki_graph_mul      (booki_graph* g, int a, int b);
int booki_graph_softmax  (booki_graph* g, int x);
int booki_graph_embedding(booki_graph* g, int ids, int weight);
int booki_graph_attention(booki_graph* g, int q, int k, int v, int num_heads);

void booki_graph_set_output(booki_graph* g, int node);

/* Bind an external tensor to an input node. The tensor's data pointer
 * must remain valid for the duration of booki_graph_run. */
int booki_graph_bind(booki_graph* g, int node, const booki_tensor* t);

/* Execute. On success [out] holds a *view* of the workspace arena —
 * caller must not free its data; it's invalidated when the workspace
 * is reset or destroyed. */
int booki_graph_run(booki_graph* g, booki_tensor* out);

/* Diagnostics. */
int         booki_graph_node_count(const booki_graph* g);
const char* booki_graph_last_error(const booki_graph* g);

#ifdef __cplusplus
}
#endif
#endif

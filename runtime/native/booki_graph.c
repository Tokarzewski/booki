/* Graph executor. Walks the node list once per call, allocating each
 * intermediate tensor from the workspace arena and dispatching to the
 * existing op kernels. No automatic differentiation, no memory reuse
 * across nodes (everything stays live until graph_run returns — fine
 * for one forward pass, which is what we care about).
 *
 * Shape inference happens at graph-construction time so we can size the
 * arena up front. Op-specific shape rules live in this file.
 */

#include "booki_graph.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    OP_INPUT,
    OP_CONST,
    OP_WEIGHT,
    OP_MATMUL,
    OP_RMSNORM,
    OP_SILU,
    OP_GELU,
    OP_LEAKY_RELU,
    OP_SIN,
    OP_COS,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_SOFTMAX,
    OP_EMBEDDING,
    OP_ATTENTION,
    OP_INSTANCE_NORM,
    OP_CONV1D,
    OP_CONV_TRANSPOSE1D,
    OP_RESIZE1D,
    OP_LSTM,
} op_kind;

typedef struct {
    op_kind  kind;
    int      in[4];
    int      n_in;
    /* Op-specific scalar params. */
    union {
        struct { float eps; }      rmsnorm;
        struct { int heads; }      attention;
        struct { float alpha; }    leaky_relu;
        struct { float eps; }      instance_norm;
        struct { int64_t stride, padding, dilation, groups; }            conv1d;
        struct { int64_t stride, padding, output_padding, dilation, groups; } conv_transpose1d;
        struct { int64_t out_size; int mode; }                            resize1d;
        struct { int h0, c0; }                                            lstm;
    } p;
    /* External binding for input/const/weight nodes. */
    const booki_tensor* external;
    const char*         weight_name;
    /* Tensor produced by this node (shape known at build time, data filled at run). */
    booki_dtype dtype;
    int         rank;
    int64_t     shape[BOOKI_MAX_RANK];
} node;

struct booki_graph {
    booki_arena*  workspace;
    booki_model*  model;
    node*         nodes;
    int           n_nodes;
    int           cap;
    /* Resolved tensors for each node. Filled on graph_run, reset on next run. */
    booki_tensor* tensors;
    int           output;
    char          err[256];
};

/* ------------------------------------------------------------------------- */
/* Plumbing                                                                  */
/* ------------------------------------------------------------------------- */

static int set_err(booki_graph* g, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(g->err, sizeof(g->err), fmt, ap);
    va_end(ap);
    return -1;
}

booki_graph* booki_graph_create(booki_arena* workspace, booki_model* model) {
    if (!workspace) return NULL;
    booki_graph* g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->workspace = workspace;
    g->model = model;
    g->cap = 32;
    g->nodes = calloc(g->cap, sizeof(node));
    g->output = -1;
    if (!g->nodes) { free(g); return NULL; }
    return g;
}

void booki_graph_destroy(booki_graph* g) {
    if (!g) return;
    free(g->nodes);
    free(g->tensors);
    free(g);
}

static int append(booki_graph* g) {
    if (g->n_nodes == g->cap) {
        int nc = g->cap * 2;
        node* nn = realloc(g->nodes, nc * sizeof(node));
        if (!nn) return -1;
        g->nodes = nn; g->cap = nc;
    }
    int id = g->n_nodes++;
    memset(&g->nodes[id], 0, sizeof(node));
    return id;
}

/* ------------------------------------------------------------------------- */
/* Shape inference per op                                                    */
/* ------------------------------------------------------------------------- */

static int copy_shape(node* dst, const node* src) {
    dst->dtype = src->dtype;
    dst->rank  = src->rank;
    memcpy(dst->shape, src->shape, sizeof(src->shape));
    return 0;
}

static int set_shape(node* n, booki_dtype dt, int rank, const int64_t* shape) {
    if (rank > BOOKI_MAX_RANK) return -1;
    n->dtype = dt;
    n->rank  = rank;
    for (int i = 0; i < rank; ++i) n->shape[i] = shape[i];
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Node constructors                                                         */
/* ------------------------------------------------------------------------- */

int booki_graph_input(booki_graph* g, const char* name,
                      booki_dtype dt, int rank, const int64_t* shape) {
    (void)name;  /* names are for diagnostics; bind uses node ids */
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = OP_INPUT;
    set_shape(n, dt, rank, shape);
    return id;
}

int booki_graph_const(booki_graph* g, const booki_tensor* t) {
    if (!t || !t->data) return set_err(g, "bad const");
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = OP_CONST;
    n->external = t;
    set_shape(n, t->dtype, t->rank, t->shape);
    return id;
}

int booki_graph_weight(booki_graph* g, const char* model_tensor_name) {
    if (!g->model) return set_err(g, "no model bound");
    booki_tensor t = {0};
    if (booki_model_tensor(g->model, model_tensor_name, &t) != 0)
        return set_err(g, "weight not found: %s", model_tensor_name);
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = OP_WEIGHT;
    n->weight_name = model_tensor_name;
    /* Stash a static tensor copy inline; data pointer is into model mmap. */
    static booki_tensor scratch_template = {0};
    (void)scratch_template;
    n->external = NULL;   /* resolved at run time from the model */
    set_shape(n, t.dtype, t.rank, t.shape);
    return id;
}

int booki_graph_matmul(booki_graph* g, int a, int b) {
    if (a < 0 || b < 0 || a >= g->n_nodes || b >= g->n_nodes)
        return set_err(g, "matmul: bad input ids");
    const node* na = &g->nodes[a];
    const node* nb = &g->nodes[b];
    if (na->rank != 2 || nb->rank != 2 || na->shape[1] != nb->shape[0])
        return set_err(g, "matmul: shape mismatch");
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = OP_MATMUL;
    n->in[0] = a; n->in[1] = b; n->n_in = 2;
    int64_t s[2] = { na->shape[0], nb->shape[1] };
    set_shape(n, BOOKI_DTYPE_F16, 2, s);
    return id;
}

static int unary(booki_graph* g, op_kind k, int x) {
    if (x < 0 || x >= g->n_nodes) return set_err(g, "unary: bad id");
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = k;
    n->in[0] = x; n->n_in = 1;
    copy_shape(n, &g->nodes[x]);
    return id;
}

int booki_graph_silu(booki_graph* g, int x)   { return unary(g, OP_SILU, x); }
int booki_graph_gelu(booki_graph* g, int x)   { return unary(g, OP_GELU, x); }
int booki_graph_softmax(booki_graph* g, int x){ return unary(g, OP_SOFTMAX, x); }
int booki_graph_sin(booki_graph* g, int x)    { return unary(g, OP_SIN, x); }
int booki_graph_cos(booki_graph* g, int x)    { return unary(g, OP_COS, x); }

int booki_graph_leaky_relu(booki_graph* g, int x, float alpha) {
    int id = unary(g, OP_LEAKY_RELU, x);
    if (id >= 0) g->nodes[id].p.leaky_relu.alpha = alpha;
    return id;
}

int booki_graph_rmsnorm(booki_graph* g, int x, int weight, float eps) {
    if (x < 0 || weight < 0) return set_err(g, "rmsnorm: bad ids");
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = OP_RMSNORM;
    n->in[0] = x; n->in[1] = weight; n->n_in = 2;
    n->p.rmsnorm.eps = eps;
    copy_shape(n, &g->nodes[x]);
    return id;
}

static int binary(booki_graph* g, op_kind k, int a, int b) {
    if (a < 0 || b < 0) return set_err(g, "binary: bad ids");
    const node* na = &g->nodes[a];
    const node* nb = &g->nodes[b];
    if (na->rank != nb->rank) return set_err(g, "binary: rank mismatch");
    for (int i = 0; i < na->rank; ++i)
        if (na->shape[i] != nb->shape[i])
            return set_err(g, "binary: shape mismatch at axis %d", i);
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = k;
    n->in[0] = a; n->in[1] = b; n->n_in = 2;
    copy_shape(n, na);
    return id;
}

int booki_graph_add(booki_graph* g, int a, int b) { return binary(g, OP_ADD, a, b); }
int booki_graph_sub(booki_graph* g, int a, int b) { return binary(g, OP_SUB, a, b); }
int booki_graph_mul(booki_graph* g, int a, int b) { return binary(g, OP_MUL, a, b); }

int booki_graph_instance_norm(booki_graph* g, int x, int scale, int bias, float eps) {
    if (x < 0 || scale < 0) return set_err(g, "instance_norm: bad ids");
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = OP_INSTANCE_NORM;
    n->in[0] = x; n->in[1] = scale;
    n->in[2] = bias; /* may be -1 for no-bias */
    n->n_in = bias >= 0 ? 3 : 2;
    n->p.instance_norm.eps = eps;
    copy_shape(n, &g->nodes[x]);
    return id;
}

int booki_graph_conv1d(booki_graph* g, int x, int w, int b,
                       int64_t stride, int64_t padding,
                       int64_t dilation, int64_t groups) {
    if (x < 0 || w < 0) return set_err(g, "conv1d: bad ids");
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = OP_CONV1D;
    n->in[0] = x; n->in[1] = w; n->in[2] = b; n->n_in = b >= 0 ? 3 : 2;
    n->p.conv1d.stride = stride;
    n->p.conv1d.padding = padding;
    n->p.conv1d.dilation = dilation;
    n->p.conv1d.groups = groups;

    /* Output shape inference. */
    const node* nx = &g->nodes[x];
    const node* nw = &g->nodes[w];
    int64_t B = nx->shape[0];
    int64_t Lin = nx->shape[2];
    int64_t Cout = nw->shape[0];
    int64_t K = nw->shape[2];
    int64_t Lout = (Lin + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int64_t s[3] = { B, Cout, Lout };
    set_shape(n, BOOKI_DTYPE_F16, 3, s);
    return id;
}

int booki_graph_conv_transpose1d(booki_graph* g, int x, int w, int b,
                                 int64_t stride, int64_t padding,
                                 int64_t output_padding, int64_t dilation,
                                 int64_t groups) {
    if (x < 0 || w < 0) return set_err(g, "conv_transpose1d: bad ids");
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = OP_CONV_TRANSPOSE1D;
    n->in[0] = x; n->in[1] = w; n->in[2] = b; n->n_in = b >= 0 ? 3 : 2;
    n->p.conv_transpose1d.stride = stride;
    n->p.conv_transpose1d.padding = padding;
    n->p.conv_transpose1d.output_padding = output_padding;
    n->p.conv_transpose1d.dilation = dilation;
    n->p.conv_transpose1d.groups = groups;

    const node* nx = &g->nodes[x];
    const node* nw = &g->nodes[w];
    int64_t B = nx->shape[0];
    int64_t Lin = nx->shape[2];
    int64_t Cout = nw->shape[1] * groups;
    int64_t K = nw->shape[2];
    int64_t Lout = (Lin - 1) * stride - 2 * padding + dilation * (K - 1)
                 + output_padding + 1;
    int64_t s[3] = { B, Cout, Lout };
    set_shape(n, BOOKI_DTYPE_F16, 3, s);
    return id;
}

int booki_graph_resize1d(booki_graph* g, int x, int64_t out_size, int mode) {
    if (x < 0) return set_err(g, "resize1d: bad id");
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = OP_RESIZE1D;
    n->in[0] = x; n->n_in = 1;
    n->p.resize1d.out_size = out_size;
    n->p.resize1d.mode = mode;
    copy_shape(n, &g->nodes[x]);
    n->shape[n->rank - 1] = out_size;
    return id;
}

int booki_graph_lstm(booki_graph* g, int x, int w, int r, int b, int h0, int c0) {
    if (x < 0 || w < 0 || r < 0) return set_err(g, "lstm: bad ids");
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = OP_LSTM;
    n->in[0] = x; n->in[1] = w; n->in[2] = r; n->in[3] = b;
    n->n_in = b >= 0 ? 4 : 3;
    n->p.lstm.h0 = h0;
    n->p.lstm.c0 = c0;

    /* Output [T, H] where H = W.rank0 / 4. */
    const node* nx = &g->nodes[x];
    const node* nw = &g->nodes[w];
    int64_t T = nx->shape[0];
    int64_t H = nw->shape[0] / 4;
    int64_t s[2] = { T, H };
    set_shape(n, BOOKI_DTYPE_F16, 2, s);
    return id;
}

int booki_graph_embedding(booki_graph* g, int ids, int weight) {
    if (ids < 0 || weight < 0) return set_err(g, "embedding: bad ids");
    const node* ni = &g->nodes[ids];
    const node* nw = &g->nodes[weight];
    if (nw->rank != 2) return set_err(g, "embedding: weight rank != 2");
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = OP_EMBEDDING;
    n->in[0] = ids; n->in[1] = weight; n->n_in = 2;
    /* Output is [T, D] flat. */
    int64_t T = 1;
    for (int i = 0; i < ni->rank; ++i) T *= ni->shape[i];
    int64_t s[2] = { T, nw->shape[1] };
    set_shape(n, BOOKI_DTYPE_F16, 2, s);
    return id;
}

int booki_graph_attention(booki_graph* g, int q, int k, int v, int num_heads) {
    if (q < 0 || k < 0 || v < 0) return set_err(g, "attention: bad ids");
    const node* nq = &g->nodes[q];
    int id = append(g); if (id < 0) return set_err(g, "alloc");
    node* n = &g->nodes[id];
    n->kind = OP_ATTENTION;
    n->in[0] = q; n->in[1] = k; n->in[2] = v; n->n_in = 3;
    n->p.attention.heads = num_heads;
    copy_shape(n, nq);
    return id;
}

void booki_graph_set_output(booki_graph* g, int node_) { g->output = node_; }

int booki_graph_bind(booki_graph* g, int node_, const booki_tensor* t) {
    if (node_ < 0 || node_ >= g->n_nodes) return set_err(g, "bind: bad node");
    if (g->nodes[node_].kind != OP_INPUT) return set_err(g, "bind: not an input");
    g->nodes[node_].external = t;
    return 0;
}

int  booki_graph_node_count(const booki_graph* g) { return g ? g->n_nodes : 0; }
const char* booki_graph_last_error(const booki_graph* g) { return g ? g->err : ""; }

/* ------------------------------------------------------------------------- */
/* Execution                                                                 */
/* ------------------------------------------------------------------------- */

static booki_tensor materialize(booki_graph* g, const node* n) {
    return booki_tensor_arena(g->workspace, n->dtype, n->rank, n->shape);
}

int booki_graph_run(booki_graph* g, booki_tensor* out_view) {
    if (!g) return -1;
    if (g->output < 0) return set_err(g, "no output set");

    /* Reset workspace + per-run tensor table. */
    booki_arena_reset(g->workspace);
    if (g->tensors) free(g->tensors);
    g->tensors = calloc(g->n_nodes, sizeof(booki_tensor));
    if (!g->tensors) return set_err(g, "tensors alloc");

    /* Scratch arena used inside attention (KT + scores). Subset of the
     * main workspace — bounded by the largest attention call. */
    booki_arena* attn_scratch = NULL;

    for (int i = 0; i < g->n_nodes; ++i) {
        node* n = &g->nodes[i];
        booki_tensor* t = &g->tensors[i];

        switch (n->kind) {
        case OP_INPUT:
            if (!n->external) return set_err(g, "input #%d not bound", i);
            *t = *n->external;
            break;
        case OP_CONST:
            *t = *n->external;
            break;
        case OP_WEIGHT:
            if (booki_model_tensor(g->model, n->weight_name, t) != 0)
                return set_err(g, "weight missing: %s", n->weight_name);
            break;
        case OP_MATMUL: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "matmul alloc");
            int rc = booki_matmul_f16(&g->tensors[n->in[0]],
                                      &g->tensors[n->in[1]], t);
            if (rc) return set_err(g, "matmul rc=%d (node %d)", rc, i);
        } break;
        case OP_RMSNORM: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "rmsnorm alloc");
            int rc = booki_rmsnorm_f16(&g->tensors[n->in[0]],
                                       &g->tensors[n->in[1]],
                                       n->p.rmsnorm.eps, t);
            if (rc) return set_err(g, "rmsnorm rc=%d", rc);
        } break;
        case OP_SILU: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "silu alloc");
            int rc = booki_silu_f16(&g->tensors[n->in[0]], t);
            if (rc) return set_err(g, "silu rc=%d", rc);
        } break;
        case OP_GELU: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "gelu alloc");
            int rc = booki_gelu_f16(&g->tensors[n->in[0]], t);
            if (rc) return set_err(g, "gelu rc=%d", rc);
        } break;
        case OP_ADD: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "add alloc");
            int rc = booki_add_f16(&g->tensors[n->in[0]],
                                   &g->tensors[n->in[1]], t);
            if (rc) return set_err(g, "add rc=%d", rc);
        } break;
        case OP_MUL: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "mul alloc");
            int rc = booki_mul_f16(&g->tensors[n->in[0]],
                                   &g->tensors[n->in[1]], t);
            if (rc) return set_err(g, "mul rc=%d", rc);
        } break;
        case OP_SOFTMAX: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "softmax alloc");
            int rc = booki_softmax_f16(&g->tensors[n->in[0]], t);
            if (rc) return set_err(g, "softmax rc=%d", rc);
        } break;
        case OP_SIN: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "sin alloc");
            int rc = booki_sin_f16(&g->tensors[n->in[0]], t);
            if (rc) return set_err(g, "sin rc=%d", rc);
        } break;
        case OP_COS: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "cos alloc");
            int rc = booki_cos_f16(&g->tensors[n->in[0]], t);
            if (rc) return set_err(g, "cos rc=%d", rc);
        } break;
        case OP_LEAKY_RELU: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "leaky_relu alloc");
            int rc = booki_leaky_relu_f16(&g->tensors[n->in[0]],
                                          n->p.leaky_relu.alpha, t);
            if (rc) return set_err(g, "leaky_relu rc=%d", rc);
        } break;
        case OP_SUB: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "sub alloc");
            int rc = booki_sub_f16(&g->tensors[n->in[0]],
                                   &g->tensors[n->in[1]], t);
            if (rc) return set_err(g, "sub rc=%d", rc);
        } break;
        case OP_INSTANCE_NORM: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "instance_norm alloc");
            const booki_tensor* bias = n->n_in >= 3 ? &g->tensors[n->in[2]] : NULL;
            int rc = booki_instance_norm_f16(&g->tensors[n->in[0]],
                                             &g->tensors[n->in[1]],
                                             bias, n->p.instance_norm.eps, t);
            if (rc) return set_err(g, "instance_norm rc=%d", rc);
        } break;
        case OP_CONV1D: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "conv1d alloc");
            const booki_tensor* bias = n->n_in >= 3 ? &g->tensors[n->in[2]] : NULL;
            int rc = booki_conv1d_f16(&g->tensors[n->in[0]],
                                      &g->tensors[n->in[1]], bias,
                                      n->p.conv1d.stride, n->p.conv1d.padding,
                                      n->p.conv1d.dilation, n->p.conv1d.groups, t);
            if (rc) return set_err(g, "conv1d rc=%d", rc);
        } break;
        case OP_CONV_TRANSPOSE1D: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "conv_transpose1d alloc");
            const booki_tensor* bias = n->n_in >= 3 ? &g->tensors[n->in[2]] : NULL;
            int rc = booki_conv_transpose1d_f16(&g->tensors[n->in[0]],
                                                &g->tensors[n->in[1]], bias,
                                                n->p.conv_transpose1d.stride,
                                                n->p.conv_transpose1d.padding,
                                                n->p.conv_transpose1d.output_padding,
                                                n->p.conv_transpose1d.dilation,
                                                n->p.conv_transpose1d.groups, t);
            if (rc) return set_err(g, "conv_transpose1d rc=%d", rc);
        } break;
        case OP_RESIZE1D: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "resize1d alloc");
            int rc = booki_resize1d_f16(&g->tensors[n->in[0]],
                                        n->p.resize1d.out_size,
                                        (booki_resize_mode)n->p.resize1d.mode, t);
            if (rc) return set_err(g, "resize1d rc=%d", rc);
        } break;
        case OP_LSTM: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "lstm alloc");
            if (!attn_scratch) attn_scratch = booki_arena_create(4ull * 1024 * 1024);
            const booki_tensor* bias = n->n_in >= 4 ? &g->tensors[n->in[3]] : NULL;
            const booki_tensor* h0t = n->p.lstm.h0 >= 0 ? &g->tensors[n->p.lstm.h0] : NULL;
            const booki_tensor* c0t = n->p.lstm.c0 >= 0 ? &g->tensors[n->p.lstm.c0] : NULL;
            int rc = booki_lstm_f16(&g->tensors[n->in[0]],
                                    &g->tensors[n->in[1]],
                                    &g->tensors[n->in[2]], bias,
                                    h0t, c0t, attn_scratch, t);
            if (rc) return set_err(g, "lstm rc=%d", rc);
        } break;
        case OP_EMBEDDING: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "embedding alloc");
            int rc = booki_embedding_f16(&g->tensors[n->in[0]],
                                         &g->tensors[n->in[1]], t);
            if (rc) return set_err(g, "embedding rc=%d", rc);
        } break;
        case OP_ATTENTION: {
            *t = materialize(g, n);
            if (!t->data) return set_err(g, "attention alloc");
            if (!attn_scratch) {
                attn_scratch = booki_arena_create(4ull * 1024 * 1024);
                if (!attn_scratch) return set_err(g, "attn scratch arena");
            }
            booki_arena_reset(attn_scratch);
            int rc = booki_multihead_attention_f16(&g->tensors[n->in[0]],
                                                   &g->tensors[n->in[1]],
                                                   &g->tensors[n->in[2]],
                                                   n->p.attention.heads,
                                                   attn_scratch, t);
            if (rc) return set_err(g, "attention rc=%d", rc);
        } break;
        }
    }

    *out_view = g->tensors[g->output];
    if (attn_scratch) booki_arena_destroy(attn_scratch);
    return 0;
}

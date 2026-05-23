/* booki_run_test_mlp — load test_mlp.gguf + test_mlp.topology.json,
 * build the graph, run forward, and compare against the y_ref tensor
 * packed alongside.  Exit code 0 = output matches reference within
 * fp16 tolerance, non-zero otherwise.
 *
 * Inputs:
 *   --gguf <path>      Booki model file
 *   --topology <path>  topology JSON
 *
 * The topology format is small enough that we parse it inline rather
 * than pull in a JSON library — string field lookups + a tiny array
 * walker is enough.
 */

#include "booki_graph.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NODES 64
#define MAX_NAME  64

typedef struct {
    char op[16];
    char name[MAX_NAME];
    char inputs[3][MAX_NAME];
    int  n_inputs;
    char dtype[8];
    int  shape[4];
    int  rank;
} parsed_node;

static int read_whole(const char* path, char** out, size_t* nlen) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(n + 1);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, n, f) != (size_t)n) { free(buf); fclose(f); return -1; }
    buf[n] = '\0';
    fclose(f);
    *out = buf; *nlen = n;
    return 0;
}

/* Skips whitespace and returns the next non-space character. */
static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r' || *p == ',') ++p;
    return p;
}

/* Tiny string-literal reader. Returns pointer after closing quote.
 * Out buffer must be at least cap bytes. */
static const char* read_str(const char* p, char* out, size_t cap) {
    p = skip_ws(p);
    if (*p != '"') return NULL;
    ++p;
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < cap) out[i++] = *p++;
    out[i] = '\0';
    if (*p == '"') ++p;
    return p;
}

/* Reads a "key": after skipping whitespace. */
static const char* read_key(const char* p, const char* key) {
    p = skip_ws(p);
    if (*p != '"') return NULL;
    char buf[64];
    p = read_str(p, buf, sizeof(buf));
    if (!p || strcmp(buf, key) != 0) return NULL;
    p = skip_ws(p);
    if (*p != ':') return NULL;
    return p + 1;
}

static const char* find_key(const char* p, const char* key) {
    /* Scan ahead in current object until we hit the named key or
     * the closing brace. Doesn't handle nested objects correctly —
     * sufficient for our flat topology format. */
    int depth = 1;
    while (*p && depth > 0) {
        if (*p == '{') depth++;
        else if (*p == '}') { depth--; if (depth == 0) return NULL; }
        else if (*p == '"') {
            const char* mark = p;
            char k[64]; p = read_str(p, k, sizeof(k));
            if (!p) return NULL;
            const char* after = skip_ws(p);
            if (*after == ':' && strcmp(k, key) == 0) return after + 1;
            p = mark + 1;
        } else ++p;
    }
    return NULL;
}

/* Parses topology JSON into parsed_node array. Returns count, or -1.  */
static int parse_topology(const char* text, parsed_node* nodes, int cap,
                          char* output_name, size_t out_cap) {
    const char* p = strstr(text, "\"nodes\"");
    if (!p) return -1;
    p = strchr(p, '[');
    if (!p) return -1;
    ++p;

    int n_nodes = 0;
    while (1) {
        p = skip_ws(p);
        if (*p == ']') { ++p; break; }
        if (*p != '{') return -1;
        ++p;
        if (n_nodes >= cap) return -1;
        parsed_node* nd = &nodes[n_nodes++];
        memset(nd, 0, sizeof(*nd));

        /* Walk to closing brace, picking up known fields. */
        int depth = 1;
        const char* obj = p;
        while (*p && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') { depth--; if (depth == 0) { ++p; break; } }
            ++p;
        }
        const char* end = p;
        (void)end;

        const char* op_p = find_key(obj, "op");
        if (op_p) read_str(op_p, nd->op, sizeof(nd->op));
        const char* name_p = find_key(obj, "name");
        if (name_p) read_str(name_p, nd->name, sizeof(nd->name));
        const char* dt = find_key(obj, "dtype");
        if (dt) read_str(dt, nd->dtype, sizeof(nd->dtype));

        const char* inp = find_key(obj, "inputs");
        if (inp) {
            inp = skip_ws(inp);
            if (*inp == '[') {
                ++inp;
                while (nd->n_inputs < 3) {
                    inp = skip_ws(inp);
                    if (*inp == ']') break;
                    inp = read_str(inp, nd->inputs[nd->n_inputs], MAX_NAME);
                    if (!inp) break;
                    nd->n_inputs++;
                }
            }
        }
        const char* sh = find_key(obj, "shape");
        if (sh) {
            sh = skip_ws(sh);
            if (*sh == '[') {
                ++sh;
                while (nd->rank < 4) {
                    sh = skip_ws(sh);
                    if (*sh == ']') break;
                    nd->shape[nd->rank++] = (int)strtol(sh, (char**)&sh, 10);
                }
            }
        }
    }

    const char* out_p = strstr(text, "\"output\"");
    if (out_p) {
        out_p = strchr(out_p, ':');
        if (out_p) read_str(out_p + 1, output_name, out_cap);
    }
    return n_nodes;
}

/* Lookup a built node id by name. */
static int find_id(parsed_node* nodes, int* node_ids, int n_nodes, const char* name) {
    for (int i = 0; i < n_nodes; ++i)
        if (strcmp(nodes[i].name, name) == 0) return node_ids[i];
    return -1;
}

int main(int argc, char** argv) {
    const char* gguf_path = NULL;
    const char* topo_path = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--gguf") && i + 1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "--topology") && i + 1 < argc) topo_path = argv[++i];
    }
    if (!gguf_path || !topo_path) {
        fprintf(stderr, "usage: booki_run_test_mlp --gguf <m> --topology <t>\n");
        return 2;
    }

    char err[256] = {0};
    booki_model* model = booki_model_open(gguf_path, err, sizeof(err));
    if (!model) { fprintf(stderr, "model: %s\n", err); return 1; }

    char* topo_text = NULL; size_t topo_n = 0;
    if (read_whole(topo_path, &topo_text, &topo_n) != 0) {
        fprintf(stderr, "read topology failed\n"); return 1;
    }

    parsed_node nodes[MAX_NODES];
    char output_name[MAX_NAME] = {0};
    int n_nodes = parse_topology(topo_text, nodes, MAX_NODES, output_name, sizeof(output_name));
    free(topo_text);
    if (n_nodes < 0) { fprintf(stderr, "topology parse failed\n"); return 1; }

    booki_arena* ws = booki_arena_create(4ull * 1024 * 1024);
    booki_graph* g  = booki_graph_create(ws, model);

    int ids[MAX_NODES] = {0};
    booki_tensor input_tensors[MAX_NODES] = {0};  /* keeps input bindings alive */

    for (int i = 0; i < n_nodes; ++i) {
        parsed_node* nd = &nodes[i];
        if (!strcmp(nd->op, "input")) {
            int64_t shape[4];
            for (int j = 0; j < nd->rank; ++j) shape[j] = nd->shape[j];
            ids[i] = booki_graph_input(g, nd->name, BOOKI_DTYPE_F16, nd->rank, shape);
            /* Bind from the GGUF (treat the input tensor as a weight for
             * test purposes — it carries the actual input data). */
            if (booki_model_tensor(model, nd->name, &input_tensors[i]) != 0) {
                fprintf(stderr, "input %s missing from gguf\n", nd->name); return 1;
            }
            booki_graph_bind(g, ids[i], &input_tensors[i]);
        } else if (!strcmp(nd->op, "weight")) {
            ids[i] = booki_graph_weight(g, nd->name);
        } else if (!strcmp(nd->op, "matmul")) {
            int a = find_id(nodes, ids, i, nd->inputs[0]);
            int b = find_id(nodes, ids, i, nd->inputs[1]);
            ids[i] = booki_graph_matmul(g, a, b);
        } else if (!strcmp(nd->op, "silu")) {
            int x = find_id(nodes, ids, i, nd->inputs[0]);
            ids[i] = booki_graph_silu(g, x);
        } else if (!strcmp(nd->op, "gelu")) {
            int x = find_id(nodes, ids, i, nd->inputs[0]);
            ids[i] = booki_graph_gelu(g, x);
        } else if (!strcmp(nd->op, "add")) {
            int a = find_id(nodes, ids, i, nd->inputs[0]);
            int b = find_id(nodes, ids, i, nd->inputs[1]);
            ids[i] = booki_graph_add(g, a, b);
        } else if (!strcmp(nd->op, "mul")) {
            int a = find_id(nodes, ids, i, nd->inputs[0]);
            int b = find_id(nodes, ids, i, nd->inputs[1]);
            ids[i] = booki_graph_mul(g, a, b);
        } else if (!strcmp(nd->op, "rmsnorm")) {
            int x = find_id(nodes, ids, i, nd->inputs[0]);
            int w = find_id(nodes, ids, i, nd->inputs[1]);
            ids[i] = booki_graph_rmsnorm(g, x, w, 1e-6f);
        } else {
            fprintf(stderr, "unknown op: %s\n", nd->op); return 1;
        }
        if (ids[i] < 0) {
            fprintf(stderr, "build %s/%s failed: %s\n", nd->op, nd->name,
                    booki_graph_last_error(g));
            return 1;
        }
    }

    int out_id = find_id(nodes, ids, n_nodes, output_name);
    if (out_id < 0) { fprintf(stderr, "output %s not built\n", output_name); return 1; }
    booki_graph_set_output(g, out_id);

    booki_tensor result;
    int rc = booki_graph_run(g, &result);
    if (rc) {
        fprintf(stderr, "graph_run: %s\n", booki_graph_last_error(g));
        return 1;
    }

    booki_tensor reference;
    if (booki_model_tensor(model, "y_ref", &reference) != 0) {
        fprintf(stderr, "y_ref missing from gguf\n"); return 1;
    }

    int64_t n = booki_tensor_elements(&reference);
    uint16_t* rp = (uint16_t*)result.data;
    uint16_t* refp = (uint16_t*)reference.data;
    float max_diff = 0.0f, mean = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float a = booki_f16_to_f32(rp[i]);
        float b = booki_f16_to_f32(refp[i]);
        float d = fabsf(a - b);
        if (d > max_diff) max_diff = d;
        mean += d;
    }
    mean /= (float)n;

    printf("elements         %lld\n", (long long)n);
    printf("max_abs_diff     %.6f\n", max_diff);
    printf("mean_abs_diff    %.6f\n", mean);
    printf("backend          %s\n", booki_backend_describe(booki_backend_active()));
    int ok = max_diff < 0.05f;
    printf("verdict          %s\n", ok ? "PASS" : "FAIL");

    booki_graph_destroy(g);
    booki_arena_destroy(ws);
    booki_model_close(model);
    return ok ? 0 : 1;
}

/* GGUF v3 reader.
 *
 * Layout reminder (little-endian throughout):
 *
 *   char[4]   magic = "GGUF"
 *   uint32    version (we support 3; 2 also works since the layout we
 *                       touch is unchanged)
 *   uint64    tensor_count
 *   uint64    metadata_kv_count
 *   <metadata kv>{kv_count}
 *   <tensor info>{tensor_count}
 *   <padding to alignment>
 *   <tensor data>
 *
 * Each metadata kv is:
 *   string    key                 // uint64 length + utf-8 bytes
 *   uint32    value_type           // gguf_type enum
 *   <value>                        // depends on type; arrays carry their own
 *                                  // type tag + length prefix
 *
 * Each tensor info is:
 *   string    name
 *   uint32    n_dims
 *   uint64[n_dims] shape
 *   uint32    type                 // ggml_type enum
 *   uint64    offset               // bytes from start of data section
 *
 * We don't try to support every ggml type yet — Kokoro fp16 needs F16 and
 * F32; INT8 lookups (for embeddings) and Q8_0 will come with quantization
 * work.
 */

#include "booki_native.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* GGUF metadata value types (subset). */
enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

/* GGML tensor types we accept right now. */
enum {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q8_0 = 8,  /* recognized but rejected for now */
    GGML_TYPE_Q4_K = 12, /* same */
    GGML_TYPE_I8   = 24,
    GGML_TYPE_I16  = 25,
    GGML_TYPE_I32  = 26,
    GGML_TYPE_I64  = 27,
};

#define ALIGN_TO 32

/* ------------------------------------------------------------------------- */
/* Internal representation                                                   */
/* ------------------------------------------------------------------------- */

typedef struct {
    char*    name;
    char*    value;   /* utf-8, owned */
} meta_string;

typedef struct {
    char*       name;
    booki_dtype dtype;
    int         rank;
    int64_t     shape[BOOKI_MAX_RANK];
    size_t      nbytes;
    const void* data;   /* pointer inside mmap'd region */
} model_tensor;

struct booki_model {
    int          fd;
    const void*  base;
    size_t       file_size;

    meta_string* meta;
    int          meta_count;

    model_tensor* tensors;
    int           tensor_count;
};

/* ------------------------------------------------------------------------- */
/* Reader state                                                              */
/* ------------------------------------------------------------------------- */

typedef struct {
    const uint8_t* p;
    const uint8_t* end;
    int            err;
    const char*    err_msg;
} reader;

static int rd_check(reader* r, size_t n) {
    if (r->err) return 0;
    if ((size_t)(r->end - r->p) < n) {
        r->err = 1; r->err_msg = "short read"; return 0;
    }
    return 1;
}

static uint32_t rd_u32(reader* r) {
    if (!rd_check(r, 4)) return 0;
    uint32_t v; memcpy(&v, r->p, 4); r->p += 4; return v;
}
static uint64_t rd_u64(reader* r) {
    if (!rd_check(r, 8)) return 0;
    uint64_t v; memcpy(&v, r->p, 8); r->p += 8; return v;
}
static int64_t  rd_i64(reader* r) { return (int64_t)rd_u64(r); }
static int32_t  rd_i32(reader* r) { return (int32_t)rd_u32(r); }

/* Reads a length-prefixed UTF-8 string, returning a malloc'd null-terminated copy. */
static char* rd_string(reader* r) {
    uint64_t n = rd_u64(r);
    if (r->err) return NULL;
    if (!rd_check(r, n)) return NULL;
    char* s = (char*)malloc(n + 1);
    if (!s) { r->err = 1; r->err_msg = "oom"; return NULL; }
    memcpy(s, r->p, n);
    s[n] = '\0';
    r->p += n;
    return s;
}

/* Skips a metadata value of [type]. Used for kv entries we don't expose. */
static void skip_value(reader* r, uint32_t type);

static void skip_array(reader* r) {
    uint32_t elem_type = rd_u32(r);
    uint64_t n = rd_u64(r);
    for (uint64_t i = 0; i < n && !r->err; ++i) skip_value(r, elem_type);
}

static void skip_value(reader* r, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL:
            if (rd_check(r, 1)) r->p += 1; break;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:
            if (rd_check(r, 2)) r->p += 2; break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32:
            if (rd_check(r, 4)) r->p += 4; break;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64:
            if (rd_check(r, 8)) r->p += 8; break;
        case GGUF_TYPE_STRING: { char* s = rd_string(r); free(s); break; }
        case GGUF_TYPE_ARRAY: skip_array(r); break;
        default:
            r->err = 1; r->err_msg = "unknown metadata type"; break;
    }
}

/* ------------------------------------------------------------------------- */
/* Type mapping                                                              */
/* ------------------------------------------------------------------------- */

static booki_dtype ggml_to_booki(uint32_t t, int* supported) {
    *supported = 1;
    switch (t) {
        case GGML_TYPE_F32: return BOOKI_DTYPE_F32;
        case GGML_TYPE_F16: return BOOKI_DTYPE_F16;
        case GGML_TYPE_I8:  return BOOKI_DTYPE_I8;
        case GGML_TYPE_I64: return BOOKI_DTYPE_I64;
        default: *supported = 0; return BOOKI_DTYPE_F32;
    }
}

/* ------------------------------------------------------------------------- */
/* Loader                                                                    */
/* ------------------------------------------------------------------------- */

static int set_err(char* out, size_t cap, const char* msg) {
    if (out && cap) { strncpy(out, msg, cap - 1); out[cap - 1] = 0; }
    return -1;
}

booki_model* booki_model_open(const char* path, char* err, size_t err_cap) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { set_err(err, err_cap, "cannot open file"); return NULL; }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd); set_err(err, err_cap, "fstat failed"); return NULL;
    }
    if ((size_t)st.st_size < 24) {
        close(fd); set_err(err, err_cap, "file too short"); return NULL;
    }
    void* base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        close(fd); set_err(err, err_cap, "mmap failed"); return NULL;
    }

    reader r = { (const uint8_t*)base, (const uint8_t*)base + st.st_size, 0, NULL };

    if (memcmp(r.p, "GGUF", 4) != 0) {
        munmap(base, st.st_size); close(fd);
        set_err(err, err_cap, "not a GGUF file");
        return NULL;
    }
    r.p += 4;
    uint32_t version = rd_u32(&r);
    if (version != 2 && version != 3) {
        munmap(base, st.st_size); close(fd);
        set_err(err, err_cap, "unsupported GGUF version");
        return NULL;
    }
    uint64_t tensor_count = rd_u64(&r);
    uint64_t meta_count   = rd_u64(&r);

    /* Metadata KV pairs */
    meta_string* meta = (meta_string*)calloc(meta_count, sizeof(meta_string));
    int saved_meta = 0;
    for (uint64_t i = 0; i < meta_count && !r.err; ++i) {
        char* key = rd_string(&r);
        uint32_t type = rd_u32(&r);
        if (r.err) { free(key); break; }
        if (type == GGUF_TYPE_STRING) {
            char* val = rd_string(&r);
            meta[saved_meta].name = key;
            meta[saved_meta].value = val;
            saved_meta++;
        } else {
            /* We only expose strings for now; skip everything else verbatim. */
            free(key);
            skip_value(&r, type);
        }
    }
    if (r.err) goto fail;

    /* Tensor info entries — we record name + dtype + shape + offset, then
     * compute pointer into the data region after alignment. */
    model_tensor* tensors = (model_tensor*)calloc(tensor_count, sizeof(model_tensor));
    if (!tensors) { r.err = 1; r.err_msg = "oom tensors"; goto fail; }

    for (uint64_t i = 0; i < tensor_count && !r.err; ++i) {
        model_tensor* t = &tensors[i];
        t->name = rd_string(&r);
        uint32_t n_dims = rd_u32(&r);
        if (n_dims > BOOKI_MAX_RANK) { r.err = 1; r.err_msg = "rank > BOOKI_MAX_RANK"; break; }
        t->rank = (int)n_dims;

        /* GGUF stores shape in ggml-order (innermost first). We re-order to
         * outermost-first (PyTorch/ONNX-style) so callers see what they
         * expect when they ask for shape[0]. */
        int64_t raw[BOOKI_MAX_RANK];
        for (uint32_t d = 0; d < n_dims; ++d) raw[d] = (int64_t)rd_u64(&r);
        for (int d = 0; d < t->rank; ++d) t->shape[d] = raw[t->rank - 1 - d];

        uint32_t gtype = rd_u32(&r);
        int supported = 0;
        t->dtype = ggml_to_booki(gtype, &supported);
        if (!supported) { r.err = 1; r.err_msg = "unsupported tensor dtype"; break; }

        uint64_t offset = rd_u64(&r);
        int64_t  elems = 1;
        for (int d = 0; d < t->rank; ++d) elems *= t->shape[d];
        t->nbytes = (size_t)elems * booki_dtype_size(t->dtype);
        /* offset is relative to the start of the data section, which begins
         * after the metadata + tensor info + alignment padding. We fix the
         * pointer up after the loop once we know where data starts. */
        t->data = (const void*)(uintptr_t)offset;
    }
    if (r.err) {
        for (uint64_t i = 0; i < tensor_count; ++i) free(tensors[i].name);
        free(tensors); goto fail;
    }

    /* Align reader position up to ALIGN_TO; that's the start of the data. */
    size_t cur = (size_t)(r.p - (const uint8_t*)base);
    size_t pad = (ALIGN_TO - (cur % ALIGN_TO)) % ALIGN_TO;
    if (cur + pad > (size_t)st.st_size) {
        for (uint64_t i = 0; i < tensor_count; ++i) free(tensors[i].name);
        free(tensors); r.err = 1; r.err_msg = "data alignment past EOF";
        goto fail;
    }
    const uint8_t* data_base = r.p + pad;
    size_t data_capacity = (size_t)st.st_size - (cur + pad);

    /* Resolve tensor pointers + bounds-check. */
    for (uint64_t i = 0; i < tensor_count; ++i) {
        model_tensor* t = &tensors[i];
        size_t off = (size_t)(uintptr_t)t->data;
        if (off + t->nbytes > data_capacity) {
            for (uint64_t j = 0; j < tensor_count; ++j) free(tensors[j].name);
            free(tensors); r.err = 1; r.err_msg = "tensor extends past EOF";
            goto fail;
        }
        t->data = data_base + off;
    }

    /* Build the model handle. */
    booki_model* m = (booki_model*)calloc(1, sizeof(*m));
    if (!m) { r.err = 1; r.err_msg = "oom model"; goto fail; }
    m->fd = fd;
    m->base = base;
    m->file_size = (size_t)st.st_size;
    m->meta = meta; m->meta_count = saved_meta;
    m->tensors = tensors; m->tensor_count = (int)tensor_count;
    return m;

fail:
    if (meta) {
        for (int i = 0; i < saved_meta; ++i) { free(meta[i].name); free(meta[i].value); }
        free(meta);
    }
    munmap(base, st.st_size); close(fd);
    set_err(err, err_cap, r.err_msg ? r.err_msg : "parse error");
    return NULL;
}

void booki_model_close(booki_model* m) {
    if (!m) return;
    for (int i = 0; i < m->meta_count; ++i) {
        free(m->meta[i].name); free(m->meta[i].value);
    }
    free(m->meta);
    for (int i = 0; i < m->tensor_count; ++i) free(m->tensors[i].name);
    free(m->tensors);
    if (m->base) munmap((void*)m->base, m->file_size);
    if (m->fd >= 0) close(m->fd);
    free(m);
}

int booki_model_tensor_count(const booki_model* m) {
    return m ? m->tensor_count : 0;
}

int booki_model_tensor(const booki_model* m, const char* name, booki_tensor* out) {
    if (!m || !name || !out) return -1;
    for (int i = 0; i < m->tensor_count; ++i) {
        if (strcmp(m->tensors[i].name, name) == 0) {
            const model_tensor* t = &m->tensors[i];
            memset(out, 0, sizeof(*out));
            out->dtype = t->dtype;
            out->rank  = t->rank;
            memcpy(out->shape, t->shape, sizeof(t->shape));
            int64_t s = 1;
            for (int d = t->rank - 1; d >= 0; --d) { out->strides[d] = s; s *= t->shape[d]; }
            out->data = (void*)t->data;   /* mmap'd, read-only via callers */
            out->nbytes = t->nbytes;
            return 0;
        }
    }
    return -2;
}

const char* booki_model_tensor_name(const booki_model* m, int i) {
    if (!m || i < 0 || i >= m->tensor_count) return NULL;
    return m->tensors[i].name;
}

const char* booki_model_meta_string(const booki_model* m, const char* key) {
    if (!m || !key) return NULL;
    for (int i = 0; i < m->meta_count; ++i) {
        if (strcmp(m->meta[i].name, key) == 0) return m->meta[i].value;
    }
    return NULL;
}

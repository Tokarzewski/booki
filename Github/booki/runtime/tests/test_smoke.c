/* Smoke test — runtime opens, errors are reported, version string sane.
 * Doesn't require a real model. */

#include "booki_baseline.h"

#include <stdio.h>
#include <string.h>

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return 1; } \
} while (0)

int main(void) {
    CHECK(booki_backend_name() != NULL,    "backend_name should be non-null");
    CHECK(booki_runtime_version() != NULL, "runtime_version should be non-null");
    CHECK(strlen(booki_backend_name()) > 0, "backend_name should be non-empty");

    /* Opening a bogus path should fail and set last_error. */
    booki_session* s = booki_open("/does/not/exist.onnx");
    CHECK(s == NULL,                       "open of bogus path should return NULL");
    CHECK(strlen(booki_last_error()) > 0,  "last_error should be set on failure");

    /* Closing NULL is a no-op. */
    booki_close(NULL);

    printf("smoke ok (backend: %s, version: %s)\n",
           booki_backend_name(), booki_runtime_version());
    return 0;
}

#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

namespace mula {
// The backends a caller may name. Each graph checks its own operations against the chosen device.
inline bool known_backend(const char * name) {
    return name && (!std::strcmp(name, "cpu") || !std::strcmp(name, "cuda") || !std::strcmp(name, "vulkan") || !std::strcmp(name, "metal"));
}

// "MUL_MAT f32[1024,4096,1,1] (bytes) <- ..." for backend refusal messages.
inline std::string describe_op(const ggml_tensor * t) {
    auto shape = [](const ggml_tensor * x) {
        return std::string(ggml_type_name(x->type)) + "[" + std::to_string(x->ne[0]) + "," + std::to_string(x->ne[1]) + "," +
               std::to_string(x->ne[2]) + "," + std::to_string(x->ne[3]) + "]" + (ggml_is_contiguous(x) ? "" : " strided");
    };
    std::string text = std::string(ggml_op_name(t->op)) + " " + shape(t) + " (" + std::to_string(ggml_nbytes(t)) + " bytes) <-";
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; ++i)
        text += " " + shape(t->src[i]) + " (" + std::to_string(ggml_nbytes(t->src[i])) + " bytes)";
    return text;
}

// Opens the named backend, preferring a discrete GPU over an integrated one and then the larger device
// (an iGPU reports shared system memory). GGML registers Metal as "MTL". Returns nullptr if none matches.
inline ggml_backend_t open_backend(const std::string & wanted, const char * who) {
    ggml_backend_dev_t chosen = nullptr;
    bool best_discrete = false;
    size_t best_memory = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        std::string name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (name == "mtl") name = "metal";
        if (name != wanted) continue;
        const bool discrete = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
        size_t free_bytes = 0, total = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total);
        if (!chosen || discrete > best_discrete || (discrete == best_discrete && total > best_memory)) {
            chosen = dev; best_discrete = discrete; best_memory = total;
        }
    }
    ggml_backend_t backend = chosen ? ggml_backend_dev_init(chosen, nullptr) : nullptr;
    if (backend)
        std::fprintf(stderr, "[mulacover] backend: %s (%s, %.1f GB, %s)\n", ggml_backend_name(backend),
                     ggml_backend_dev_description(chosen), double(best_memory) / (1024.0 * 1024.0 * 1024.0), who);
    return backend;
}
}

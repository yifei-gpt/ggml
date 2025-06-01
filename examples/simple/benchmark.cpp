#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <chrono>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
extern "C" void quantize_row_q4_0(const float * x, void * y, int64_t k); // ← declare quantizer
// ---------- Quantized Block Definitions ----------
#define QK4_1 32
struct block_q4_1 {
    ggml_fp16_t d;
    ggml_fp16_t m;
    uint8_t qs[QK4_1 / 2];
};

void init_q4_1_tensor(struct ggml_tensor *a) {
    int blocks = ggml_nelements(a) / QK4_1;
    for (int i = 0; i < blocks; ++i) {
        block_q4_1 *blk = (block_q4_1 *)((uint8_t *)a->data + i * sizeof(block_q4_1));
        blk->d = ggml_fp32_to_fp16(0.1f);
        blk->m = ggml_fp32_to_fp16(0.0f);
        for (int j = 0; j < QK4_1 / 2; ++j) {
            uint8_t lo = rand() % 16;
            uint8_t hi = rand() % 16;
            blk->qs[j] = (hi << 4) | lo;
        }
    }
}

#define QK4_0 32
struct block_q4_0 {
    ggml_fp16_t d;
    uint8_t qs[QK4_0 / 2];
};

void init_q4_0_tensor(struct ggml_tensor *a) {
    int blocks = ggml_nelements(a) / QK4_0;
    for (int i = 0; i < blocks; ++i) {
        block_q4_0 *blk = (block_q4_0 *)((uint8_t *)a->data + i * sizeof(block_q4_0));
        blk->d = ggml_fp32_to_fp16(0.1f);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            uint8_t lo = rand() % 16;
            uint8_t hi = rand() % 16;
            blk->qs[j] = (hi << 4) | lo;
        }
    }
}

#define QK8_0 32
struct block_q8_0 {
    ggml_fp16_t d;
    int8_t qs[QK8_0];
};

void init_q8_0_tensor_fixed(struct ggml_tensor *a) {
    int blocks = ggml_nelements(a) / QK8_0;
    for (int b = 0; b < blocks; ++b) {
        block_q8_0 *blk = (block_q8_0 *)((uint8_t *)a->data + b * sizeof(block_q8_0));
        blk->d = ggml_fp32_to_fp16(0.05f);
        for (int i = 0; i < QK8_0; ++i) {
            blk->qs[i] = 40;
        }
    }
}

// ---------- Helpers ----------
void fill_tensor_random(struct ggml_tensor *t) {
    float *data = (float *)t->data;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < ggml_nelements(t); ++i)
        data[i] = dist(rng);
}

void init_tensor_quant(struct ggml_tensor *t) {
    switch (t->type) {
        case GGML_TYPE_Q4_0:
            init_q4_0_tensor(t);
            break;
        case GGML_TYPE_Q4_1:
            init_q4_1_tensor(t);
            break;
        case GGML_TYPE_Q8_0:
            init_q8_0_tensor_fixed(t);
            break;
        default:
            fprintf(stderr, "Unsupported quant type in init_tensor_quant()\n");
            exit(1);
    }
}

void benchmark_matmul(enum ggml_type weight_type, const char* label, int n_repeat = 5) {
    const int DIM = 2048;
    const size_t memory_pool_size = 800 * 1024 * 1024;
    double total_time = 0.0;

    printf("\nBenchmarking %s (%d runs)\n", label, n_repeat);

    for (int iter = 0; iter < n_repeat; ++iter) {
        struct ggml_init_params params = { memory_pool_size, NULL, false };
        struct ggml_context *ctx = ggml_init(params);

        struct ggml_tensor *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIM, DIM);
        fill_tensor_random(input);

        struct ggml_tensor *weight;
        if (weight_type == GGML_TYPE_F32 || weight_type == GGML_TYPE_F16) {
            weight = ggml_new_tensor_2d(ctx, weight_type, DIM, DIM);
            fill_tensor_random(weight);
        } else {
            weight = ggml_new_tensor_2d(ctx, weight_type, DIM, DIM);
            init_tensor_quant(weight);
        }

        struct ggml_tensor *out = ggml_mul_mat(ctx, weight, input);
        struct ggml_cgraph *gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_set_output(out);

        auto start = std::chrono::high_resolution_clock::now();
        ggml_graph_compute_with_ctx(ctx, gf, 2);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        total_time += elapsed;

        ggml_free(ctx);
    }

    printf("%s avg matmul time: %.6f seconds\n", label, total_time / n_repeat);
}

int main(int argc, char** argv) {
    int repeat = 10;
    if (argc == 2) {
        repeat = std::stoi(argv[1]);
    }

    benchmark_matmul(GGML_TYPE_F32,  "FP32", repeat);
    benchmark_matmul(GGML_TYPE_F16,  "FP16", repeat);
    benchmark_matmul(GGML_TYPE_Q4_0, "Q4_0", repeat);
    benchmark_matmul(GGML_TYPE_Q4_1, "Q4_1", repeat);
    benchmark_matmul(GGML_TYPE_Q8_0, "Q8_0", repeat);
    return 0;
}

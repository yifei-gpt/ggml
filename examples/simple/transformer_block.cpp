#define GGML_IMPLEMENTATION
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <sys/resource.h>

// Quantization block constants
#define QK4_1 32
#define QK8_0 32

// 4-bit quantized block
struct block_q4_1 {
    ggml_fp16_t d;           // scale
    ggml_fp16_t m;           // zero point
    uint8_t qs[QK4_1 / 2];   // packed 4-bit values
};

// 8-bit quantized block
struct block_q8_0 {
    ggml_fp16_t d;           // scale
    int8_t qs[QK8_0];        // quantized values
};

void init_q4_1_tensor(struct ggml_tensor *a) {
    const int64_t nelements = ggml_nelements(a);
    const int blocks = nelements / QK4_1;
    for (int b = 0; b < blocks; ++b) {
        block_q4_1 *blk = (block_q4_1 *)((uint8_t *)a->data + b * sizeof(block_q4_1));
        blk->d = ggml_fp32_to_fp16(0.0f);
        blk->m = ggml_fp32_to_fp16(0.0f);
        for (int i = 0; i < QK4_1 / 2; ++i) {
            uint8_t lo = rand() % 16;
            uint8_t hi = rand() % 16;
            blk->qs[i] = (uint8_t)((hi << 4) | lo);
        }
    }
}

void init_q8_0_tensor_random(struct ggml_tensor *a) {
    const int64_t nelements = ggml_nelements(a);
    const int blocks = nelements / QK8_0;
    const float scale = 0.015f;  
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int8_t> dist(-64, 63); 
    for (int b = 0; b < blocks; ++b) {
        block_q8_0 *blk = (block_q8_0 *)((uint8_t *)a->data + b * sizeof(block_q8_0));
        blk->d = ggml_fp32_to_fp16(scale);
        for (int i = 0; i < QK8_0; ++i) {
            blk->qs[i] = dist(rng);
        }
    }
}

// Fill FP16 tensor with random float values in range [min, max]
void ggml_random_tensor_f16(struct ggml_tensor *tensor, float min_val, float max_val) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    ggml_fp16_t *data = (ggml_fp16_t *)tensor->data;
    const int64_t N = ggml_nelements(tensor);
    for (int64_t i = 0; i < N; ++i) {
        data[i] = ggml_fp32_to_fp16(dist(rng));
    }
}

// Fill FP32 tensor with random float values in range [min, max]
void ggml_random_tensor_f32(struct ggml_tensor *tensor, float min_val, float max_val) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    float *data = (float *)tensor->data;
    const int64_t N = ggml_nelements(tensor);
    for (int64_t i = 0; i < N; ++i) {
        data[i] = dist(rng);
    }
}

// Debug utility: print tensor shape
void print_tensor_shape(const char* name, const struct ggml_tensor* t) {
    printf("%s shape: [", name);
    bool first = true;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        if (t->ne[i] == 0) continue;
        if (!first) printf(", ");
        printf("%lld", (long long)t->ne[i]);
        first = false;
    }
    printf("]\n");
}

struct ModelTensors {
    struct ggml_tensor *attn_norm_w;
    struct ggml_tensor *ffn_norm_w;
    // Attention 
    struct ggml_tensor *wq_w;
    struct ggml_tensor *wk_w;
    struct ggml_tensor *wv_w;
    struct ggml_tensor *wo_w;
    // FFN 
    struct ggml_tensor *ffn_w_gate;
    struct ggml_tensor *ffn_w_up;
    struct ggml_tensor *ffn_w_down;
};

ModelTensors init_weights(
    struct ggml_context *ctx,
    int64_t n_embd, int64_t n_head_q, int64_t n_head_kv,
    int64_t n_embd_head, int64_t n_tokens, int64_t ffn_hidden_dim,
    const std::string &dtype
) {
    ModelTensors m{};

    // Always FP32 for LayerNorm weights
    m.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    m.ffn_norm_w  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_random_tensor_f32(m.attn_norm_w, 0.9f, 1.1f);
    ggml_random_tensor_f32(m.ffn_norm_w, 0.9f, 1.1f);

    // 2) Attention 
    const int64_t kv_dim = n_head_kv * n_embd_head;

    // --- wq ---
    if (dtype == "fp32") {
        m.wq_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        ggml_random_tensor_f32(m.wq_w, -0.1f, 0.1f);
    } else if (dtype == "fp16") {
        m.wq_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_embd);
        ggml_random_tensor_f16(m.wq_w, -0.1f, 0.1f);
    } else if (dtype == "int4") {
        assert((n_embd * n_embd) % QK4_1 == 0);
        m.wq_w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, n_embd, n_embd);
        init_q4_1_tensor(m.wq_w);
    } else { // int8
        assert((n_embd * n_embd) % QK8_0 == 0);
        m.wq_w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_embd, n_embd);
        init_q8_0_tensor_random(m.wq_w);
    }

    // --- wk ---
    if (dtype == "fp32") {
        m.wk_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, kv_dim);
        ggml_random_tensor_f32(m.wk_w, -0.1f, 0.1f);
    } else if (dtype == "fp16") {
        m.wk_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, kv_dim);
        ggml_random_tensor_f16(m.wk_w, -0.1f, 0.1f);
    } else if (dtype == "int4") {
        assert((n_embd * kv_dim) % QK4_1 == 0);
        m.wk_w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, n_embd, kv_dim);
        init_q4_1_tensor(m.wk_w);
    } else {
        assert((n_embd * kv_dim) % QK8_0 == 0);
        m.wk_w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_embd, kv_dim);
        init_q8_0_tensor_random(m.wk_w);
    }

    // --- wv ---
    if (dtype == "fp32") {
        m.wv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, kv_dim);
        ggml_random_tensor_f32(m.wv_w, -0.1f, 0.1f);
    } else if (dtype == "fp16") {
        m.wv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, kv_dim);
        ggml_random_tensor_f16(m.wv_w, -0.1f, 0.1f);
    } else if (dtype == "int4") {
        assert((n_embd * kv_dim) % QK4_1 == 0);
        m.wv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, n_embd, kv_dim);
        init_q4_1_tensor(m.wv_w);
    } else {
        assert((n_embd * kv_dim) % QK8_0 == 0);
        m.wv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_embd, kv_dim);
        init_q8_0_tensor_random(m.wv_w);
    }

    // --- wo ---
    if (dtype == "fp32") {
        m.wo_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        ggml_random_tensor_f32(m.wo_w, -0.1f, 0.1f);
    } else if (dtype == "fp16") {
        m.wo_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_embd);
        ggml_random_tensor_f16(m.wo_w, -0.1f, 0.1f);
    } else if (dtype == "int4") {
        assert((n_embd * n_embd) % QK4_1 == 0);
        m.wo_w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, n_embd, n_embd);
        init_q4_1_tensor(m.wo_w);
    } else {
        assert((n_embd * n_embd) % QK8_0 == 0);
        m.wo_w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_embd, n_embd);
        init_q8_0_tensor_random(m.wo_w);
    }

    // ffn_w_gate, ffn_w_up, ffn_w_down
    if (dtype == "fp32") {
        m.ffn_w_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, ffn_hidden_dim);
        m.ffn_w_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, ffn_hidden_dim);
        m.ffn_w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn_hidden_dim, n_embd);
        ggml_random_tensor_f32(m.ffn_w_gate, -0.1f, 0.1f);
        ggml_random_tensor_f32(m.ffn_w_up,   -0.1f, 0.1f);
        ggml_random_tensor_f32(m.ffn_w_down, -0.1f, 0.1f);
    } else if (dtype == "fp16") {
        m.ffn_w_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, ffn_hidden_dim);
        m.ffn_w_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, ffn_hidden_dim);
        m.ffn_w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ffn_hidden_dim, n_embd);
        ggml_random_tensor_f16(m.ffn_w_gate, -0.1f, 0.1f);
        ggml_random_tensor_f16(m.ffn_w_up,   -0.1f, 0.1f);
        ggml_random_tensor_f16(m.ffn_w_down, -0.1f, 0.1f);
    } else if (dtype == "int4") {
        assert((n_embd * ffn_hidden_dim) % QK4_1 == 0);
        assert((ffn_hidden_dim * n_embd) % QK4_1 == 0);
        m.ffn_w_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, n_embd, ffn_hidden_dim);
        m.ffn_w_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, n_embd, ffn_hidden_dim);
        m.ffn_w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, ffn_hidden_dim, n_embd);
        init_q4_1_tensor(m.ffn_w_gate);
        init_q4_1_tensor(m.ffn_w_up);
        init_q4_1_tensor(m.ffn_w_down);
    } else { // int8
        assert((n_embd * ffn_hidden_dim) % QK8_0 == 0);
        assert((ffn_hidden_dim * n_embd) % QK8_0 == 0);
        m.ffn_w_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_embd, ffn_hidden_dim);
        m.ffn_w_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_embd, ffn_hidden_dim);
        m.ffn_w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ffn_hidden_dim, n_embd);
        init_q8_0_tensor_random(m.ffn_w_gate);
        init_q8_0_tensor_random(m.ffn_w_up);
        init_q8_0_tensor_random(m.ffn_w_down);
    }

    return m;
}

// RMS‐Norm 
struct ggml_tensor *rms_norm_op(
    struct ggml_context *ctx, struct ggml_cgraph *gf,
    struct ggml_tensor *x, struct ggml_tensor *w, float eps
) {
    struct ggml_tensor *x_normed = ggml_rms_norm(ctx, x, eps);
    struct ggml_tensor *w_rep = ggml_repeat(ctx, w, x_normed);
    struct ggml_tensor *scaled = ggml_mul(ctx, w_rep, x_normed);
    return scaled;
}

struct ggml_tensor *apply_rope(
    struct ggml_context *ctx, struct ggml_cgraph *gf,
    struct ggml_tensor *qk_in, struct ggml_tensor *pos,
    int64_t n_rot, int64_t n_embd_head,
    float rope_freq_base, int64_t n_ctx
) {
    struct ggml_tensor *qk_perm = ggml_permute(ctx, qk_in, 0, 2, 1, 3);
    struct ggml_tensor *qk_roped = ggml_rope_ext(
        ctx, qk_perm, pos, nullptr,
        n_rot, 0, n_ctx,
        rope_freq_base,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f
    );
    struct ggml_tensor *qk_out = ggml_permute(ctx, qk_roped, 0, 2, 1, 3);
    return qk_out;
}

struct ggml_tensor *build_transformer_block_graph(
    struct ggml_context        *ctx,
    struct ggml_cgraph         *gf,
    const ModelTensors         &m,
    struct ggml_tensor         *inp_hidden_states,
    struct ggml_tensor         *inp_pos,
    int64_t                      n_embd,
    int64_t                      n_head_q,
    int64_t                      n_head_kv,
    int64_t                      n_embd_head,
    int64_t                      n_tokens,
    int64_t                      batch_size,
    int64_t                      ffn_hidden_dim,
    float                        rms_norm_eps,
    float                        rope_freq_base,
    int64_t                      n_ctx_example
) {
    // current_h = [n_embd, n_tokens, batch]
    struct ggml_tensor *current_h = inp_hidden_states;
    print_tensor_shape("current_h", current_h);
    // ---- LayerNorm → Attention ----
    struct ggml_tensor *attn_normed = rms_norm_op(
        ctx, gf, current_h, m.attn_norm_w, rms_norm_eps
    );
    ggml_build_forward_expand(gf, attn_normed);
    // Q/K/V 
    struct ggml_tensor *q_proj = ggml_mul_mat(ctx, m.wq_w, attn_normed);
    struct ggml_tensor *k_proj = ggml_mul_mat(ctx, m.wk_w, attn_normed);
    struct ggml_tensor *v_proj = ggml_mul_mat(ctx, m.wv_w, attn_normed);
    // ggml_build_forward_expand(gf, q_proj);
    // ggml_build_forward_expand(gf, k_proj);
    // ggml_build_forward_expand(gf, v_proj);
    // reshape → [n_embd_head, n_head, n_tokens, batch]
    struct ggml_tensor *q_heads = ggml_permute(
        ctx,
        ggml_reshape_4d(ctx, q_proj, n_embd_head, n_head_q, n_tokens, batch_size),
        0, 2, 1, 3
    );
    struct ggml_tensor *k_heads = ggml_permute(
        ctx,
        ggml_reshape_4d(ctx, k_proj, n_embd_head, n_head_kv, n_tokens, batch_size),
        0, 2, 1, 3
    );
    struct ggml_tensor *v_heads = ggml_permute(
        ctx,
        ggml_reshape_4d(ctx, v_proj, n_embd_head, n_head_kv, n_tokens, batch_size),
        0, 2, 1, 3
    );
    // ggml_build_forward_expand(gf, q_heads);
    // ggml_build_forward_expand(gf, k_heads);
    // ggml_build_forward_expand(gf, v_heads);

    // // // RoPE   
    struct ggml_tensor *q_roped = apply_rope(
        ctx, gf, q_heads, inp_pos,
        (int)n_embd_head, n_embd_head, rope_freq_base, n_ctx_example
    );
    struct ggml_tensor *k_roped = apply_rope(
        ctx, gf, k_heads, inp_pos,
        (int)n_embd_head, n_embd_head, rope_freq_base, n_ctx_example
    );  
    // ggml_build_forward_expand(gf, q_roped);
    // ggml_build_forward_expand(gf, k_roped); 



    // // // // GQA
    k_roped = ggml_repeat(ctx, k_roped, q_roped);
    v_heads = ggml_repeat(ctx, v_heads, q_roped);

    // ggml_build_forward_expand(gf, v_heads);
    // ggml_build_forward_expand(gf, k_roped); 

    // // // // Attention scores: K^T × Q
    struct ggml_tensor *attn_scores = ggml_mul_mat(ctx, k_roped, q_roped);
    // ggml_build_forward_expand(gf, attn_scores);
    const float scale = 1.0f / sqrtf((float)n_embd_head);
    struct ggml_tensor *attn_scaled = ggml_scale(ctx, attn_scores, scale);
    // ggml_build_forward_expand(gf, attn_scaled);
    struct ggml_tensor *attn_probs  = ggml_soft_max(ctx, attn_scaled);
    // ggml_build_forward_expand(gf, attn_probs);

    // // // // Weighted sum: attn_probs × V
    struct ggml_tensor *v_perm = ggml_permute(ctx, v_heads, 1, 0, 2, 3);
    struct ggml_tensor *v_cont = (v_perm->nb[0] == (int)ggml_type_size(v_perm->type))
                                    ? v_perm
                                    : ggml_cont(ctx, v_perm);
    struct ggml_tensor *attn_out_heads = ggml_mul_mat(ctx, attn_probs, v_cont);
    // ggml_build_forward_expand(gf, attn_out_heads);
    struct ggml_tensor *attn_perm = ggml_permute(ctx, attn_out_heads, 1, 2, 0, 3);
    // ggml_build_forward_expand(gf, attn_perm);
    struct ggml_tensor *attn_cont = ggml_cont(ctx, attn_perm);

    struct ggml_tensor *attn_merged = ggml_reshape_3d(ctx, attn_cont, n_embd, n_tokens, batch_size);
    struct ggml_tensor *attn_final  = ggml_mul_mat(ctx, m.wo_w, attn_merged);
    // ggml_build_forward_expand(gf, attn_final);
    struct ggml_tensor *attn_res    = ggml_add(ctx, current_h, attn_final);
    current_h = attn_res;

    // // // // ---- Feed‐forward ----   

    struct ggml_tensor *ffn_normed = rms_norm_op(
        ctx, gf, current_h, m.ffn_norm_w, rms_norm_eps
    );

    // gate_proj = W_gate × ffn_normed  0.55
    struct ggml_tensor *gate_proj = ggml_mul_mat(ctx, m.ffn_w_gate, ffn_normed);               
    ggml_build_forward_expand(gf, gate_proj);
    struct ggml_tensor *gate_proj_silu = ggml_silu_inplace(ctx, gate_proj); 
    // // ggml_build_forward_expand(gf, gate_proj); 
    // // up_proj = W_up × ffn_normed  0.55
    struct ggml_tensor *up_proj = ggml_mul_mat(ctx, m.ffn_w_up, ffn_normed);
    ggml_build_forward_expand(gf, up_proj);

    // // gate_proj *= up_proj 
    struct ggml_tensor *ffn_activated = ggml_mul(ctx, gate_proj_silu, up_proj); 
    ggml_build_forward_expand(gf, ffn_activated);
    // ffn_down = W_down × (gate_proj * up_proj) 0.55
    struct ggml_tensor *ffn_down = ggml_mul_mat(ctx, m.ffn_w_down, ffn_activated);   
    ggml_build_forward_expand(gf, ffn_down);

    // // current_h += ffn_down （Residual add）
    current_h = ggml_add(ctx, current_h, ffn_down); 

    ggml_build_forward_expand(gf, current_h);


    return current_h;
}


// main 
//  <n_tokens> <n_embd> <n_head_q> <n_head_kv>
//       <ffn_hidden_dim> <batch_size> <dtype> <N_INPUTS>
int main(int argc, char **argv) {
    ggml_time_init();

    if (argc < 10) {
        printf("Usage: %s <n_tokens> <n_embd> <n_head_q> <n_head_kv> "
               "<ffn_hidden_dim> <batch_size> <dtype> <N_INPUTS> <threads>\n",
               argv[0]);
        printf("  dtype: int4 | int8 | fp16 | fp32\n");
        return 1;
    }

    int64_t n_tokens        = std::stoll(argv[1]);
    int64_t n_embd          = std::stoll(argv[2]);
    int64_t n_head_q        = std::stoll(argv[3]);
    int64_t n_head_kv       = std::stoll(argv[4]);
    int64_t ffn_hidden_dim  = std::stoll(argv[5]);
    int64_t batch_size      = std::stoll(argv[6]);
    std::string dtype       = argv[7];
    int      N_INPUTS       = std::stoi(argv[8]);
    int      threads        = std::stoi(argv[9]);

    if (n_embd % n_head_q != 0) {
        fprintf(stderr, "Error: n_embd (%lld) must be divisible by n_head_q (%lld)\n",
                (long long)n_embd, (long long)n_head_q);
        return 1;
    }

    const int64_t n_embd_head = n_embd / n_head_q;
    const float rms_norm_eps   = 1e-5f;
    const float rope_freq_base = 500000.0f;
    const int   n_ctx_example  = 2048;
    const size_t memory_pool_size = 900LL * 1024 * 1024; // 900 MB
    const int n_runs = (N_INPUTS + batch_size - 1) / batch_size;

    printf("Configuration:\n");
    printf("  n_tokens       = %lld\n", (long long)n_tokens);
    printf("  n_embd         = %lld\n", (long long)n_embd);
    printf("  n_head_q       = %lld\n", (long long)n_head_q);
    printf("  n_head_kv      = %lld\n", (long long)n_head_kv);
    printf("  ffn_hidden_dim = %lld\n", (long long)ffn_hidden_dim);
    printf("  batch_size     = %lld\n", (long long)batch_size);
    printf("  dtype          = %s\n", dtype.c_str());
    printf("  N_INPUTS       = %d\n", N_INPUTS);
    printf("  threads        = %d\n", threads);
    printf("  n_runs         = %d\n\n", n_runs);

    struct ggml_init_params params_w = {
        memory_pool_size,
        NULL,
        false,
    };
    struct ggml_context *ctx_w = ggml_init(params_w);
    if (!ctx_w) {
        fprintf(stderr, "Failed to initialize ggml context for weights\n");
        return 1;
    }

    ModelTensors weights = init_weights(
        ctx_w, n_embd, n_head_q, n_head_kv, n_embd_head,
        n_tokens, ffn_hidden_dim, dtype
    );

    std::vector<double> mean_output((size_t)n_embd * (size_t)n_tokens, 0.0);

    int inputs_counted = 0;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int run_i = 0; run_i < n_runs; ++run_i) {
        int this_batch = std::min<int64_t>(
            batch_size, (int64_t)N_INPUTS - (int64_t)inputs_counted
        );

        struct ggml_init_params params_s = {
            memory_pool_size,
            NULL,
            false,
        };
        struct ggml_context *ctx_s = ggml_init(params_s);
        if (!ctx_s) {
            fprintf(stderr, "Failed to initialize ggml scratch context (run %d)\n", run_i);
            return 1;
        }

        struct ggml_cgraph *gf = ggml_new_graph(ctx_s);

        struct ggml_tensor *inp_hidden = ggml_new_tensor_3d(
            ctx_s, GGML_TYPE_F32, n_embd, n_tokens, this_batch
        );
        ggml_random_tensor_f32(inp_hidden, -0.5f, 0.5f);
        ggml_set_input(inp_hidden);

        struct ggml_tensor *inp_pos = ggml_new_tensor_1d(
            ctx_s, GGML_TYPE_I32, n_tokens
        );
        std::vector<int> pos_data(n_tokens);
        std::iota(pos_data.begin(), pos_data.end(), 0);
        memcpy(inp_pos->data, pos_data.data(), ggml_nbytes(inp_pos));
        ggml_set_input(inp_pos);

        auto run_start = std::chrono::high_resolution_clock::now();

        struct ggml_tensor *output_tensor = build_transformer_block_graph(
            ctx_s, gf, weights,
            inp_hidden, inp_pos,
            n_embd, n_head_q, n_head_kv, n_embd_head, n_tokens,
            this_batch, ffn_hidden_dim, rms_norm_eps, rope_freq_base, n_ctx_example
        );
        ggml_set_output(output_tensor);

        ggml_graph_compute_with_ctx(ctx_s, gf, threads);

        ggml_graph_print(gf);

        float *out_data = (float *)output_tensor->data;
        const int64_t single_out_elems = n_embd * n_tokens;
        for (int b = 0; b < this_batch && inputs_counted < N_INPUTS; ++b, ++inputs_counted) {
            int64_t base = (int64_t)b * single_out_elems;
            for (int64_t i = 0; i < single_out_elems; ++i) {
                mean_output[i] += (double)out_data[base + i];
            }
        }

        ggml_free(ctx_s);

        auto run_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(run_end - run_start).count();
        printf("Iteration %d done (batch_size = %d), time: %.6fs\n",
               run_i + 1, this_batch, elapsed);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(total_end - total_start).count();
    printf("\nTotal time for %d runs: %.6fs  (avg: %.6fs per run)\n",
           n_runs, total_elapsed, total_elapsed / n_runs);

    for (size_t i = 0; i < mean_output.size(); ++i) {
        mean_output[i] /= (double)N_INPUTS;
    }

    printf("Mean output over %d runs (first 10 elements):\n", N_INPUTS);
    for (int i = 0; i < std::min<int64_t>(10, (int64_t)mean_output.size()); ++i) {
        printf("  mean_output[%d] = %f\n", i, mean_output[i]);
    }

    {
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            printf("Peak resident set size: %ld KB\n", usage.ru_maxrss);
        } else {
            perror("getrusage");
        }
    }

    ggml_free(ctx_w);
    printf("Done.\n");
    return 0;
}

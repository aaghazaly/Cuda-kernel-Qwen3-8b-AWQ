// -----------------------------------------------------------------------------
// File: qwen3_awq_multi_gpu_infer.cu
//
// This is a multi-GPU version of the QWen3 AWQ (4-bit) inference example.
// We assume exactly 2 GPUs for simplicity. Layers 0–17 go on GPU0; layers 18–35 go
// on GPU1. The embedding lives on GPU0; the final unembedding (logits) lives on GPU1.
//
// Compile with (for example):
//    nvcc -O3 -arch=sm_80 qwen3_awq_multi_gpu_infer.cu -lcublas -o qwen3_awq_infer
//
// Run with:
//    ./qwen3_awq_infer /path/to/weights
//
// Requirements:
//   - CUDA 11+
//   - cuBLAS installed
//   - At least 2 GPUs with P2P enabled (compute capability ≥7.0).
//
// The code below is a single .cu file. Keep it in one place and compile as above.
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cmath>

// -----------------------------------------------------------------------------
// 1) Model hyperparameters (from your config.json)
// -----------------------------------------------------------------------------
static constexpr int    HIDDEN_SIZE          = 4096;
static constexpr int    INTERMEDIATE_SIZE    = 12288;
static constexpr int    NUM_ATTENTION_HEADS  = 32;
static constexpr int    HEAD_DIM             = 128;            // HIDDEN_SIZE / NUM_ATTENTION_HEADS
static constexpr int    NUM_HIDDEN_LAYERS    = 36;
static constexpr int    VOCAB_SIZE           = 151936;
static constexpr int    MAX_POS_EMBEDDINGS   = 40960;
static constexpr int    GROUP_SIZE           = 128;            // AWQ per-group quant (4 bits per weight)
static constexpr float  RMS_NORM_EPS         = 1e-6f;

// We assume exactly 2 GPUs. Split layers evenly:
//   Layers [0 .. SPLIT_IDX-1] → GPU 0
//   Layers [SPLIT_IDX .. NUM_HIDDEN_LAYERS-1] → GPU 1
static constexpr int NUM_GPUS    = 2;
static constexpr int SPLIT_IDX   = NUM_HIDDEN_LAYERS / NUM_GPUS; // = 36/2 = 18

// -----------------------------------------------------------------------------
// 2) Utility macros / error‐check
// -----------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                      \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA ERROR @ " << __FILE__ << ":" << __LINE__              \
                << " code=" << (int)err << " \"" << cudaGetErrorString(err)    \
                << "\"" << std::endl;                                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(expr)                                                    \
  do {                                                                         \
    cublasStatus_t stat = (expr);                                              \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "CUBLAS ERROR @ " << __FILE__ << ":" << __LINE__            \
                << " code=" << stat << std::endl;                              \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// -----------------------------------------------------------------------------
// 3) AWQ Dequantization Kernel (4-bit → FP16), per-group
// -----------------------------------------------------------------------------
__global__ void dequantAwqKernel(
    const uint8_t* __restrict__       qweight,      // [rows * packed_cols]
    const half* __restrict__          scales,       // [num_groups]
    const half* __restrict__          zp,           // [num_groups]
    half* __restrict__                fp16_out,     // [rows * cols]
    int                                rows,
    int                                cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int r = idx / cols;
    int c = idx % cols;
    int group_id = c / GROUP_SIZE;

    int packed_col_idx = c >> 1; // c/2
    uint8_t packed_byte = qweight[r * ((cols + 1) / 2) + packed_col_idx];
    uint8_t quant4;
    if ((c & 1) == 0) {
        quant4 = (packed_byte & 0x0F);
    } else {
        quant4 = (packed_byte >> 4) & 0x0F;
    }
    half s = scales[group_id];
    half z = zp  [group_id];
    half qh = __float2half((float)quant4);
    half sub = __hsub(qh, z);
    half res = __hmul(sub, s);
    fp16_out[r * cols + c] = res;
}

// -----------------------------------------------------------------------------
// 4) RMSNorm Kernel (FP16)
// -----------------------------------------------------------------------------
__global__ void rmsNormKernel(
    const half* __restrict__ x,    // [batch_size, hidden_size]
    const half* __restrict__ gamma,// [hidden_size]
    half* __restrict__       out,  // [batch_size, hidden_size]
    int                        hidden_size,
    float                      eps)
{
    extern __shared__ float shared_buf[];
    int batch_idx = blockIdx.x;    // one block per sample
    int tid       = threadIdx.x;

    const half* xs  = x + batch_idx * hidden_size;
    float acc = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = __half2float(xs[i]);
        acc += v * v;
    }
    shared_buf[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_buf[tid] += shared_buf[tid + stride];
        }
        __syncthreads();
    }
    float mean_sq = shared_buf[0] / float(hidden_size);
    float denom = rsqrtf(mean_sq + eps);
    half scale = __float2half(denom);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        half xi = xs[i];
        half gi = gamma[i];
        half tmp = __hmul(xi, scale);
        out[batch_idx * hidden_size + i] = __hmul(tmp, gi);
    }
}

// -----------------------------------------------------------------------------
// 5) Rotary Embedding Kernel (FP16, pairwise on head_dim)
// -----------------------------------------------------------------------------
__global__ void applyRotaryKernel(
    half* __restrict__       q_or_k,     // [num_heads, head_dim] for a single token
    const half* __restrict__ cos_table,  // [max_pos, head_dim/2]
    const half* __restrict__ sin_table,  // [max_pos, head_dim/2]
    int                       num_heads,
    int                       head_dim,
    int                       pos)
{
    int head_id = blockIdx.x;
    int lane   = threadIdx.x; // 0..(head_dim/2 - 1)

    int offset = head_id * head_dim;
    int pair_i = lane;
    int idx0   = offset + (2 * pair_i);
    int idx1   = offset + (2 * pair_i + 1);

    half Qi0 = q_or_k[idx0];
    half Qi1 = q_or_k[idx1];

    half cosv = cos_table[pos * (head_dim/2) + pair_i];
    half sinv = sin_table[pos * (head_dim/2) + pair_i];

    half t0 = __hsub(__hmul(Qi0, cosv), __hmul(Qi1, sinv));
    half t1 = __hadd(__hmul(Qi0, sinv), __hmul(Qi1, cosv));

    q_or_k[idx0] = t0;
    q_or_k[idx1] = t1;
}

// -----------------------------------------------------------------------------
// 6) Scaled-Dot-Product Attention Kernel (naive, FP16)
// -----------------------------------------------------------------------------
__global__ void attentionKernel(
    const half* __restrict__   Q,             // [num_heads, head_dim]
    const half* __restrict__   K_cache,       // [num_heads, max_seq_len, head_dim]
    const half* __restrict__   V_cache,       // [num_heads, max_seq_len, head_dim]
    half* __restrict__         context_out,   // [num_heads, head_dim]
    int                         num_heads,
    int                         head_dim,
    int                         max_seq_len,
    int                         current_len)
{
    extern __shared__ float smem[];
    int head_id = blockIdx.x;
    int tid     = threadIdx.x;

    const half* Qh         = Q         + head_id * head_dim;
    const half* Kh_base    = K_cache   + head_id * (max_seq_len * head_dim);
    const half* Vh_base    = V_cache   + head_id * (max_seq_len * head_dim);
    half*       Ch         = context_out + head_id * head_dim;

    // 1) Compute dot-products for each past position k
    for (int k = 0; k < current_len; ++k) {
        float acc = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            float qv = __half2float(Qh[i]);
            float kv = __half2float(Kh_base[k * head_dim + i]);
            acc += qv * kv;
        }
        smem[tid] = acc;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                smem[tid] += smem[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            float scale = 1.0f / sqrtf((float)head_dim);
            smem[0] = smem[0] * scale;
        }
        __syncthreads();
        if (tid == 0) {
            smem[k] = smem[0];
        }
        __syncthreads();
    }

    // 2) Softmax across smem[0..current_len-1]
    float local_max = -1e9f;
    if (tid == 0) {
        for (int k = 0; k < current_len; ++k) {
            local_max = fmaxf(local_max, smem[k]);
        }
        smem[current_len] = local_max;
    }
    __syncthreads();
    float max_score = smem[current_len];

    float local_sum = 0.0f;
    if (tid == 0) {
        for (int k = 0; k < current_len; ++k) {
            float ex = expf(smem[k] - max_score);
            smem[k] = ex;
            local_sum += ex;
        }
        smem[head_dim] = local_sum;
    }
    __syncthreads();
    float sum_exp = smem[head_dim];

    if (tid == 0) {
        for (int k = 0; k < current_len; ++k) {
            smem[k] = smem[k] / sum_exp;
        }
    }
    __syncthreads();

    // 3) Compute context[h, i] = Σ_{k=0..current_len-1} smem[k] * V_cache[h, k, i]
    float acc_ctx = 0.0f;
    for (int k = 0; k < current_len; ++k) {
        float w = smem[k];
        float vv = __half2float(Vh_base[k * head_dim + tid]);
        acc_ctx += w * vv;
    }
    Ch[tid] = __float2half(acc_ctx);
}

// -----------------------------------------------------------------------------
// 7) GeLU Kernel (FP16)
// -----------------------------------------------------------------------------
__global__ void geluKernel(half* __restrict__ x, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    float v = __half2float(x[idx]);
    float c = 0.70710678f; // 1/sqrt(2)
    float erfv = erff(v * c);
    float y = 0.5f * v * (1.0f + erfv);
    x[idx] = __float2half(y);
}

// -----------------------------------------------------------------------------
// 8) Final Softmax + Convert to FP32 (FP16 → FP32), Kernel to populate out_scores
// -----------------------------------------------------------------------------
__global__ void finalSoftmaxKernel(
    const half* __restrict__ logits,  // [vocab_size]
    float* __restrict__  out_scores,  // [vocab_size]
    int                   vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size) return;
    out_scores[idx] = __half2float(logits[idx]);
}

// -----------------------------------------------------------------------------
// 9) Host-Side: Read a binary file of known length into device memory
// -----------------------------------------------------------------------------
void readBinaryFileToDevice(const std::string& filename, void* d_ptr, size_t bytes) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "ERROR: could not open " << filename << "\n";
        std::exit(1);
    }
    std::vector<uint8_t> buffer(bytes);
    ifs.read(reinterpret_cast<char*>(buffer.data()), bytes);
    if (!ifs) {
        std::cerr << "ERROR: read failed for " << filename << "\n";
        std::exit(1);
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, buffer.data(), bytes, cudaMemcpyHostToDevice));
    ifs.close();
}

// -----------------------------------------------------------------------------
// 10) Data Structures for Multi-GPU
// -----------------------------------------------------------------------------

// (a) We group all per-layer weights into an AWQWeight struct.
struct AWQWeight {
    uint8_t* d_packed;  // [out_dim * ceil(in_dim/2)]
    half*    d_scale;   // [out_dim / GROUP_SIZE]
    half*    d_zp;      // [out_dim / GROUP_SIZE]
    half*    d_dq;      // [out_dim * in_dim]  (FP16 after dequant)
    int      out_dim;
    int      in_dim;
};

// (b) Each GPU has its own “scratch” / activation buffers and cuBLAS handle.
struct GPUContext {
    int           device_id;
    cudaStream_t  stream;
    cublasHandle_t cublas_handle;

    // Activation & scratch buffers (all on this GPU):
    half* d_hidden_state;      // [HIDDEN_SIZE]
    half* d_normed_state;      // [HIDDEN_SIZE]

    half* d_QKV;               // [3*HIDDEN_SIZE]
    half* d_Q;                 // [NUM_HEADS * HEAD_DIM]
    half* d_K_curr;            // [NUM_HEADS * HEAD_DIM]
    half* d_V_curr;            // [NUM_HEADS * HEAD_DIM]
    half* d_context;           // [NUM_HEADS * HEAD_DIM]
    half* d_attn_output;       // [HIDDEN_SIZE]

    half* d_mlp_intermediate;  // [INTERMEDIATE_SIZE]
    half* d_mlp_output;        // [HIDDEN_SIZE]

    half* d_logits;            // [VOCAB_SIZE]
    float* d_logits_fp32;      // [VOCAB_SIZE]

    half* d_cublas_scratch;    // if we need extra scratch for cuBLAS (not really used)
};

// -----------------------------------------------------------------------------
// 11) QWen3AWQInfer Class (Multi-GPU Version)
// -----------------------------------------------------------------------------
class QWen3AWQInfer {
public:
    QWen3AWQInfer(const std::string& weight_dir);
    ~QWen3AWQInfer();

    // prompt_ids: input token IDs
    // max_new_tokens: how many to generate
    // returns only newly generated token IDs
    std::vector<int> generateResponse(
        const std::vector<int>& prompt_ids,
        int max_new_tokens);

private:
    void initMultiGPU();                // check & enable P2P, init GPUContexts
    void loadAllWeights(const std::string& weight_dir);
    void dequantAllWeights();
    void allocateBuffers();
    void initCublas();

    // Helper: decide which GPU a given layer_id belongs to
    inline int layerToDevice(int layer_id) const {
        return (layer_id < SPLIT_IDX ? 0 : 1);
    }
    // Helper: if we need to copy hidden_state from one GPU to another
    void copyHiddenBetweenGPUs(int src_dev, int dst_dev);

    // Single‐layer forward on “correct” GPU. Must be called with cudaSetDevice(dev).
    void runOneLayer(int layer_id, int position_in_sequence);

    // Sample next token: do final RMSNorm + unembedding on GPU 1 ⇒ softmax/argmax on host
    int sampleNextToken();

    // Per-layer weights, stored in arrays of length NUM_HIDDEN_LAYERS
    AWQWeight layer_wqkv_[NUM_HIDDEN_LAYERS];
    AWQWeight layer_wo_   [NUM_HIDDEN_LAYERS];
    AWQWeight layer_w1_   [NUM_HIDDEN_LAYERS];
    AWQWeight layer_w2_   [NUM_HIDDEN_LAYERS];
    half* layer_rms_gamma_[NUM_HIDDEN_LAYERS]; // [HIDDEN_SIZE] each

    // Rotary tables (we store once per GPU, but identical data). We load them onto GPU0 and GPU1.
    half* d_cos_table_[NUM_GPUS]; // [MAX_POS_EMBEDDINGS * (HEAD_DIM/2)] each
    half* d_sin_table_[NUM_GPUS]; // [MAX_POS_EMBEDDINGS * (HEAD_DIM/2)] each

    // Embedding & Unembedding (AWQ)
    AWQWeight word_embedding_;   // on GPU0: [HIDDEN_SIZE, VOCAB_SIZE]
    AWQWeight unembed_;          // on GPU1: [VOCAB_SIZE, HIDDEN_SIZE]

    // ** KV caches: ** each layer on its own device, so we allocate per-layer on that layer’s device.
    //    K_cache[l]: [NUM_HEADS, MAX_POS_EMBEDDINGS, HEAD_DIM]  (flattened)
    //    V_cache[l]: [NUM_HEADS, MAX_POS_EMBEDDINGS, HEAD_DIM]
    half* d_K_cache_[NUM_HIDDEN_LAYERS];
    half* d_V_cache_[NUM_HIDDEN_LAYERS];
    int    current_seq_len_;    // overall current sequence length (prompt + generated so far)

    // Per-GPU contexts (each one holds its own cuBLAS handle + activation buffers)
    GPUContext gpu_ctx_[NUM_GPUS];

    // Host scratch for final softmax
    float*  h_logits_fp32_;
};

// -----------------------------------------------------------------------------
// 12) Implementation of QWen3AWQInfer Methods (Multi-GPU)
// -----------------------------------------------------------------------------

// 12.1) Constructor: check P2P, allocate buffers, load+dequantize weights, zero caches, build rotary tables
QWen3AWQInfer::QWen3AWQInfer(const std::string& weight_dir) {
    initMultiGPU();
    allocateBuffers();
    initCublas();
    loadAllWeights(weight_dir);
    dequantAllWeights();

    // Zero out K/V caches and set current_seq_len_ = 0
    current_seq_len_ = 0;
    for (int l = 0; l < NUM_HIDDEN_LAYERS; ++l) {
        int dev = layerToDevice(l);
        CUDA_CHECK(cudaSetDevice(dev));
        size_t kv_bytes = sizeof(half) * NUM_ATTENTION_HEADS * MAX_POS_EMBEDDINGS * HEAD_DIM;
        CUDA_CHECK(cudaMemsetAsync(d_K_cache_[l], 0, kv_bytes, gpu_ctx_[dev].stream));
        CUDA_CHECK(cudaMemsetAsync(d_V_cache_[l], 0, kv_bytes, gpu_ctx_[dev].stream));
    }
    for (int dev = 0; dev < NUM_GPUS; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaStreamSynchronize(gpu_ctx_[dev].stream));
    }

    // Build + copy rotary tables onto each GPU
    {
        std::vector<half> h_cos(MAX_POS_EMBEDDINGS * (HEAD_DIM/2));
        std::vector<half> h_sin(MAX_POS_EMBEDDINGS * (HEAD_DIM/2));
        for (int pos = 0; pos < MAX_POS_EMBEDDINGS; ++pos) {
            for (int i = 0; i < HEAD_DIM/2; ++i) {
                float inv_freq = 1.0f / powf(10000.0f, float(2 * i) / float(HEAD_DIM));
                float angle = float(pos) * inv_freq;
                h_cos[pos * (HEAD_DIM/2) + i] = __float2half(cosf(angle));
                h_sin[pos * (HEAD_DIM/2) + i] = __float2half(sinf(angle));
            }
        }
        size_t table_bytes = sizeof(half) * MAX_POS_EMBEDDINGS * (HEAD_DIM/2);
        for (int dev = 0; dev < NUM_GPUS; ++dev) {
            CUDA_CHECK(cudaSetDevice(dev));
            CUDA_CHECK(cudaMemcpyAsync(d_cos_table_[dev], h_cos.data(), table_bytes, cudaMemcpyHostToDevice, gpu_ctx_[dev].stream));
            CUDA_CHECK(cudaMemcpyAsync(d_sin_table_[dev], h_sin.data(), table_bytes, cudaMemcpyHostToDevice, gpu_ctx_[dev].stream));
        }
        for (int dev = 0; dev < NUM_GPUS; ++dev) {
            CUDA_CHECK(cudaSetDevice(dev));
            CUDA_CHECK(cudaStreamSynchronize(gpu_ctx_[dev].stream));
        }
    }

    // Allocate host scratch for final softmax
    h_logits_fp32_ = (float*)malloc(sizeof(float) * VOCAB_SIZE);
    assert(h_logits_fp32_);
}

// -----------------------------------------------------------------------------
// 12.2) Destructor: free all device pointers, cuBLAS handles, streams
// -----------------------------------------------------------------------------
QWen3AWQInfer::~QWen3AWQInfer() {
    // Destroy cuBLAS & CUDA streams
    for (int dev = 0; dev < NUM_GPUS; ++dev) {
        cudaSetDevice(dev);
        cublasDestroy(gpu_ctx_[dev].cublas_handle);
        cudaStreamDestroy(gpu_ctx_[dev].stream);
    }

    // Free per-layer weights & caches
    for (int l = 0; l < NUM_HIDDEN_LAYERS; ++l) {
        int dev = layerToDevice(l);
        CUDA_CHECK(cudaSetDevice(dev));

        CUDA_CHECK(cudaFree(layer_wqkv_[l].d_packed));
        CUDA_CHECK(cudaFree(layer_wqkv_[l].d_scale));
        CUDA_CHECK(cudaFree(layer_wqkv_[l].d_zp));
        CUDA_CHECK(cudaFree(layer_wqkv_[l].d_dq));

        CUDA_CHECK(cudaFree(layer_wo_[l].d_packed));
        CUDA_CHECK(cudaFree(layer_wo_[l].d_scale));
        CUDA_CHECK(cudaFree(layer_wo_[l].d_zp));
        CUDA_CHECK(cudaFree(layer_wo_[l].d_dq));

        CUDA_CHECK(cudaFree(layer_w1_[l].d_packed));
        CUDA_CHECK(cudaFree(layer_w1_[l].d_scale));
        CUDA_CHECK(cudaFree(layer_w1_[l].d_zp));
        CUDA_CHECK(cudaFree(layer_w1_[l].d_dq));

        CUDA_CHECK(cudaFree(layer_w2_[l].d_packed));
        CUDA_CHECK(cudaFree(layer_w2_[l].d_scale));
        CUDA_CHECK(cudaFree(layer_w2_[l].d_zp));
        CUDA_CHECK(cudaFree(layer_w2_[l].d_dq));

        CUDA_CHECK(cudaFree(layer_rms_gamma_[l]));

        CUDA_CHECK(cudaFree(d_K_cache_[l]));
        CUDA_CHECK(cudaFree(d_V_cache_[l]));
    }

    // Embedding on GPU0
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(word_embedding_.d_packed));
    CUDA_CHECK(cudaFree(word_embedding_.d_scale));
    CUDA_CHECK(cudaFree(word_embedding_.d_zp));
    CUDA_CHECK(cudaFree(word_embedding_.d_dq));

    // Unembedding on GPU1
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(unembed_.d_packed));
    CUDA_CHECK(cudaFree(unembed_.d_scale));
    CUDA_CHECK(cudaFree(unembed_.d_zp));
    CUDA_CHECK(cudaFree(unembed_.d_dq));

    // Rotary tables
    for (int dev = 0; dev < NUM_GPUS; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaFree(d_cos_table_[dev]));
        CUDA_CHECK(cudaFree(d_sin_table_[dev]));
    }

    // Activation & scratch buffers
    for (int dev = 0; dev < NUM_GPUS; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_hidden_state));
        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_normed_state));

        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_QKV));
        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_Q));
        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_K_curr));
        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_V_curr));
        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_context));
        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_attn_output));

        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_mlp_intermediate));
        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_mlp_output));

        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_logits));
        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_logits_fp32));

        CUDA_CHECK(cudaFree(gpu_ctx_[dev].d_cublas_scratch));
    }

    free(h_logits_fp32_);
}

// -----------------------------------------------------------------------------
// 12.3) initMultiGPU: Check + Enable Peer Access, Initialize GPUContexts
// -----------------------------------------------------------------------------
void QWen3AWQInfer::initMultiGPU() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < NUM_GPUS) {
        std::cerr << "ERROR: Need at least " << NUM_GPUS << " GPUs, but got " << device_count << "\n";
        std::exit(1);
    }

    // Enable P2P between GPU 0 and GPU 1
    int canPeer01 = 0, canPeer10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canPeer01, 0, 1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canPeer10, 1, 0));
    if (!canPeer01 || !canPeer10) {
        std::cerr << "ERROR: GPUs 0 and 1 cannot access each other via P2P. Make sure your system supports it.\n";
        std::exit(1);
    }
    // Enable peer access both ways
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));

    // Initialize GPUContexts
    for (int dev = 0; dev < NUM_GPUS; ++dev) {
        gpu_ctx_[dev].device_id = dev;
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaStreamCreate(&gpu_ctx_[dev].stream));

        // cuBLAS
        CUBLAS_CHECK(cublasCreate(&gpu_ctx_[dev].cublas_handle));
        CUBLAS_CHECK(cublasSetStream(gpu_ctx_[dev].cublas_handle, gpu_ctx_[dev].stream));
        CUBLAS_CHECK(cublasSetMathMode(gpu_ctx_[dev].cublas_handle, CUBLAS_TENSOR_OP_MATH));
    }
}

// -----------------------------------------------------------------------------
// 12.4) initCublas: (Already done in initMultiGPU, so this is empty here.)
// -----------------------------------------------------------------------------
void QWen3AWQInfer::initCublas() {
    // Nothing: cublas was created in initMultiGPU() per GPU.
}

// -----------------------------------------------------------------------------
// 12.5) allocateBuffers: Allocate all weights + activation buffers on correct GPUs
// -----------------------------------------------------------------------------
void QWen3AWQInfer::allocateBuffers() {
    // For each layer, allocate AWQ packed + scale + zp + dequantized on layer’s GPU
    for (int l = 0; l < NUM_HIDDEN_LAYERS; ++l) {
        int dev = layerToDevice(l);
        CUDA_CHECK(cudaSetDevice(dev));

        // 1) Wqkv: out_dim = 3*HIDDEN_SIZE, in_dim = HIDDEN_SIZE
        {
            int out_dim = 3 * HIDDEN_SIZE, in_dim = HIDDEN_SIZE;
            int packed_cols = (in_dim + 1) / 2;
            size_t packed_bytes = sizeof(uint8_t) * out_dim * packed_cols;
            size_t num_groups   = out_dim / GROUP_SIZE;

            layer_wqkv_[l].out_dim = out_dim;
            layer_wqkv_[l].in_dim  = in_dim;
            CUDA_CHECK(cudaMalloc(&layer_wqkv_[l].d_packed, packed_bytes));
            CUDA_CHECK(cudaMalloc(&layer_wqkv_[l].d_scale,  sizeof(half) * num_groups));
            CUDA_CHECK(cudaMalloc(&layer_wqkv_[l].d_zp,     sizeof(half) * num_groups));
            CUDA_CHECK(cudaMalloc(&layer_wqkv_[l].d_dq,     sizeof(half) * out_dim * in_dim));
        }
        // 2) Wo: out_dim = HIDDEN_SIZE, in_dim = HIDDEN_SIZE
        {
            int out_dim = HIDDEN_SIZE, in_dim = HIDDEN_SIZE;
            int packed_cols = (in_dim + 1) / 2;
            size_t packed_bytes = sizeof(uint8_t) * out_dim * packed_cols;
            size_t num_groups   = out_dim / GROUP_SIZE;

            layer_wo_[l].out_dim = out_dim;
            layer_wo_[l].in_dim  = in_dim;
            CUDA_CHECK(cudaMalloc(&layer_wo_[l].d_packed, packed_bytes));
            CUDA_CHECK(cudaMalloc(&layer_wo_[l].d_scale,  sizeof(half) * num_groups));
            CUDA_CHECK(cudaMalloc(&layer_wo_[l].d_zp,     sizeof(half) * num_groups));
            CUDA_CHECK(cudaMalloc(&layer_wo_[l].d_dq,     sizeof(half) * out_dim * in_dim));
        }
        // 3) W1: out_dim = INTERMEDIATE_SIZE, in_dim = HIDDEN_SIZE
        {
            int out_dim = INTERMEDIATE_SIZE, in_dim = HIDDEN_SIZE;
            int packed_cols = (in_dim + 1) / 2;
            size_t packed_bytes = sizeof(uint8_t) * out_dim * packed_cols;
            size_t num_groups   = out_dim / GROUP_SIZE;

            layer_w1_[l].out_dim = out_dim;
            layer_w1_[l].in_dim  = in_dim;
            CUDA_CHECK(cudaMalloc(&layer_w1_[l].d_packed, packed_bytes));
            CUDA_CHECK(cudaMalloc(&layer_w1_[l].d_scale,  sizeof(half) * num_groups));
            CUDA_CHECK(cudaMalloc(&layer_w1_[l].d_zp,     sizeof(half) * num_groups));
            CUDA_CHECK(cudaMalloc(&layer_w1_[l].d_dq,     sizeof(half) * out_dim * in_dim));
        }
        // 4) W2: out_dim = HIDDEN_SIZE, in_dim = INTERMEDIATE_SIZE
        {
            int out_dim = HIDDEN_SIZE, in_dim = INTERMEDIATE_SIZE;
            int packed_cols = (in_dim + 1) / 2;
            size_t packed_bytes = sizeof(uint8_t) * out_dim * packed_cols;
            size_t num_groups   = out_dim / GROUP_SIZE;

            layer_w2_[l].out_dim = out_dim;
            layer_w2_[l].in_dim  = in_dim;
            CUDA_CHECK(cudaMalloc(&layer_w2_[l].d_packed, packed_bytes));
            CUDA_CHECK(cudaMalloc(&layer_w2_[l].d_scale,  sizeof(half) * num_groups));
            CUDA_CHECK(cudaMalloc(&layer_w2_[l].d_zp,     sizeof(half) * num_groups));
            CUDA_CHECK(cudaMalloc(&layer_w2_[l].d_dq,     sizeof(half) * out_dim * in_dim));
        }
        // 5) RMSNorm gamma
        {
            CUDA_CHECK(cudaMalloc(&layer_rms_gamma_[l], sizeof(half) * HIDDEN_SIZE));
        }
        // 6) KV caches
        {
            size_t kv_bytes = sizeof(half) * NUM_ATTENTION_HEADS * MAX_POS_EMBEDDINGS * HEAD_DIM;
            CUDA_CHECK(cudaMalloc(&d_K_cache_[l], kv_bytes));
            CUDA_CHECK(cudaMalloc(&d_V_cache_[l], kv_bytes));
        }
    }

    // Rotary tables on each GPU
    for (int dev = 0; dev < NUM_GPUS; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        size_t table_bytes = sizeof(half) * MAX_POS_EMBEDDINGS * (HEAD_DIM/2);
        CUDA_CHECK(cudaMalloc(&d_cos_table_[dev], table_bytes));
        CUDA_CHECK(cudaMalloc(&d_sin_table_[dev], table_bytes));
    }

    // Embedding on GPU0
    {
        CUDA_CHECK(cudaSetDevice(0));
        int out_dim = HIDDEN_SIZE, in_dim = VOCAB_SIZE;
        int packed_cols = (in_dim + 1) / 2;
        size_t packed_bytes = sizeof(uint8_t) * out_dim * packed_cols;
        size_t num_groups   = out_dim / GROUP_SIZE;

        CUDA_CHECK(cudaMalloc(&word_embedding_.d_packed, packed_bytes));
        CUDA_CHECK(cudaMalloc(&word_embedding_.d_scale,  sizeof(half) * num_groups));
        CUDA_CHECK(cudaMalloc(&word_embedding_.d_zp,     sizeof(half) * num_groups));
        CUDA_CHECK(cudaMalloc(&word_embedding_.d_dq,     sizeof(half) * out_dim * in_dim));
        word_embedding_.out_dim = out_dim;
        word_embedding_.in_dim  = in_dim;
    }
    // Unembedding on GPU1
    {
        CUDA_CHECK(cudaSetDevice(1));
        int out_dim = VOCAB_SIZE, in_dim = HIDDEN_SIZE;
        int packed_cols = (in_dim + 1) / 2;
        size_t packed_bytes = sizeof(uint8_t) * out_dim * packed_cols;
        size_t num_groups   = out_dim / GROUP_SIZE;

        CUDA_CHECK(cudaMalloc(&unembed_.d_packed, packed_bytes));
        CUDA_CHECK(cudaMalloc(&unembed_.d_scale,  sizeof(half) * num_groups));
        CUDA_CHECK(cudaMalloc(&unembed_.d_zp,     sizeof(half) * num_groups));
        CUDA_CHECK(cudaMalloc(&unembed_.d_dq,     sizeof(half) * out_dim * in_dim));
        unembed_.out_dim = out_dim;
        unembed_.in_dim  = in_dim;
    }

    // Activation & scratch buffers on each GPU
    for (int dev = 0; dev < NUM_GPUS; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        gpu_ctx_[dev].d_hidden_state      = nullptr;
        gpu_ctx_[dev].d_normed_state      = nullptr;
        gpu_ctx_[dev].d_QKV               = nullptr;
        gpu_ctx_[dev].d_Q                 = nullptr;
        gpu_ctx_[dev].d_K_curr            = nullptr;
        gpu_ctx_[dev].d_V_curr            = nullptr;
        gpu_ctx_[dev].d_context           = nullptr;
        gpu_ctx_[dev].d_attn_output       = nullptr;
        gpu_ctx_[dev].d_mlp_intermediate  = nullptr;
        gpu_ctx_[dev].d_mlp_output        = nullptr;
        gpu_ctx_[dev].d_logits            = nullptr;
        gpu_ctx_[dev].d_logits_fp32       = nullptr;
        gpu_ctx_[dev].d_cublas_scratch    = nullptr;

        // hidden_state & normed_state
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_hidden_state,   sizeof(half) * HIDDEN_SIZE));
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_normed_state,   sizeof(half) * HIDDEN_SIZE));

        // QKV scratch
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_QKV,            sizeof(half) * 3 * HIDDEN_SIZE));
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_Q,              sizeof(half) * NUM_ATTENTION_HEADS * HEAD_DIM));
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_K_curr,         sizeof(half) * NUM_ATTENTION_HEADS * HEAD_DIM));
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_V_curr,         sizeof(half) * NUM_ATTENTION_HEADS * HEAD_DIM));
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_context,        sizeof(half) * NUM_ATTENTION_HEADS * HEAD_DIM));
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_attn_output,    sizeof(half) * HIDDEN_SIZE));

        // MLP scratch
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_mlp_intermediate, sizeof(half) * INTERMEDIATE_SIZE));
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_mlp_output,       sizeof(half) * HIDDEN_SIZE));

        // Logits scratch (only used on GPU1, but we allocate on both for symmetry)
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_logits,       sizeof(half) * VOCAB_SIZE));
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_logits_fp32,  sizeof(float) * VOCAB_SIZE));

        // cuBLAS scratch (optional)
        CUDA_CHECK(cudaMalloc(&gpu_ctx_[dev].d_cublas_scratch, sizeof(half) * HIDDEN_SIZE * HIDDEN_SIZE));
    }
}

// -----------------------------------------------------------------------------
// 12.6) loadAllWeights: Read all AWQ files into device buffers (on correct GPUs)
// -----------------------------------------------------------------------------
void QWen3AWQInfer::loadAllWeights(const std::string& weight_dir) {
    for (int l = 0; l < NUM_HIDDEN_LAYERS; ++l) {
        int dev = layerToDevice(l);
        CUDA_CHECK(cudaSetDevice(dev));

        // --- QKV ---
        {
            int out_dim = 3 * HIDDEN_SIZE, in_dim = HIDDEN_SIZE;
            int packed_cols = (in_dim + 1) / 2;
            size_t packed_bytes = sizeof(uint8_t) * out_dim * packed_cols;
            size_t num_groups   = out_dim / GROUP_SIZE;

            // Packed data
            {
                std::string fn = weight_dir + "/layer_" + std::to_string(l) + "_wqkv.bin";
                readBinaryFileToDevice(fn, layer_wqkv_[l].d_packed, packed_bytes);
            }
            // Scales
            {
                std::string fn = weight_dir + "/layer_" + std::to_string(l) + "_bqkv_scale.bin";
                readBinaryFileToDevice(fn, layer_wqkv_[l].d_scale, sizeof(half) * num_groups);
            }
            // Zero points
            {
                std::string fn = weight_dir + "/layer_" + std::to_string(l) + "_bqkv_zp.bin";
                readBinaryFileToDevice(fn, layer_wqkv_[l].d_zp, sizeof(half) * num_groups);
            }
        }
        // --- Wo ---
        {
            int out_dim = HIDDEN_SIZE, in_dim = HIDDEN_SIZE;
            int packed_cols = (in_dim + 1) / 2;
            size_t packed_bytes = sizeof(uint8_t) * out_dim * packed_cols;
            size_t num_groups   = out_dim / GROUP_SIZE;

            {
                std::string fn = weight_dir + "/layer_" + std::to_string(l) + "_wo.bin";
                readBinaryFileToDevice(fn, layer_wo_[l].d_packed, packed_bytes);
            }
            {
                std::string fn = weight_dir + "/layer_" + std::to_string(l) + "_wo_scale.bin";
                readBinaryFileToDevice(fn, layer_wo_[l].d_scale, sizeof(half) * num_groups);
            }
            {
                std::string fn = weight_dir + "/layer_" + std::to_string(l) + "_wo_zp.bin";
                readBinaryFileToDevice(fn, layer_wo_[l].d_zp, sizeof(half) * num_groups);
            }
        }
        // --- W1 (MLP) ---
        {
            int out_dim = INTERMEDIATE_SIZE, in_dim = HIDDEN_SIZE;
            int packed_cols = (in_dim + 1) / 2;
            size_t packed_bytes = sizeof(uint8_t) * out_dim * packed_cols;
            size_t num_groups   = out_dim / GROUP_SIZE;

            {
                std::string fn = weight_dir + "/layer_" + std::to_string(l) + "_w1.bin";
                readBinaryFileToDevice(fn, layer_w1_[l].d_packed, packed_bytes);
            }
            {
                std::string fn = weight_dir + "/layer_" + std::to_string(l) + "_w1_scale.bin";
                readBinaryFileToDevice(fn, layer_w1_[l].d_scale, sizeof(half) * num_groups);
            }
            {
                std::string fn = weight_dir + "/layer_" + std::to_string(l) + "_w1_zp.bin";
                readBinaryFileToDevice(fn, layer_w1_[l].d_zp, sizeof(half) * num_groups);
            }
        }
        // --- W2 (MLP) ---
        {
            int out_dim = HIDDEN_SIZE, in_dim = INTERMEDIATE_SIZE;
            int packed_cols = (in_dim + 1) / 2;
            size_t packed_bytes = sizeof(uint8_t) * out_dim * packed_cols;
            size_t num_groups   = out_dim / GROUP_SIZE;

            {
                std::string fn = weight_dir + "/layer_" + std::to_string(l) + "_w2.bin";
                readBinaryFileToDevice(fn, layer_w2_[l].d_packed, packed_bytes);
            }
            {
                std::string fn = weight_dir + "/layer_" + std::to_string(l) + "_w2_scale.bin";
                readBinaryFileToDevice(fn, layer_w2_[l].d_scale, sizeof(half) * num_groups);
            }
            {
                std::string fn = weight_dir + "/layer_" + std::to_string(l) + "_w2_zp.bin";
                readBinaryFileToDevice(fn, layer_w2_[l].d_zp, sizeof(half) * num_groups);
            }
        }
        // --- RMSNorm gamma ---
        {
            std::string fn = weight_dir + "/rms_norm_gamma_" + std::to_string(l) + ".bin";
            readBinaryFileToDevice(fn, layer_rms_gamma_[l], sizeof(half) * HIDDEN_SIZE);
        }
    }

    // --- Embedding on GPU0 ---
    {
        CUDA_CHECK(cudaSetDevice(0));
        int out_dim = HIDDEN_SIZE, in_dim = VOCAB_SIZE;
        int packed_cols = (in_dim + 1) / 2;
        size_t packed_bytes = sizeof(uint8_t) * out_dim * packed_cols;
        size_t num_groups   = out_dim / GROUP_SIZE;

        {
            std::string fn = weight_dir + "/word_embedding.bin";
            readBinaryFileToDevice(fn, word_embedding_.d_packed, packed_bytes);
        }
        {
            std::string fn = weight_dir + "/word_embed_scale.bin";
            readBinaryFileToDevice(fn, word_embedding_.d_scale, sizeof(half) * num_groups);
        }
        {
            std::string fn = weight_dir + "/word_embed_zp.bin";
            readBinaryFileToDevice(fn, word_embedding_.d_zp, sizeof(half) * num_groups);
        }
    }
    // --- Unembedding on GPU1 ---
    {
        CUDA_CHECK(cudaSetDevice(1));
        int out_dim = VOCAB_SIZE, in_dim = HIDDEN_SIZE;
        int packed_cols = (in_dim + 1) / 2;
        size_t packed_bytes = sizeof(uint8_t) * out_dim * packed_cols;
        size_t num_groups   = out_dim / GROUP_SIZE;

        {
            std::string fn = weight_dir + "/unembed.bin";
            readBinaryFileToDevice(fn, unembed_.d_packed, packed_bytes);
        }
        {
            std::string fn = weight_dir + "/unembed_scale.bin";
            readBinaryFileToDevice(fn, unembed_.d_scale, sizeof(half) * num_groups);
        }
        {
            std::string fn = weight_dir + "/unembed_zp.bin";
            readBinaryFileToDevice(fn, unembed_.d_zp, sizeof(half) * num_groups);
        }
    }
}

// -----------------------------------------------------------------------------
// 12.7) dequantAllWeights: Launch dequantAwqKernel on every AWQ tensor (on correct GPU)
// -----------------------------------------------------------------------------
void QWen3AWQInfer::dequantAllWeights() {
    const int TPB = 256;
    for (int l = 0; l < NUM_HIDDEN_LAYERS; ++l) {
        int dev = layerToDevice(l);
        CUDA_CHECK(cudaSetDevice(dev));

        // QKV
        {
            int rows = layer_wqkv_[l].out_dim;
            int cols = layer_wqkv_[l].in_dim;
            int total = rows * cols;
            int grid  = (total + TPB - 1) / TPB;
            dequantAwqKernel<<<grid, TPB, 0, gpu_ctx_[dev].stream>>>(
                layer_wqkv_[l].d_packed,
                layer_wqkv_[l].d_scale,
                layer_wqkv_[l].d_zp,
                layer_wqkv_[l].d_dq,
                rows, cols
            );
        }
        // Wo
        {
            int rows = layer_wo_[l].out_dim;
            int cols = layer_wo_[l].in_dim;
            int total = rows * cols;
            int grid  = (total + TPB - 1) / TPB;
            dequantAwqKernel<<<grid, TPB, 0, gpu_ctx_[dev].stream>>>(
                layer_wo_[l].d_packed,
                layer_wo_[l].d_scale,
                layer_wo_[l].d_zp,
                layer_wo_[l].d_dq,
                rows, cols
            );
        }
        // W1
        {
            int rows = layer_w1_[l].out_dim;
            int cols = layer_w1_[l].in_dim;
            int total = rows * cols;
            int grid  = (total + TPB - 1) / TPB;
            dequantAwqKernel<<<grid, TPB, 0, gpu_ctx_[dev].stream>>>(
                layer_w1_[l].d_packed,
                layer_w1_[l].d_scale,
                layer_w1_[l].d_zp,
                layer_w1_[l].d_dq,
                rows, cols
            );
        }
        // W2
        {
            int rows = layer_w2_[l].out_dim;
            int cols = layer_w2_[l].in_dim;
            int total = rows * cols;
            int grid  = (total + TPB - 1) / TPB;
            dequantAwqKernel<<<grid, TPB, 0, gpu_ctx_[dev].stream>>>(
                layer_w2_[l].d_packed,
                layer_w2_[l].d_scale,
                layer_w2_[l].d_zp,
                layer_w2_[l].d_dq,
                rows, cols
            );
        }
    }

    // Embedding on GPU0
    {
        CUDA_CHECK(cudaSetDevice(0));
        int rows = word_embedding_.out_dim;
        int cols = word_embedding_.in_dim;
        int total = rows * cols;
        int grid  = (total + TPB - 1) / TPB;
        dequantAwqKernel<<<grid, TPB, 0, gpu_ctx_[0].stream>>>(
            word_embedding_.d_packed,
            word_embedding_.d_scale,
            word_embedding_.d_zp,
            word_embedding_.d_dq,
            rows, cols
        );
    }

    // Unembedding on GPU1
    {
        CUDA_CHECK(cudaSetDevice(1));
        int rows = unembed_.out_dim;
        int cols = unembed_.in_dim;
        int total = rows * cols;
        int grid  = (total + TPB - 1) / TPB;
        dequantAwqKernel<<<grid, TPB, 0, gpu_ctx_[1].stream>>>(
            unembed_.d_packed,
            unembed_.d_scale,
            unembed_.d_zp,
            unembed_.d_dq,
            rows, cols
        );
    }

    for (int dev = 0; dev < NUM_GPUS; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaStreamSynchronize(gpu_ctx_[dev].stream));
    }
}

// -----------------------------------------------------------------------------
// 12.8) copyHiddenBetweenGPUs: copy d_hidden_state from src_dev → dst_dev
// -----------------------------------------------------------------------------
void QWen3AWQInfer::copyHiddenBetweenGPUs(int src_dev, int dst_dev) {
    // We assume P2P is enabled. Use cudaMemcpyPeerAsync.
    CUDA_CHECK(cudaMemcpyPeerAsync(
        gpu_ctx_[dst_dev].d_hidden_state, dst_dev,
        gpu_ctx_[src_dev].d_hidden_state, src_dev,
        sizeof(half) * HIDDEN_SIZE,
        gpu_ctx_[dst_dev].stream
    ));
    // Also copy the “hidden state” as the new input for dst_dev's next layer.
    // (No need to sync streams here; the next kernel on dst_dev will wait on its stream.)
}

// -----------------------------------------------------------------------------
// 12.9) runOneLayer: forward for a single transformer layer, on the layer’s device
//     Must call cudaSetDevice(dev) before calling this.
// -----------------------------------------------------------------------------
void QWen3AWQInfer::runOneLayer(int layer_id, int position_in_sequence) {
    int dev = layerToDevice(layer_id);
    cudaSetDevice(dev);
    cudaStream_t stream = gpu_ctx_[dev].stream;
    cublasHandle_t handle = gpu_ctx_[dev].cublas_handle;

    // 1) RMSNorm on d_hidden_state → d_normed_state
    {
        int threads = 256;
        int blocks  = 1;
        size_t shared = sizeof(float) * threads;
        rmsNormKernel<<<blocks, threads, shared, stream>>>(
            gpu_ctx_[dev].d_hidden_state,
            layer_rms_gamma_[layer_id],
            gpu_ctx_[dev].d_normed_state,
            HIDDEN_SIZE,
            RMS_NORM_EPS
        );
    }

    // 2) Wqkv projection: [3H, H] × [H, 1] → [3H, 1]
    {
        const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            3 * HIDDEN_SIZE,    // m
            1,                  // n
            HIDDEN_SIZE,        // k
            &alpha,
            layer_wqkv_[layer_id].d_dq, CUDA_R_16F, 3 * HIDDEN_SIZE,
            gpu_ctx_[dev].d_normed_state, CUDA_R_16F, HIDDEN_SIZE,
            &beta,
            gpu_ctx_[dev].d_QKV,        CUDA_R_16F, 3 * HIDDEN_SIZE,
            CUBLAS_COMPUTE_16F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }

    // 3) Split QKV and apply Rotary
    {
        half* Q_part = gpu_ctx_[dev].d_QKV;
        half* K_part = gpu_ctx_[dev].d_QKV + HIDDEN_SIZE;
        half* V_part = gpu_ctx_[dev].d_QKV + 2 * HIDDEN_SIZE;

        // Copy them into head-major buffers
        cudaMemcpyAsync(gpu_ctx_[dev].d_Q,      Q_part, sizeof(half) * HIDDEN_SIZE, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(gpu_ctx_[dev].d_K_curr, K_part, sizeof(half) * HIDDEN_SIZE, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(gpu_ctx_[dev].d_V_curr, V_part, sizeof(half) * HIDDEN_SIZE, cudaMemcpyDeviceToDevice, stream);

        // Rotary: one block per head, one thread per (pair index)
        dim3 grid(NUM_ATTENTION_HEADS);
        dim3 block(HEAD_DIM / 2);
        applyRotaryKernel<<<grid, block, 0, stream>>>(
            gpu_ctx_[dev].d_Q,
            d_cos_table_[dev], d_sin_table_[dev],
            NUM_ATTENTION_HEADS, HEAD_DIM, position_in_sequence
        );
        applyRotaryKernel<<<grid, block, 0, stream>>>(
            gpu_ctx_[dev].d_K_curr,
            d_cos_table_[dev], d_sin_table_[dev],
            NUM_ATTENTION_HEADS, HEAD_DIM, position_in_sequence
        );

        // 4) Write K_curr & V_curr into KV cache at index “position_in_sequence”
        int offset = position_in_sequence * HEAD_DIM;
        size_t head_stride = sizeof(half) * HEAD_DIM;
        for (int h = 0; h < NUM_ATTENTION_HEADS; ++h) {
            half* dstK = d_K_cache_[layer_id] + h * MAX_POS_EMBEDDINGS * HEAD_DIM + offset;
            half* dstV = d_V_cache_[layer_id] + h * MAX_POS_EMBEDDINGS * HEAD_DIM + offset;
            half* srcK = gpu_ctx_[dev].d_K_curr + h * HEAD_DIM;
            half* srcV = gpu_ctx_[dev].d_V_curr + h * HEAD_DIM;
            cudaMemcpyAsync(dstK, srcK, head_stride, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(dstV, srcV, head_stride, cudaMemcpyDeviceToDevice, stream);
        }
    }

    // 5) Attention: compute context → gpu_ctx_[dev].d_context
    {
        int shared_bytes = sizeof(float) * HEAD_DIM;
        dim3 grid(NUM_ATTENTION_HEADS), block(HEAD_DIM);
        attentionKernel<<<grid, block, shared_bytes, stream>>>(
            gpu_ctx_[dev].d_Q,
            d_K_cache_[layer_id],
            d_V_cache_[layer_id],
            gpu_ctx_[dev].d_context,
            NUM_ATTENTION_HEADS,
            HEAD_DIM,
            MAX_POS_EMBEDDINGS,
            current_seq_len_ + 1
        );
    }

    // 6) Merge heads & project with Wo
    {
        // d_context is [NUM_HEADS * HEAD_DIM] (4096). Copy → d_attn_output
        cudaMemcpyAsync(gpu_ctx_[dev].d_attn_output, gpu_ctx_[dev].d_context, sizeof(half) * HIDDEN_SIZE, cudaMemcpyDeviceToDevice, stream);

        const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            HIDDEN_SIZE,
            1,
            HIDDEN_SIZE,
            &alpha,
            layer_wo_[layer_id].d_dq, CUDA_R_16F, HIDDEN_SIZE,
            gpu_ctx_[dev].d_attn_output, CUDA_R_16F, HIDDEN_SIZE,
            &beta,
            gpu_ctx_[dev].d_attn_output, CUDA_R_16F, HIDDEN_SIZE,
            CUBLAS_COMPUTE_16F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }

    // 7) Residual: hidden_state += attn_output
    {
        int total = HIDDEN_SIZE;
        int TPB = 256;
        int grid = (total + TPB - 1) / TPB;
        auto addKernel = [] __device__ (const half* a, const half* b, half* out, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                out[idx] = __hadd(a[idx], b[idx]);
            }
        };
        addKernel<<<grid, TPB, 0, stream>>>(gpu_ctx_[dev].d_hidden_state, gpu_ctx_[dev].d_attn_output, gpu_ctx_[dev].d_hidden_state, HIDDEN_SIZE);
    }

    // 8) RMSNorm on updated hidden → d_normed_state
    {
        int threads = 256;
        int blocks  = 1;
        size_t shared = sizeof(float) * threads;
        rmsNormKernel<<<blocks, threads, shared, stream>>>(
            gpu_ctx_[dev].d_hidden_state,
            layer_rms_gamma_[layer_id],
            gpu_ctx_[dev].d_normed_state,
            HIDDEN_SIZE,
            RMS_NORM_EPS
        );
    }

    // 9) MLP:
    //    a) W1
    {
        const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            INTERMEDIATE_SIZE,
            1,
            HIDDEN_SIZE,
            &alpha,
            layer_w1_[layer_id].d_dq, CUDA_R_16F, INTERMEDIATE_SIZE,
            gpu_ctx_[dev].d_normed_state, CUDA_R_16F, HIDDEN_SIZE,
            &beta,
            gpu_ctx_[dev].d_mlp_intermediate, CUDA_R_16F, INTERMEDIATE_SIZE,
            CUBLAS_COMPUTE_16F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    //    b) GeLU
    {
        int total = INTERMEDIATE_SIZE;
        int TPB = 256;
        int grid = (total + TPB - 1) / TPB;
        geluKernel<<<grid, TPB, 0, stream>>>(gpu_ctx_[dev].d_mlp_intermediate, total);
    }
    //    c) W2
    {
        const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            HIDDEN_SIZE,
            1,
            INTERMEDIATE_SIZE,
            &alpha,
            layer_w2_[layer_id].d_dq, CUDA_R_16F, HIDDEN_SIZE,
            gpu_ctx_[dev].d_mlp_intermediate, CUDA_R_16F, INTERMEDIATE_SIZE,
            &beta,
            gpu_ctx_[dev].d_mlp_output, CUDA_R_16F, HIDDEN_SIZE,
            CUBLAS_COMPUTE_16F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    //    d) Residual: hidden += mlp_output
    {
        int total = HIDDEN_SIZE;
        int TPB = 256;
        int grid = (total + TPB - 1) / TPB;
        auto addKernel = [] __device__ (const half* a, const half* b, half* out, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                out[idx] = __hadd(a[idx], b[idx]);
            }
        };
        addKernel<<<grid, TPB, 0, stream>>>(gpu_ctx_[dev].d_hidden_state, gpu_ctx_[dev].d_mlp_output, gpu_ctx_[dev].d_hidden_state, HIDDEN_SIZE);
    }
    // (We do NOT increment current_seq_len_ here; that happens in generateResponse.)
}

// -----------------------------------------------------------------------------
// 12.10) sampleNextToken: Final RMSNorm + Unembed → Softmax+Argmax on host
// -----------------------------------------------------------------------------
int QWen3AWQInfer::sampleNextToken() {
    // After finishing last transformer layer (which is on GPU1), we do:
    int dev = 1;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaStream_t stream = gpu_ctx_[dev].stream;
    cublasHandle_t handle = gpu_ctx_[dev].cublas_handle;

    // 1) RMSNorm on hidden_state → normed_state (use last layer’s gamma)
    {
        int threads = 256;
        int blocks  = 1;
        size_t shared = sizeof(float) * threads;
        rmsNormKernel<<<blocks, threads, shared, stream>>>(
            gpu_ctx_[dev].d_hidden_state,
            layer_rms_gamma_[NUM_HIDDEN_LAYERS - 1],
            gpu_ctx_[dev].d_normed_state,
            HIDDEN_SIZE,
            RMS_NORM_EPS
        );
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 2) Unembedding: [VOCAB_SIZE, HIDDEN_SIZE] × [HIDDEN_SIZE,1] → [VOCAB_SIZE,1]
    {
        const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            VOCAB_SIZE,        // m
            1,                 // n
            HIDDEN_SIZE,       // k
            &alpha,
            unembed_.d_dq,     CUDA_R_16F, VOCAB_SIZE,
            gpu_ctx_[dev].d_normed_state, CUDA_R_16F, HIDDEN_SIZE,
            &beta,
            gpu_ctx_[dev].d_logits,       CUDA_R_16F, VOCAB_SIZE,
            CUBLAS_COMPUTE_16F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    // 3) Softmax + Argmax on host
    {
        // Convert FP16 logits → FP32 array
        finalSoftmaxKernel<<<(VOCAB_SIZE+255)/256, 256, 0, stream>>>(gpu_ctx_[dev].d_logits, gpu_ctx_[dev].d_logits_fp32, VOCAB_SIZE);
        CUDA_CHECK(cudaMemcpyAsync(h_logits_fp32_, gpu_ctx_[dev].d_logits_fp32, sizeof(float) * VOCAB_SIZE, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int best_id = 0;
        float best_score = h_logits_fp32_[0];
        for (int i = 1; i < VOCAB_SIZE; ++i) {
            if (h_logits_fp32_[i] > best_score) {
                best_score = h_logits_fp32_[i];
                best_id = i;
            }
        }
        return best_id;
    }
}

// -----------------------------------------------------------------------------
// 12.11) generateResponse: the user-facing API
// -----------------------------------------------------------------------------
std::vector<int> QWen3AWQInfer::generateResponse(
    const std::vector<int>& prompt_ids,
    int                     max_new_tokens)
{
    assert((int)prompt_ids.size() < MAX_POS_EMBEDDINGS);

    // 1) Process each prompt token on GPU0 (embedding + layers 0..17), then copy → GPU1 for layers 18..35
    for (size_t t = 0; t < prompt_ids.size(); ++t) {
        int token = prompt_ids[t];

        // 1.a) Embed on GPU0: copy column “token” from word_embedding_.d_dq → gpu_ctx_[0].d_hidden_state
        {
            CUDA_CHECK(cudaSetDevice(0));
            auto gatherKernel = [] __device__ (const half* __restrict__ W, half* __restrict__ out, int vocab_size, int hidden_size, int tok) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < hidden_size) {
                    out[idx] = W[tok + idx * vocab_size];
                }
            };
            int threads = 256;
            int grid = (HIDDEN_SIZE + threads - 1) / threads;
            gatherKernel<<<grid, threads, 0, gpu_ctx_[0].stream>>>(
                word_embedding_.d_dq,
                gpu_ctx_[0].d_hidden_state,
                VOCAB_SIZE,
                HIDDEN_SIZE,
                token
            );
            cudaStreamSynchronize(gpu_ctx_[0].stream);
        }

        // 1.b) Run layers 0..17 all on GPU0
        for (int l = 0; l < SPLIT_IDX; ++l) {
            runOneLayer(l, (int)t);
            current_seq_len_ = (int)t; // we have now placed t-th token’s KV in cache
        }

        // 1.c) Copy hidden_state from GPU0 → GPU1
        copyHiddenBetweenGPUs(0, 1);
        CUDA_CHECK(cudaStreamSynchronize(gpu_ctx_[1].stream));

        // 1.d) Run layers 18..35 on GPU1
        for (int l = SPLIT_IDX; l < NUM_HIDDEN_LAYERS; ++l) {
            runOneLayer(l, (int)t);
            current_seq_len_ = (int)t;
        }
    }

    // Now prompt is fully processed. current_seq_len_ = prompt_len-1.

    int prompt_len = (int)prompt_ids.size();
    std::vector<int> generated;
    generated.reserve(max_new_tokens);

    // 2) Autoregressive generation
    for (int step = 0; step < max_new_tokens; ++step) {
        int pos = prompt_len + step;

        // 2.a) We start with hidden_state on GPU1 (from last layer). We will run layers 0..17 on GPU0:
        //       So first, copy hidden_state from GPU1 → GPU0
        copyHiddenBetweenGPUs(1, 0);
        CUDA_CHECK(cudaStreamSynchronize(gpu_ctx_[0].stream));

        // 2.b) Run layers 0..17 on GPU0
        for (int l = 0; l < SPLIT_IDX; ++l) {
            runOneLayer(l, pos);
        }
        current_seq_len_ = pos;

        // 2.c) Copy hidden_state from GPU0 → GPU1
        copyHiddenBetweenGPUs(0, 1);
        CUDA_CHECK(cudaStreamSynchronize(gpu_ctx_[1].stream));

        // 2.d) Run layers 18..35 on GPU1
        for (int l = SPLIT_IDX; l < NUM_HIDDEN_LAYERS; ++l) {
            runOneLayer(l, pos);
        }
        current_seq_len_ = pos;

        // 2.e) Sample next token (on GPU1)
        int next_tok = sampleNextToken();
        generated.push_back(next_tok);

        // 2.f) Embed next_tok on GPU0 for next iteration, but we’ll have to copy it to GPU0:
        {
            // Embed on GPU0
            CUDA_CHECK(cudaSetDevice(0));
            auto gatherKernel = [] __device__ (const half* __restrict__ W, half* __restrict__ out, int vocab_size, int hidden_size, int tok) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < hidden_size) {
                    out[idx] = W[tok + idx * vocab_size];
                }
            };
            int threads = 256;
            int grid = (HIDDEN_SIZE + threads - 1) / threads;
            gatherKernel<<<grid, threads, 0, gpu_ctx_[0].stream>>>(
                word_embedding_.d_dq,
                gpu_ctx_[0].d_hidden_state,
                VOCAB_SIZE,
                HIDDEN_SIZE,
                next_tok
            );
            cudaStreamSynchronize(gpu_ctx_[0].stream);
        }

        // 2.g) Now we must copy that new hidden_state from GPU0 → GPU1 before the next iteration
        copyHiddenBetweenGPUs(0, 1);
        CUDA_CHECK(cudaStreamSynchronize(gpu_ctx_[1].stream));
    }

    return generated;
}

// -----------------------------------------------------------------------------
// 13) main: quick test harness
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_weight_folder>\n";
        return 1;
    }
    std::string weight_dir = argv[1];
    QWen3AWQInfer infer(weight_dir);

    // Example prompt: [ 151643 ]  (just BOS token)
    std::vector<int> prompt = { 151643 };
    int max_new_tokens = 20;

    std::vector<int> out = infer.generateResponse(prompt, max_new_tokens);

    std::cout << "Generated tokens: ";
    for (int tok : out) {
        std::cout << tok << " ";
    }
    std::cout << std::endl;
    return 0;
}

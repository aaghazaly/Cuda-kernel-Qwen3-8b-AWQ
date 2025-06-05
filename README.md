# Cuda-kernel-Qwen3-8b-AWQ
it supports multiple gpu , but i test it with nvidia tesla m60
Note: If you have more than 2 GPUs, you can generalize:

Set NUM_GPUS = <number_of_gpus>, SPLIT_IDX = NUM_HIDDEN_LAYERS / NUM_GPUS.

Assign layer l to device (l / SPLIT_IDX).

Copy hidden_state from one device to the next device in that chain whenever crossing the “split boundary.”

# How to Compile

1. Check Your GPUs
Make sure you have at least two GPUs:

nvidia-smi

You should see GPU 0 and GPU 1 (and more, if present).


2. Compile Command
In a terminal (Linux) with CUDA 11+ installed, run:

nvcc -O3 -arch=sm_80 qwen3_awq_multi_gpu_infer.cu -lcublas -o qwen3_awq_infer

Replace sm_80 with the compute capability of your GPUs (e.g. sm_75, sm_86, etc.) if needed.

-lcublas links cuBLAS for the GEMMs.



3. Make Sure Libraries Are in $LD_LIBRARY_PATH
Example:

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

Adjust if your CUDA is installed elsewhere.



# How to Run

Suppose your weights directory is /home/user/qwen3_weights/. Then:

./qwen3_awq_infer /home/user/qwen3_weights

You should see output like:

Generated tokens: 151645 105345 34567 8945  ...  (20 token IDs)

E) Weights Directory Layout (Example)

Your weights folder (/home/user/qwen3_weights/) must contain exactly these files:

layer_0_wqkv.bin
layer_0_bqkv_scale.bin
layer_0_bqkv_zp.bin
layer_0_wo.bin
layer_0_wo_scale.bin
layer_0_wo_zp.bin
layer_0_w1.bin
layer_0_w1_scale.bin
layer_0_w1_zp.bin
layer_0_w2.bin
layer_0_w2_scale.bin
layer_0_w2_zp.bin
rms_norm_gamma_0.bin

layer_1_wqkv.bin
layer_1_bqkv_scale.bin
layer_1_bqkv_zp.bin
layer_1_wo.bin
layer_1_wo_scale.bin
layer_1_wo_zp.bin
layer_1_w1.bin
layer_1_w1_scale.bin
layer_1_w1_zp.bin
layer_1_w2.bin
layer_1_w2_scale.bin
layer_1_w2_zp.bin
rms_norm_gamma_1.bin

  ...  (repeat for layers 2..35)

layer_35_wqkv.bin
layer_35_bqkv_scale.bin
layer_35_bqkv_zp.bin
layer_35_wo.bin
layer_35_wo_scale.bin
layer_35_wo_zp.bin
layer_35_w1.bin
layer_35_w1_scale.bin
layer_35_w1_zp.bin
layer_35_w2.bin
layer_35_w2_scale.bin
layer_35_w2_zp.bin
rms_norm_gamma_35.bin

word_embedding.bin
word_embed_scale.bin
word_embed_zp.bin

unembed.bin
unembed_scale.bin
unembed_zp.bin

Each *.bin file must contain exactly the raw bytes (no header) for the AWQ quantized data or the FP16 scale/zero-point arrays. For example:

layer_0_wqkv.bin has size = (3*4096) * ceil(4096/2) = 12288 * 2048 = 25,165,824 bytes.

layer_0_bqkv_scale.bin has size = (3*4096)/128 * sizeof(half) = 96 * 2 = 192 bytes (since 3×4096/128 = 96 groups).

layer_0_bqkv_zp.bin also 192 bytes.

word_embedding.bin has size = 4096 * ceil(151936/2) = 4096 * 75968 = 311,951,360 bytes, etc.

word_embed_scale.bin size = (4096 / 128) * 2 = 32 * 2 = 64 bytes.

word_embed_zp.bin also 64 bytes.

Likewise for unembed.bin (size = 151936 * ceil(4096/2) = 151936 * 2048 = 311,951,360), etc.


Make sure your files match exactly these




# How the Flow Works at Run-Time

1. Construction (main → QWen3AWQInfer infer(weight_dir)):

Calls initMultiGPU(), which:

Checks we have ≥ 2 GPUs.

Enables P2P between GPU 0 ⟷ GPU 1.

Creates a stream & cuBLAS handle on each GPU.


Calls allocateBuffers(), which cudaSetDevice(dev) appropriately for each sub-allocation:

AWQ weights + caches for each layer on the layer’s GPU.

Rotary tables on both GPUs.

Embedding on GPU 0; Unembedding on GPU 1.

Activation buffers (hidden_state, normed_state, QKV, Q, K_curr, V_curr, context, attn_output, mlp_intermediate, mlp_output, logits, logits_fp32) on each GPU.


Calls loadAllWeights(weight_dir), which does cudaMemcpy(...) into each device pointer (again, cudaSetDevice first).

Calls dequantAllWeights(), which launches dequantAwqKernel on GPU 0 for layers 0..17 + embedding, and on GPU 1 for layers 18..35 + unembedding.

Zeros out all KV caches (each on its GPU).

Builds rotary tables on the host, then copies them to both GPUs.

Allocates h_logits_fp32_ on the host.



2. generateResponse(prompt_ids, max_new_tokens):

For each prompt token:

Embed on GPU 0: gather the column for prompt_ids[t] from word_embedding_.d_dq.

Run layers 0..17 (GPU 0). After layer 17, gpu_ctx_[0].d_hidden_state is the “last‐token hidden.”

Copy hidden_state 0→1.

Run layers 18..35 (GPU 1). Now gpu_ctx_[1].d_hidden_state is the hidden of that token.

current_seq_len_ = t.


After prompt: current_seq_len_ = prompt_len−1. gpu_ctx_[1].d_hidden_state is the hidden of the last prompt token.

For each new generation step (max_new_tokens times):

1. Copy hidden 1→0 (because layer 0..17 are on GPU 0).


2. Run layers 0..17 on GPU 0 (runOneLayer(l, pos)). Now gpu_ctx_[0].d_hidden_state is the hidden of the new token before layer 18.


3. Copy hidden 0→1.


4. Run layers 18..35 on GPU 1. Now gpu_ctx_[1].d_hidden_state is the hidden of the new token after layer 35.


5. sampleNextToken() on GPU 1:

Final RMSNorm on gpu_ctx_[1].d_hidden_state.

Unembedding (FP16 GEMM) → gpu_ctx_[1].d_logits.

finalSoftmaxKernel converts to FP32 → gpu_ctx_[1].d_logits_fp32.

Copy to host → greedy argmax → next token ID.



6. Embed next token on GPU 0 (gather from word_embedding_.d_dq → gpu_ctx_[0].d_hidden_state).


7. Copy hidden 0→1 so that GPU 1 has the new token’s hidden to start its layer 18.


8. Repeat.





3. Return the vector of newly generated token IDs.

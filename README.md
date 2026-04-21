# **Proglot MAS LLM Optimization: Gemma 4 MoE Expert Pruning**

## **Overview**

This repository contains the experimental codebase for optimizing the `google/gemma-4-26b-a4b-it` Mixture-of-Experts (MoE) large language model. The project is developed as part of a Master's thesis in Data Science at MIPT for the "Proglot" electronic food diary multi-agent system (MAS).

The primary objective is to dramatically reduce the VRAM footprint and inference latency for deployment on consumer-grade Apple Silicon and local GPUs. By applying a custom REAP-inspired methodology, we successfully compressed the model from ~40GB down to ~12.3GB, while maintaining clinical accuracy in JSON generation, structured data parsing, and dietary planning.

## **Project Status & Conclusion**

Синтез структурного сжатия модели (MoE Pruning) и алгоритмического квантования (GGUF 4-bit) устраняет проблему высоких аппаратных требований для тяжелых мультимодальных LLM. Это делает технически и экономически целесообразным внедрение 26B параметров в real-time контур электронного дневника питания.

**Key Achievements:**
* Successfully profiled a fine-grained MoE architecture (128 experts per layer).
* Safely pruned 25% of the model's expert weights (dropping the bottom 32 experts per layer) with negligible loss to the target domain's active knowledge base.
* Quantized the resulting FP16 model (38.9 GB) down to a highly efficient Q4_K_M GGUF format (12.3 GB, 5.32 BPW).

## **Methodology**

The optimization pipeline consists of three core components:

1. **Activation Profiling (Forward Hooks):** We profiled the routing distributions across all layers using a custom multi-agent dataset (`waterfall_dataset_2.jsonl`). PyTorch `register_forward_hook` was attached to the Gemma 4 routers to capture top-2 expert selections.
2. **Surgical Expert Pruning (REAP):** Based on the generated `expert_heatmap.json`, we identified a strict long-tail distribution. We sliced the fused expert tensors (`down_proj` and `gate_up_proj`), permanently removing the 32 least utilized experts per layer, retaining the top 96.
3. **Quantization & Export:** The pruned FP16 weights were merged and quantized using `llama.cpp` into the `Q4_K_M` GGUF format for strict memory constraints.

## **Repository Structure**

* `reap_profiler.py`: Attaches PyTorch forward hooks to the `language_model.layers` routers to calculate expert importance scores based on a custom JSON-parsing dataset. Runs in FP16 dynamically distributed across GPUs.
* `reap_surgery.py`: Loads the unquantized weights into RAM, reads the heatmap distribution, modifies the PyTorch `nn.Parameter` tensors to drop targeted expert slices, and renormalizes the model configuration.
* `waterfall_dataset_2.jsonl`: The multi-agent calibration dataset containing planner, critic, and parser prompts.

## **Quick Start (Dockerized Cluster Execution)**

Due to the size of the base model, this pipeline is designed to be executed via Docker on a remote GPU cluster.

### **1. Model Acquisition**

Download the unquantized Gemma 4 26B MoE FP16 base weights directly into your cluster's Docker storage volume:

```bash
docker run -d \
  --name gemma-downloader \
  -v /your/remote/volume/:/models \
  -e HF_TOKEN="YOUR_HF_TOKEN" \
  python:3.11-slim \
  bash -c "pip install huggingface_hub && hf download google/gemma-4-26b-moe-it --local-dir /models/gemma-26b-weights"
```

### **2. Profiling Activations**

Build the PyTorch 2.4.0 container and run the calibration script to compute expert scores. Ensure you restrict `max_length` to prevent OOM on large context windows.

```bash
docker build -t gemma-profiler .
docker run -d \
  --name profiler-run \
  --gpus '"device=0,3"' \
  -v /your/remote/volume/:/models \
  gemma-profiler
```
*Output: `/models/expert_heatmap.json`*

### **3. Surgical Pruning**

Apply the structural modifications to drop 25% of the experts. This runs on CPU/RAM.

```bash
docker run -d \
  --name surgeon-run \
  -v /your/remote/volume/:/models \
  gemma-profiler python reap_surgery.py
```
*Output: `/models/gemma-26b-pruned-json/` (Pruned FP16 HuggingFace format)*

### **4. GGUF Quantization**

Use `llama.cpp` to convert and quantize the pruned model for local deployment:

```bash
# Convert to GGUF (FP16)
docker run --rm -it -v /your/remote/volume/:/models --entrypoint python3 ghcr.io/ggml-org/llama.cpp:full /app/convert_hf_to_gguf.py /models/gemma-26b-pruned-json --outfile /models/gemma-pruned-f16.gguf

# Quantize to 4-bit (Q4_K_M)
docker run --rm -it -v /your/remote/volume/:/models --entrypoint /app/llama-quantize ghcr.io/ggml-org/llama.cpp:full /models/gemma-pruned-f16.gguf /models/gemma-pruned-q4_k_m.gguf Q4_K_M
```

***

How does this new structure look to you—do you want to include the specific `llama.cpp` Docker commands in the Quick Start, or would you prefer to keep that section focused strictly on your custom Python scripts?
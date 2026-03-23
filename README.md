# **Proglot MAS LLM Optimization: MoE Expert Pruning**

## **Overview**

This repository contains the experimental codebase for optimizing the gpt-oss-20b Mixture-of-Experts (MoE) large language model. The project is developed as part of a Master's thesis in Data Science at MIPT for the "Proglot" electronic food diary multi-agent system (MAS).  
The primary objective is to reduce the inference latency (Time To First Token, TTFT) to under 500ms and limit VRAM consumption to under 10GB for consumer-grade GPU deployment, while maintaining \>95% clinical accuracy in generating dietary recommendations.

## **Project Status & Conclusion**

Синтез структурного сжатия модели и алгоритмического ускорения устраняет проблему высоких задержек в MAS. Это делает технически и экономически целесообразным внедрение тяжелых LLM в real-time контур электронного дневника питания.  
Целевые аппаратные метрики скорости (ускорение инференса в 4–5 раз) достигнуты. Влияние на удержание (retention) измеряется A/B тестированием в рамках текущего пилотного запуска.

## **Methodology**

The optimization pipeline consists of three core components:

1. **Expert Pruning (REAP):** Routing-Guided Expert Activation Pruning. We profile the activation norms of the 32 experts per layer using a domain-specific dataset (FoodyLLM/TADA) and surgically remove the 4 least utilized experts (12.5% reduction) for nutritional reasoning tasks.  
2. **Quantization:** Converting the pruned BF16 weights into 4-bit formats (MXFP4 / GGUF) for strict memory constraints.  
3. **Speculative Decoding:** Integrating a distilled gpt-oss-Nano model as a draft agent within the MAS to accelerate token generation.

## **Repository Structure**

* `scripts/profile\_reap.py`: Attaches PyTorch forward hooks to MoeLayer blocks to calculate expert importance scores via router logits and activation norms. Includes memory-mapping and checkpointing for execution on 32GB VRAM workstations.  
* `scripts/surgery.py`: Modifies the PyTorch nn.Module state dictionary, drops targeted expert tensors, and renormalizes the router weight matrices.  
* `scripts/evaluate.py`: Benchmarking scripts for measuring TTFT, throughput (TPS), and evaluating domain accuracy against the FoodyLLM QA baseline.  
* `notebooks/`: Jupyter notebooks containing exploratory data analysis of expert activation distributions and A/B test telemetry processing (retention/churn metrics).

## **Quick Start**

### **1\. Environment Setup**

``` bash
python3 \-m venv prune\_env  
source prune\_env/bin/activate  
pip install \-r requirements.txt
```

### **2\. Model Acquisition**

Download the unquantized, abliterated BF16 base weights (approx. 41 GB):  

``` bash
huggingface-cli download huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated \\  
  \--local-dir ./models/gpt-oss-20b-hf \\  
  \--resume-download
```

### **3\. Profiling Activations**

Run the calibration script to compute REAP scores. This will generate `reap\_profiling\_checkpoint.pt`.

``` bash
python scripts/profile\_reap.py \--model ./models/gpt-oss-20b-hf \--dataset data/foody\_qa\_calibration.json
```

### **4\. Pruning and Export**

Apply the structural modifications and save the new weights:  

```bash
python scripts/surgery.py \--checkpoint reap\_profiling\_checkpoint.pt \--output ./models/gpt-oss-20b-pruned  
```

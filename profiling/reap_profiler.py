import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

local_model_path = "/models/gemma-26b-weights"

print(f"Загрузка чистой FP16 модели из {local_model_path} на две видеокарты...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    device_map="auto" 
)

expert_heatmap = defaultdict(lambda: defaultdict(int))

def get_activation_hook(layer_idx):
    def hook(module, inputs, outputs):
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        _, selected_experts = torch.topk(logits, k=2, dim=-1)
        expert_ids, counts = torch.unique(selected_experts, return_counts=True)
        
        for exp_id, count in zip(expert_ids.tolist(), counts.tolist()):
            expert_heatmap[layer_idx][exp_id] += count
    return hook
print("Установка хуков на маршрутизаторы...")
hooks = []

if hasattr(model.model, "language_model"):
    target_layers = model.model.language_model.layers
else:
    target_layers = model.model.layers

for i, layer in enumerate(target_layers):
    # ИЗМЕНЕНИЕ ЗДЕСЬ: Ищем новый модуль 'router' вместо 'block_sparse_moe'
    if hasattr(layer, "router"):
        h = layer.router.register_forward_hook(get_activation_hook(i))
        hooks.append(h)
    else:
        print(f"Внимание: На слое {i} не найден маршрутизатор!")

dataset_file = "waterfall_dataset_2.jsonl"
print(f"Старт профилирования файла {dataset_file}...")

with open(dataset_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

with torch.no_grad():
    for idx, line in enumerate(lines):
        data = json.loads(line)
        full_text = (
            data.get("planner_prompt", "") + "\n" +
            data.get("planner_response", "") + "\n" +
            data.get("critic_prompt", "") + "\n" +
            data.get("critic_response", "") + "\n" +
            data.get("parser_prompt", "") + "\n" +
            data.get("parser_response", "")
        )
        
        # STRATEGY: Limit the context window strictly to save VRAM during forward pass
        # 2048 tokens is usually sufficient to capture the relevant expert activations for this task.
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to("cuda:0")
        
        try:
            _ = model(**inputs)
        except torch.cuda.OutOfMemoryError:
            print(f"Skipping sequence {idx} due to OOM (sequence length: {inputs.input_ids.shape[1]})")
            torch.cuda.empty_cache() # Try to recover memory before the next sequence
            continue
            
        if (idx + 1) % 10 == 0:
            print(f"Обработано {idx + 1}/{len(lines)} каскадов...")

for h in hooks:
    h.remove()

output_path = "/models/expert_heatmap.json"
with open(output_path, "w") as out:
    json.dump(expert_heatmap, out, indent=2)

print(f"\nГотово! Тепловая карта сохранена в {output_path}.")

import json
import torch
from transformers import AutoModelForCausalLM, AutoConfig

local_model_path = "/models/gemma-26b-weights"
save_path = "/models/gemma-26b-pruned-json"
heatmap_path = "/models/expert_heatmap.json"

EXPERTS_TO_KEEP = 96  # Из 128 оставляем 96 лучших (отрезаем 25%)

print("Загрузка тепловой карты...")
with open(heatmap_path, "r") as f:
    heatmap = json.load(f)

print(f"Загрузка оригинальной модели (на CPU/RAM, чтобы не забивать GPU)...")
# Для хирургии нам не нужны видеокарты, операции с тензорами легко пройдут в оперативной памяти
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    device_map="cpu" 
)

if hasattr(model.model, "language_model"):
    layers = model.model.language_model.layers
else:
    layers = model.model.layers

print("\n=== НАЧАЛО ОПЕРАЦИИ ПО ОБРЕЗКЕ ===")
for layer_idx, layer in enumerate(layers):
    str_idx = str(layer_idx)
    if str_idx not in heatmap:
        continue
        
    # Сортируем экспертов этого слоя по популярности
    layer_stats = heatmap[str_idx]
    sorted_experts = sorted(layer_stats.items(), key=lambda x: x[1], reverse=True)
    
    # Берем индексы топ-96 экспертов и сортируем их по возрастанию (чтобы сохранить порядок)
    kept_indices = [int(exp_id) for exp_id, count in sorted_experts[:EXPERTS_TO_KEEP]]
    kept_indices.sort()
    
    kept_tensor = torch.tensor(kept_indices, dtype=torch.long)
    
    # 1. Обрезаем Маршрутизатор (Router)
    if hasattr(layer, "router"):
        old_router_weight = layer.router.proj.weight.data
        # Оставляем только строки тех экспертов, которые выжили
        layer.router.proj.weight.data = old_router_weight[kept_tensor, :]
        
        # Если есть вектор масштабирования
        if hasattr(layer.router, "per_expert_scale") and layer.router.per_expert_scale is not None:
            layer.router.per_expert_scale.data = layer.router.per_expert_scale.data[kept_tensor]

    # 2. Обрезаем самих Экспертов (Fused Tensors)
    if hasattr(layer, "experts"):
        # down_proj — это сам Parameter, обращаемся к .data напрямую
        if hasattr(layer.experts, "down_proj"):
            layer.experts.down_proj.data = layer.experts.down_proj.data[kept_tensor]
            
        if hasattr(layer.experts, "gate_up_proj"):
            layer.experts.gate_up_proj.data = layer.experts.gate_up_proj.data[kept_tensor]

    if (layer_idx + 1) % 5 == 0:
        print(f"Слой {layer_idx + 1} успешно обрезан.")

print("\nОбновление конфигурации модели...")
# Говорим конфигурации, что экспертов теперь меньше
if hasattr(model.config, "num_local_experts"):
    model.config.num_local_experts = EXPERTS_TO_KEEP
elif hasattr(model.config, "num_experts"):
    model.config.num_experts = EXPERTS_TO_KEEP

print(f"Сохранение новой, облегченной модели в {save_path}...")
model.save_pretrained(save_path)
model.config.save_pretrained(save_path)

print("Операция завершена! Модель готова к конвертации в GGUF.")

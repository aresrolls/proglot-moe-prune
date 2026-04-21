from llama_cpp import Llama
import json
import random
import os

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf")

print("Загрузка модели в память Metal...")
# Укажите правильный путь к вашему .gguf файлу
llm = Llama(
    model_path=model_path, 
    n_gpu_layers=-1, 
    n_ctx=16384, 
    flash_attn=True,
    verbose=False 
)

def run_waterfall_generation(iterations=800, output_filename="waterfall_dataset_2.jsonl"):
    if os.path.exists(output_filename):
        print(f"Файл {output_filename} найден. Новые данные будут ДОБАВЛЯТЬСЯ в конец файла.")
    else:
        print(f"Файл {output_filename} не найден. Будет создан новый файл.")

    print(f"Запуск генерации {iterations} водопадов (Планировщик -> Критик -> Парсер)...")
    
    with open(output_filename, 'a', encoding='utf-8') as f:
        for i in range(iterations):
            print(f"\n--- Итерация {i+1}/{iterations} ---")
            
            cals = random.choice([1200, 1500, 1800, 2000, 2500])
            diet_type = random.choice(["веганская", "кето", "высокобелковая", "средиземноморская", "стандартная"])
            
            # === ШАГ 1: ПЛАНИРОВЩИК ===
            prompt_planner = f"Ты профессиональный диетолог. Составь подробную {diet_type} диету на {cals} калорий на один день."
            print("Планировщик пишет диету...")
            
            response_planner = llm(
                prompt=f"<start_of_turn>user\n{prompt_planner}<end_of_turn>\n<start_of_turn>model\n",
                max_tokens=8000,
                temperature=0.7
            )
            diet_plan = response_planner["choices"][0]["text"].strip()
            
            # === ШАГ 2: КРИТИК ===
            prompt_critic = (
                f"Проанализируй следующую диету: \n{diet_plan}\n\n"
                f"Найди в ней недостатки, проверь соответствие калориям ({cals}) и типу ({diet_type}). Укажи, что нужно исправить."
            )
            print("Критик ищет ошибки...")
            
            response_critic = llm(
                prompt=f"<start_of_turn>user\n{prompt_critic}<end_of_turn>\n<start_of_turn>model\n",
                max_tokens=8000,
                temperature=0.3 
            )
            critique = response_critic["choices"][0]["text"].strip()
            
            # === ШАГ 3: ПАРСЕР ===
            prompt_parser = (
                f"Оригинальная диета: {diet_plan}\n"
                f"Замечания критика: {critique}\n\n"
                f"Учитывая эти замечания, перепиши диету и выведи её СТРОГО в формате JSON без лишнего текста."
            )
            print("Парсер формирует JSON...")
            
            response_parser = llm(
                prompt=f"<start_of_turn>user\n{prompt_parser}<end_of_turn>\n<start_of_turn>model\n",
                max_tokens=8000,
                temperature=0.1 
            )
            final_json = response_parser["choices"][0]["text"].strip()
            
            # === ИСПРАВЛЕНИЕ ЗДЕСЬ ===
            # Теперь мы сохраняем и промпты (входы), и сгенерированные ответы (выходы)
            dataset_entry = {
                "planner_prompt": prompt_planner,
                "planner_response": diet_plan,
                "critic_prompt": prompt_critic,
                "critic_response": critique,
                "parser_prompt": prompt_parser,
                "parser_response": final_json
            }
            
            f.write(json.dumps(dataset_entry, ensure_ascii=False) + '\n')
            f.flush() 
            
            print(f"Итерация {i+1} успешно сохранена на диск.")
            
    print(f"\nГотово! Датасет сохранен в {output_filename}")

if __name__ == "__main__":
    run_waterfall_generation(iterations=800)
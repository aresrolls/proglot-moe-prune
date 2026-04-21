FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# ДОБАВЛЕН bitsandbytes для поддержки 4-битного квантования
RUN pip install transformers accelerate bitsandbytes

COPY reap_profiler.py .
COPY waterfall_dataset_2.jsonl .

CMD ["python", "reap_profiler.py"]

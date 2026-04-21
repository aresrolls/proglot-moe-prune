FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /app
RUN pip install transformers accelerate
COPY reap_surgery.py .
CMD ["python", "reap_surgery.py"]

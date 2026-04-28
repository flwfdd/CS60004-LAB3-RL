MODEL_PATH="/mlx/users/fanliwen.2333/playground/models/Qwen3-8B"
MODEL_NAME="Qwen3-8B"

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8080 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85
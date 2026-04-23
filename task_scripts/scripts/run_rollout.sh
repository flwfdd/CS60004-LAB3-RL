MODEL_PATH="/mlx/users/fanliwen.2333/playground/models/Qwen3-0.6B"
MODEL_NAME="Qwen3-0.6B"
PORT=8006
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.3

export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --weight-transfer-config '{"backend": "ipc"}' \
  --load-format dummy \
  --enforce-eager \
  

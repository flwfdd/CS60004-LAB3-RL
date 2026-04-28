MODEL_PATH="/mlx/users/fanliwen.2333/playground/models/Qwen3-0.6B"
MODEL_PATH="/mlx/users/fanliwen.2333/playground/code/CS60004-LAB3-RL/data/ckpt/grpo_trainbs64_minibs16_gs8_test_basedpoep10/best"
MODEL_NAME="Qwen3-0.6B"

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8006 \
  --max-model-len 8192 \
  --max-num-seqs 1024 \
  --gpu-memory-utilization 0.85
  
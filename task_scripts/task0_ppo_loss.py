# Task 0: PPO 损失函数实现示例（请替换为实际代码）
import torch

def compute_ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_ratio=0.2):
    # ==================== 代码开始 ====================
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    loss = -torch.min(ratio * advantages, clipped * advantages)
    # ==================== 代码结束 ====================
    return loss

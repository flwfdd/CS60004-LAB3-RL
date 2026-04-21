import torch


def compute_ppo_clip_loss(
    old_log_probs: torch.Tensor,  # 旧策略对数概率 (batch,)
    new_log_probs: torch.Tensor,  # 新策略对数概率 (batch,)
    advantages: torch.Tensor,  # 优势函数 (batch,)
    clip_ratio: float = 0.2,  # PPO clip 阈值
) -> torch.Tensor:
    # ====================== 代码开始 ======================
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    loss = -torch.min(ratio * advantages, clipped * advantages)
    # ====================== 代码结束 ======================
    return loss


def validate_ppo_implementation():
    torch.manual_seed(42)
    old_log_probs = torch.tensor([-0.5, -1.0, -0.3, -0.8], requires_grad=False)
    new_log_probs = torch.tensor([-0.4, -1.1, -0.25, -0.7], requires_grad=True)
    advantages = torch.tensor([1.2, -0.8, 0.9, -1.5])
    clip_ratio = 0.2
    loss = compute_ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_ratio)
    total_loss = loss.mean()
    total_loss.backward()
    student_grad = new_log_probs.grad.clone()  # type:ignore
    true_loss = torch.tensor([-1.3262053, 0.7238699, -0.9461439, 1.6577564])
    true_grad = torch.tensor([-0.3315513, 0.18096748, -0.23653598, 0.4144391])
    loss_correct = torch.allclose(loss, true_loss, atol=1e-4)
    grad_correct = torch.allclose(student_grad, true_grad, atol=1e-4)
    print(f"损失计算: {loss_correct} (loss: {loss} true_loss: {true_loss})")
    print(f"梯度计算: {grad_correct} (grad: {student_grad} true_grad: {true_grad})")


# 运行实验校验
if __name__ == "__main__":
    validate_ppo_implementation()

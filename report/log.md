# CS60004-LAB3-RL

## 数据预处理与基线实验

抽取 10000 条数据，其中 9000 条作为训练集，1000 条作为测试集。

如无特殊说明，均使用官方推荐的推理参数（`Temperature=0.6`，`TopP=0.95`，`TopK=20`），以及任务文档中给出的提示词模板加上一条 `You are a helpful assistant.` 系统提示词。

Qwen3-0.6B，最大输出长度 1024，批大小 128 测试 100 条数据，使用 `Transformers` 耗时 1:34 正确率 0.43 (43/100)，使用 `VLLM` 耗时 0:12 正确率 0.41。由于 VLLM 的速度优势，后续仅需推理的情况下都选择在 VLLM 上进行。

由于 100 条数据太少，多次运行发现随机性较大，VLLM 的推理速度还相对可观，因此后续决定将评测规模增加到 250 条。

观察到去掉系统提示词后，正确率会显著降低（0.457 -> 0.320），因此后续都保留系统提示词。

控制输出长度为变量：
1024: 0.432, 2048: 0.588, 4096: 0.692
可以看到输出长度的增加对正确率有显著提升。

控制温度为变量：
0.0: 0.468, 0.1: 0.476, 0.2: 0.436, 0.6: 0.432, 0.8: 0.420, 1.0: 0.368
尽管官方文档中说不要使用贪婪解码，但根据实验结果，在这个任务上降低温度能够带来一定的性能提升。

对于 Qwen3-8B 控制输出长度为变量：
1024: 0.412, 2048: 0.696, 4096: 0.832
可以发现在最大输出长度仅有 1024 的情况下，8B 模型的表现甚至还不如 0.6B 模型，但是随着输出长度的增加，8B 模型的正确率提升更为显著。

又想到有可能只是因为模型不知道有上下文限制所以比较啰嗦，因此又尝试了一下在提示词中加上一句 `You have limited tokens budget, so do not think too much.`，0.6B 模型的正确率从 0.432 提升到了 0.484，而 8B 模型的正确率从 0.412 提升到了 0.528。由此可见 8B 其实有着更强的性能，果然 PUA 还是有用的，后面都加上吧。这时候再降低温度到 0.1 正确率却只有 0.452，猜测之前降低温度带来的提升可能只是限制模型发散探索更早收敛结束，还是用官方推荐参数吧。

使用最终的配置，完整的 1000 条测试集上正确率 0.461。

Evaluated: 1000
Correct: 461 (0.4610)
Format Correct: 471 (0.4710)
Avg output len (tokens): all=803.5, correct=554.7, format_ok_wrong=608.8, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=366 (0.8262), format_correct=372 (0.8397), avg_output_len=596.2, avg_output_len_correct=514.1, avg_output_len_format_ok_wrong=540.8, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=87 (0.2047), format_correct=91 (0.2141), avg_output_len=953.1, avg_output_len_correct=691.8, avg_output_len_format_ok_wrong=710.8, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=8 (0.0606), format_correct=8 (0.0606), avg_output_len=1017.6, avg_output_len_correct=918.0, avg_output_len_format_ok_wrong=0.0, avg_output_len_format_wrong=1024.0

搞到一半发现评判标准是必须所有数字都出现，我错写成了只要不多用就行了可以少用，导致所有评测和 reward 其实都是错的，但影响没有很大，主要影响在数字 5 上。

Evaluated: 1000
Correct: 461 (0.4610)
Format Correct: 496 (0.4960)
Avg output len (tokens): all=795.0, correct=558.5, format_ok_wrong=613.0, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=362 (0.8172), format_correct=374 (0.8442), avg_output_len=599.3, avg_output_len_correct=518.4, avg_output_len_format_ok_wrong=595.0, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=89 (0.2094), format_correct=111 (0.2612), avg_output_len=933.2, avg_output_len_correct=691.5, avg_output_len_format_ok_wrong=615.0, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=10 (0.0758), format_correct=11 (0.0833), avg_output_len=1007.2, avg_output_len_correct=826.3, avg_output_len_format_ok_wrong=787.0, avg_output_len_format_wrong=1024.0

## Task 1：DPO 训练

让 0.6B 模型在 9000 条训练集上跑两遍，如果对于一条数据，一次成功一次失败，则作为一条 DPO 训练数据，这样一共获取到 1815 条，出于训练速度考虑，只取前 1000 条。

使用学习率 1e-6，批量大小 16（使用梯度累加），beta 0.1，发现训练 loss 很快就从 0.69 降到了 0，评测集上的表现一下就掉到了 0.35 左右。可以看到 rejected 样本的长度基本都打满了，说明错误样本基本上是因为超长，而 DPO 训练中计算 log_prob 时是句子级别的，计算方法就是将句子中的所有 token 的 log_prob 相加，这样的话由于正负样本间的长度差距过大，任何一点概率波动都会被放大，很快 loss 归零后就梯度消失了。

首先想到了一个优化方法，将 log_prob 计算方法从句子级别改为 token 级别，loss 下降明显平缓了许多，但是 chosen 的概率却一直下降，只是 rejected 的概率降的更多而已。并没有从根本上解决这个问题。

首要解决的问题不仅仅是提高正确率，更重要的是让模型降低输出长度，所以在之前的数据构造策略基础上，如果两次都成功，也作为一条 DPO 训练数据，其中长度较短的作为 chosen，构造得 4963 条，同样取前 1000 条。开始的时候正确率似乎还行，但很快 chosen 的概率就也持续下降，直接崩坏了。

加入了 NLL 损失函数后，chosen 概率下降得到了比较有效的缓解，训练完成后，正确率也提升到了 0.481，同时统计了格式正确率为 0.490，也就是只要正常输出的基本都正确了，而根据输出长度的统计，格式不正确的都是超长的。

又尝试了只使用 NLL 损失函数，此时等同于 SFT，虽然 DPO loss 已经涨飞了，但是最终正确率有 0.488，说明 SFT 也能得到比较好的效果。


![wandb_dpo](imgs/wandb_dpo.png)

## Task 3：vLLM 加速

由于前期测试的时候发现训练速度实在慢的有点难以忍受，正所谓工欲善其事必先利其器，所以我们先把 Task 3 提到前面来，优化好 Infra 再开始大规模的训练探索。

https://docs.vllm.ai/en/latest/training/weight_transfer/ipc/

https://docs.vllm.ai/en/latest/examples/rl/rlhf_http_ipc/


5090上：
    rollout_backend = "vllm_http_ipc"
    eval_backend = rollout_backend
    rollout_sync_freq = 1
    rollout_base_url = os.getenv("ROLLOUT_BASE_URL", "http://localhost:8006")
    rollout_model_name = os.getenv("MODEL_NAME_0P6B", "Qwen3-0.6B")
    rollout_api_key = os.getenv("ROLLOUT_API_KEY", "EMPTY")
    vllm_gpu_memory_utilization = 0.3
    vllm_enforce_eager = True
    vllm_max_model_len = 8192
    shuffle_rollout = True
    async_rollout = True
    prompt_batch_size = 8
    group_size = 8
    train_batch_size = prompt_batch_size * group_size
    mini_batch_size = 32
    micro_batch_size = 2
    rollout_logp_micro_batch = 8
    train_sample_start = 0
    train_samples = 256
    lr = 2e-6
    scheduler_type = "constant"
    warmup_ratio = 0.1
    epsilon = 0.2
    beta = 1e-4  # for kl loss
    eval_every_train_steps = 10
    eval_samples = 1000
    eval_batch_size = 256
    rollout_max_new_tokens = 1024
    rollout_temperature = 1.0
    eval_temperature = 0.6
    eval_max_new_tokens = 1024

每次评测1000耗时1:30，一共6次评测耗时9分钟，扣除后训练耗时约6:37
{'train/loss': 0.020731416635200617, 'train/policy_loss': 0.020730556454509497, 'train/kl_loss': 0.008598215878009796, 'train/reward': 0.6627014130353928, 'train/reward_std': 0.11667962558567524, 'train/accuracy': 0.59375, 'train/format_accuracy': 0.59375, 'train/advantage_abs': 0.47604679642245173, 'train/entropy': 0.4820147193968296, 'train/response_tokens': 674.0, 'time/rollout_sec': 5.649571403046139, 'time/rollout_wait_sec': 6.170477718114853e-06, 'time/precompute_sec': 0.8390890695154667, 'time/compute_sec': 2.3756459581200033, 'time/total_sec': 3.214741198113188, 'train/prompts': 256, 'train/samples': 2016, 'train/train_step': 63, 'train/lr': 2e-06}
Training GRPO: 100%|█████████████████████████████████████████████| 2048/2048 [15:37<00:00, 13.50it/s]{'train/loss': -0.022125677564559965, 'train/policy_loss': -0.022126823663711548, 'train/kl_loss': 0.011493321508169174, 'train/reward': 0.5610839799046516, 'train/reward_std': 0.14020730275660753, 'train/accuracy': 0.5, 'train/format_accuracy': 0.5, 'train/advantage_abs': 0.5017369966953993, 'train/entropy': 0.49252266995608807, 'train/response_tokens': 714.75, 'time/rollout_sec': 5.649571403046139, 'time/rollout_wait_sec': 6.170477718114853e-06, 'time/precompute_sec': 0.8390890695154667, 'time/compute_sec': 2.4134757560677826, 'time/total_sec': 3.2525709960609674, 'train/prompts': 256, 'train/samples': 2048, 'train/train_step': 64, 'train/lr': 2e-06}

vllm：
Training GRPO: 100%|█████████████████████████████████████████████| 2048/2048 [24:38<00:00,  9.55it/s]{'train/loss': -0.03291419311764088, 'train/policy_loss': -0.03291554329916835, 'train/kl_loss': 0.013470489531755447, 'train/reward': 0.592010498046875, 'train/reward_std': 0.20887851202860475, 'train/accuracy': 0.53125, 'train/format_accuracy': 0.53125, 'train/advantage_abs': 0.49864673614501953, 'train/entropy': 0.48514987621456385, 'train/response_tokens': 716.25, 'time/rollout_sec': 5.489136448130012, 'time/rollout_wait_sec': 3.119930624961853e-06, 'time/precompute_sec': 1.2738394988700747, 'time/compute_sec': 3.3577221389859915, 'time/total_sec': 4.631564757786691, 'train/prompts': 256, 'train/samples': 2048, 'train/train_step': 64, 'train/lr': 2e-06}
之前的超参会爆显存，所以把micro batch size都改成1了，但影响应该不是特别大，因为瓶颈主要在计算上，多出来的时间都是重启vllm带来的。

transformers：
eval batch size 得从 256 降到 64 才不会 OOM，单次耗时 09:05，由于时间太长，完整测试的时候就把评测关了，单纯训练耗时
Training GRPO: 100%|█████████████████████████████████████████████| 2048/2048 [23:18<00:00,  9.42it/s]{'train/loss': -0.03356845119223806, 'train/policy_loss': -0.03356928797438741, 'train/kl_loss': 0.008410260081291199, 'train/reward': 0.5272644080687314, 'train/reward_std': 0.2727518726605922, 'train/accuracy': 0.46875, 'train/format_accuracy': 0.5, 'train/advantage_abs': 0.45552800269797444, 'train/entropy': 0.4985915580764413, 'train/response_tokens': 730.15625, 'time/rollout_sec': 17.32243198528886, 'time/rollout_wait_sec': 1.778826117515564e-07, 'time/precompute_sec': 1.2593635600060225, 'time/compute_sec': 3.273150883615017, 'time/total_sec': 4.532514621503651, 'train/prompts': 256, 'train/samples': 2048, 'train/train_step': 64, 'train/lr': 2e-06}


## Task 2：RLVR的GRPO实验

1_grpo_gs8
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/9yhzf6bk
1024 tokens:
Evaluated: 1000
Correct: 582 (0.5820)
Format Correct: 616 (0.6160)
Avg output len (tokens): all=600.7, correct=330.1, format_ok_wrong=453.4, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=388 (0.8758), format_correct=403 (0.9097), avg_output_len=317.7, avg_output_len_correct=239.8, avg_output_len_format_ok_wrong=448.5, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=160 (0.3765), format_correct=176 (0.4141), avg_output_len=803.8, avg_output_len_correct=494.2, avg_output_len_format_ok_wrong=473.0, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=34 (0.2576), format_correct=37 (0.2803), avg_output_len=896.8, avg_output_len_correct=587.4, avg_output_len_format_ok_wrong=373.3, avg_output_len_format_wrong=1024.0
更正后：
Evaluated: 1000
Correct: 482 (0.4820)
Format Correct: 605 (0.6050)
Avg output len (tokens): all=605.3, correct=302.2, format_ok_wrong=448.3, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=377 (0.8510), format_correct=399 (0.9007), avg_output_len=323.6, avg_output_len_correct=243.6, avg_output_len_format_ok_wrong=293.9, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=98 (0.2306), format_correct=172 (0.4047), avg_output_len=806.5, avg_output_len_correct=502.1, avg_output_len_format_ok_wrong=465.9, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=7 (0.0530), format_correct=34 (0.2576), avg_output_len=902.8, avg_output_len_correct=660.6, avg_output_len_format_ok_wrong=525.8, avg_output_len_format_wrong=1024.0
烂完了啊，所以之前都是靠作弊来的。
2048 tokens:
Evaluated: 1000
Correct: 624 (0.6240)
Format Correct: 694 (0.6940)
Avg output len (tokens): all=948.7, correct=429.6, format_ok_wrong=770.4, format_wrong=2048.0
By nums count:
  nums=3: total=443, correct=407 (0.9187), format_correct=426 (0.9616), avg_output_len=378.2, avg_output_len_correct=296.0, avg_output_len_format_ok_wrong=643.7, avg_output_len_format_wrong=2048.0
  nums=4: total=425, correct=179 (0.4212), format_correct=221 (0.5200), avg_output_len=1340.3, avg_output_len_correct=652.6, avg_output_len_format_ok_wrong=833.5, avg_output_len_format_wrong=2048.0
  nums=5: total=132, correct=38 (0.2879), format_correct=47 (0.3561), avg_output_len=1602.7, avg_output_len_correct=810.2, avg_output_len_format_ok_wrong=743.3, avg_output_len_format_wrong=2048.0


依然在 2048 tokens 上训练，抽取 5 个数字的问题，有较大提升，但对 1024 tokens 上无帮助。
grpo_trainbs64_minibs64_gs16_nums5_base1
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/u9d1nev2
1024 tokens:
Evaluated: 1000
Correct: 577 (0.5770)
Format Correct: 615 (0.6150)
Avg output len (tokens): all=616.6, correct=350.9, format_ok_wrong=523.5, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=389 (0.8781), format_correct=409 (0.9233), avg_output_len=338.6, avg_output_len_correct=271.1, avg_output_len_format_ok_wrong=485.4, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=153 (0.3600), format_correct=169 (0.3976), avg_output_len=817.2, avg_output_len_correct=495.0, avg_output_len_format_ok_wrong=589.0, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=35 (0.2652), format_correct=37 (0.2803), avg_output_len=904.0, avg_output_len_correct=608.2, avg_output_len_format_ok_wrong=380.0, avg_output_len_format_wrong=1024.0
2048 tokens:
Evaluated: 1000
Correct: 659 (0.6590)
Format Correct: 721 (0.7210)
Avg output len (tokens): all=951.6, correct=494.8, format_ok_wrong=872.8, format_wrong=2048.0
By nums count:
  nums=3: total=443, correct=395 (0.8916), format_correct=421 (0.9503), avg_output_len=400.7, avg_output_len_correct=295.6, avg_output_len_format_ok_wrong=603.5, avg_output_len_format_wrong=2048.0
  nums=4: total=425, correct=206 (0.4847), format_correct=238 (0.5600), avg_output_len=1339.8, avg_output_len_correct=747.6, avg_output_len_format_ok_wrong=1013.1, avg_output_len_format_wrong=2048.0
  nums=5: total=132, correct=58 (0.4394), format_correct=62 (0.4697), avg_output_len=1550.5, avg_output_len_correct=953.6, avg_output_len_format_ok_wrong=1500.8, avg_output_len_format_wrong=2048.0

继续在 2048 tokens 上训练无法泛化到 1024 tokens 了，直接在 1024 tokens 上训练，降低学习率，放开 KL Loss。


在 1024 tokens 上训练 4 和 5：
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/ligklf8d
Evaluated: 1000
Correct: 605 (0.6050)
Format Correct: 626 (0.6260)
Avg output len (tokens): all=622.4, correct=374.7, format_ok_wrong=607.7, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=388 (0.8758), format_correct=402 (0.9074), avg_output_len=377.3, avg_output_len_correct=301.5, avg_output_len_format_ok_wrong=583.4, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=178 (0.4188), format_correct=185 (0.4353), avg_output_len=795.0, avg_output_len_correct=491.8, avg_output_len_format_ok_wrong=656.4, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=39 (0.2955), format_correct=39 (0.2955), avg_output_len=889.1, avg_output_len_correct=567.5, avg_output_len_format_ok_wrong=0.0, avg_output_len_format_wrong=1024.0

8B:
Loaded 2000 records from data/benchmark/8b_4096_long.jsonl

=== 2D Metrics Table (trunc_len x num_count) ===

[trunc_len=512]
num_count       total   correct accuracy        format_correct  format_accuracy avg_output_len
3       843     283     0.3357  283     0.3357  464.8
4       864     49      0.0567  49      0.0567  507.9
5       293     2       0.0068  2       0.0068  511.7

[trunc_len=1024]
num_count       total   correct accuracy        format_correct  format_accuracy avg_output_len
3       843     653     0.7746  653     0.7746  678.0
4       864     240     0.2778  240     0.2778  936.3
5       293     26      0.0887  26      0.0887  1004.4

[trunc_len=2048]
num_count       total   correct accuracy        format_correct  format_accuracy avg_output_len
3       843     816     0.9680  817     0.9692  765.6
4       864     454     0.5255  460     0.5324  1522.9
5       293     133     0.4539  133     0.4539  1763.7

[trunc_len=4096]
num_count       total   correct accuracy        format_correct  format_accuracy avg_output_len
3       843     837     0.9929  838     0.9941  799.6
4       864     550     0.6366  561     0.6493  2332.4
5       293     227     0.7747  232     0.7918  2450.6

Saved 2D stats json to data/benchmark/8b_4096_long_2d_stats.json

重开：

grpo_trainbs128_minibs64_gs16_2048：在2048上下文上训
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/77775kat?nw=nwuserflwfdd
1024 tokens：
Evaluated: 1000
Correct: 500 (0.5000)
Format Correct: 560 (0.5600)
Avg output len (tokens): all=688.1, correct=409.9, format_ok_wrong=542.9, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=372 (0.8397), format_correct=386 (0.8713), avg_output_len=438.7, avg_output_len_correct=349.2, avg_output_len_format_ok_wrong=435.4, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=118 (0.2776), format_correct=153 (0.3600), avg_output_len=863.0, avg_output_len_correct=581.4, avg_output_len_format_ok_wrong=561.4, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=10 (0.0758), format_correct=21 (0.1591), avg_output_len=961.6, avg_output_len_correct=643.7, avg_output_len_format_ok_wrong=620.6, avg_output_len_format_wrong=1024.0
2048 tokens：
Evaluated: 1000
Correct: 591 (0.5910)
Format Correct: 687 (0.6870)
Avg output len (tokens): all=1061.6, correct=554.8, format_ok_wrong=965.7, format_wrong=2047.7
By nums count:
  nums=3: total=443, correct=402 (0.9074), format_correct=424 (0.9571), avg_output_len=497.0, avg_output_len_correct=413.9, avg_output_len_format_ok_wrong=674.7, avg_output_len_format_wrong=2048.0
  nums=4: total=425, correct=163 (0.3835), format_correct=212 (0.4988), avg_output_len=1451.0, avg_output_len_correct=802.9, avg_output_len_format_ok_wrong=1013.8, avg_output_len_format_wrong=2047.6
  nums=5: total=132, correct=26 (0.1970), format_correct=51 (0.3864), avg_output_len=1702.4, avg_output_len_correct=1178.5, avg_output_len_format_ok_wrong=1127.7, avg_output_len_format_wrong=2048.0


grpo_trainbs128_minibs64_gs16_nodensity：没有截断时的density奖励
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/roy3scjy?nw=nwuserflwfdd
1024 tokens：
Evaluated: 1000
Correct: 510 (0.5100)
Format Correct: 563 (0.5630)
Avg output len (tokens): all=702.1, correct=440.7, format_ok_wrong=563.6, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=388 (0.8758), format_correct=399 (0.9007), avg_output_len=453.3, avg_output_len_correct=386.5, avg_output_len_format_ok_wrong=528.2, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=112 (0.2635), format_correct=139 (0.3271), avg_output_len=881.7, avg_output_len_correct=603.1, avg_output_len_format_ok_wrong=529.9, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=10 (0.0758), format_correct=25 (0.1894), avg_output_len=958.8, avg_output_len_correct=723.5, avg_output_len_format_ok_wrong=650.4, avg_output_len_format_wrong=1024.0
2048 tokens：
Evaluated: 1000
Correct: 613 (0.6130)
Format Correct: 691 (0.6910)
Avg output len (tokens): all=1062.7, correct=595.3, format_ok_wrong=861.5, format_wrong=2040.8
By nums count:
  nums=3: total=443, correct=407 (0.9187), format_correct=424 (0.9571), avg_output_len=501.3, avg_output_len_correct=424.9, avg_output_len_format_ok_wrong=733.9, avg_output_len_format_wrong=1930.3
  nums=4: total=425, correct=162 (0.3812), format_correct=210 (0.4941), avg_output_len=1457.9, avg_output_len_correct=842.9, avg_output_len_format_ok_wrong=890.4, avg_output_len_format_wrong=2048.0
  nums=5: total=132, correct=44 (0.3333), format_correct=57 (0.4318), avg_output_len=1674.7, avg_output_len_correct=1260.9, avg_output_len_format_ok_wrong=921.8, avg_output_len_format_wrong=2048.0

用 2048 训基本上没有什么收益？

grpo_trainbs128_minibs64_gs16：1024 tokens baseline
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/5aybli2v?nw=nwuserflwfdd
Evaluated: 1000
Correct: 497 (0.4970)
Format Correct: 552 (0.5520)
Avg output len (tokens): all=697.3, correct=419.8, format_ok_wrong=542.7, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=372 (0.8397), format_correct=388 (0.8758), avg_output_len=450.6, avg_output_len_correct=361.1, avg_output_len_format_ok_wrong=560.1, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=114 (0.2682), format_correct=143 (0.3365), avg_output_len=869.3, avg_output_len_correct=579.2, avg_output_len_format_ok_wrong=505.8, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=11 (0.0833), format_correct=21 (0.1591), avg_output_len=971.0, avg_output_len_correct=753.6, avg_output_len_format_ok_wrong=622.1, avg_output_len_format_wrong=1024.0


grpo_trainbs64_minibs32_gs8_fast：group size从16降到8，同时mini batch也从64降到32，保持训练步数一致
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/nk37655f?nw=nwuserflwfdd
Evaluated: 1000
Correct: 520 (0.5200)
Format Correct: 588 (0.5880)
Avg output len (tokens): all=688.7, correct=444.2, format_ok_wrong=534.0, format_wrong=1022.8
By nums count:
  nums=3: total=443, correct=367 (0.8284), format_correct=391 (0.8826), avg_output_len=451.7, avg_output_len_correct=371.7, avg_output_len_format_ok_wrong=435.2, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=135 (0.3176), format_correct=166 (0.3906), avg_output_len=851.2, avg_output_len_correct=595.9, avg_output_len_format_ok_wrong=535.1, avg_output_len_format_wrong=1022.1
  nums=5: total=132, correct=18 (0.1364), format_correct=31 (0.2348), avg_output_len=960.9, avg_output_len_correct=785.1, avg_output_len_format_ok_wrong=713.9, avg_output_len_format_wrong=1024.0

grpo_trainbs64_minibs32_gs8_fast_nokl：去掉了 kl loss
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/6lxzhksr?nw=nwuserflwfdd
Evaluated: 1000
Correct: 481 (0.4810)
Format Correct: 562 (0.5620)
Avg output len (tokens): all=682.7, correct=408.5, format_ok_wrong=466.0, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=374 (0.8442), format_correct=396 (0.8939), avg_output_len=425.6, avg_output_len_correct=351.9, avg_output_len_format_ok_wrong=399.0, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=98 (0.2306), format_correct=142 (0.3341), avg_output_len=868.0, avg_output_len_correct=590.7, avg_output_len_format_ok_wrong=482.3, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=9 (0.0682), format_correct=24 (0.1818), avg_output_len=949.2, avg_output_len_correct=773.4, avg_output_len_format_ok_wrong=516.3, avg_output_len_format_wrong=1024.0
kl 直接炸了，最后直接inf了，但居然这样评测上居然甚至还行

grpo_trainbs64_minibs32_gs8_fast_45：只用45个数字的难题
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/g4elj4n5?nw=nwuserflwfdd
Evaluated: 1000
Correct: 473 (0.4730)
Format Correct: 525 (0.5250)
Avg output len (tokens): all=740.0, correct=469.9, format_ok_wrong=602.0, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=371 (0.8375), format_correct=385 (0.8691), avg_output_len=504.8, avg_output_len_correct=421.1, avg_output_len_format_ok_wrong=570.9, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=94 (0.2212), format_correct=127 (0.2988), avg_output_len=904.2, avg_output_len_correct=632.9, avg_output_len_format_ok_wrong=594.9, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=8 (0.0606), format_correct=13 (0.0985), avg_output_len=1000.7, avg_output_len_correct=819.1, avg_output_len_format_ok_wrong=736.2, avg_output_len_format_wrong=1024.0

grpo_trainbs64_minibs32_gs8_fast_nodensity_1024samples：没有截断数学密度奖励，样本数由256提升到1024
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/pjypvdd9?nw=nwuserflwfdd
Evaluated: 1000
Correct: 534 (0.5340)
Format Correct: 631 (0.6310)
Avg output len (tokens): all=613.7, correct=348.4, format_ok_wrong=513.4, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=380 (0.8578), format_correct=412 (0.9300), avg_output_len=333.2, avg_output_len_correct=272.6, avg_output_len_format_ok_wrong=383.2, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=135 (0.3176), format_correct=185 (0.4353), avg_output_len=805.7, avg_output_len_correct=516.9, avg_output_len_format_ok_wrong=538.1, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=19 (0.1439), format_correct=34 (0.2576), avg_output_len=936.7, avg_output_len_correct=666.5, avg_output_len_format_ok_wrong=708.9, avg_output_len_format_wrong=1024.0

看eval在140/256达到峰值，后面降低了又回升，输出长度确实一直在降，但用了4倍时间似乎没有带来太大提升？理论上只要长度一直降就不断会有能做对的到上下文限制内转化为正确率。


grpo_trainbs64_minibs32_gs8_fast_lr5en6：把lr从2e-6加到5e-6
Evaluated: 1000
Correct: 486 (0.4860)
Format Correct: 561 (0.5610)
Avg output len (tokens): all=669.6, correct=380.3, format_ok_wrong=470.1, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=365 (0.8239), format_correct=385 (0.8691), avg_output_len=407.7, avg_output_len_correct=311.4, avg_output_len_format_ok_wrong=377.5, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=111 (0.2612), format_correct=156 (0.3671), avg_output_len=849.3, avg_output_len_correct=570.7, avg_output_len_format_ok_wrong=492.2, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=10 (0.0758), format_correct=20 (0.1515), avg_output_len=970.3, avg_output_len_correct=783.7, avg_output_len_format_ok_wrong=556.0, avg_output_len_format_wrong=1024.0

grpo_trainbs64_minibs32_gs8_fast_kl1en5：把kl loss从1e-4降到1e-5
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/ggp4nkjc?nw=nwuserflwfdd
Evaluated: 1000
Correct: 524 (0.5240)
Format Correct: 575 (0.5750)
Avg output len (tokens): all=694.4, correct=444.2, format_ok_wrong=518.1, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=380 (0.8578), format_correct=396 (0.8939), avg_output_len=452.1, avg_output_len_correct=379.5, avg_output_len_format_ok_wrong=496.4, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=126 (0.2965), format_correct=152 (0.3576), avg_output_len=863.1, avg_output_len_correct=590.4, avg_output_len_format_ok_wrong=494.8, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=18 (0.1364), format_correct=27 (0.2045), avg_output_len=964.1, avg_output_len_correct=784.7, avg_output_len_format_ok_wrong=623.8, avg_output_len_format_wrong=1024.0


grpo_trainbs64_minibs32_gs8_fast_nolen：
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/xq1svn7l?nw=nwuserflwfdd
Evaluated: 1000
Correct: 492 (0.4920)
Format Correct: 544 (0.5440)
Avg output len (tokens): all=725.8, correct=460.7, format_ok_wrong=618.6, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=367 (0.8284), format_correct=382 (0.8623), avg_output_len=494.4, avg_output_len_correct=405.3, avg_output_len_format_ok_wrong=519.7, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=112 (0.2635), format_correct=139 (0.3271), avg_output_len=887.9, avg_output_len_correct=605.3, avg_output_len_format_ok_wrong=618.8, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=13 (0.0985), format_correct=23 (0.1742), avg_output_len=980.5, avg_output_len_correct=780.5, avg_output_len_format_ok_wrong=766.5, avg_output_len_format_wrong=1024.0

grpo_trainbs64_minibs32_gs8_fast_temp0p1：温度由0.6改为0.1
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/6j7ccome/overview?nw=nwuserflwfdd
Evaluated: 1000
Correct: 494 (0.4940)
Format Correct: 555 (0.5550)
Avg output len (tokens): all=701.0, correct=434.4, format_ok_wrong=503.9, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=367 (0.8284), format_correct=384 (0.8668), avg_output_len=461.4, avg_output_len_correct=369.9, avg_output_len_format_ok_wrong=486.1, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=114 (0.2682), format_correct=146 (0.3435), avg_output_len=870.8, avg_output_len_correct=598.8, avg_output_len_format_ok_wrong=503.8, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=13 (0.0985), format_correct=25 (0.1894), avg_output_len=958.6, avg_output_len_correct=816.7, avg_output_len_format_ok_wrong=529.4, avg_output_len_format_wrong=1024.0


grpo_trainbs64_minibs32_gs8_fast_temp1：
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/dmaj4jj7/overview?nw=nwuserflwfdd
Evaluated: 1000
Correct: 536 (0.5360)
Format Correct: 579 (0.5790)
Avg output len (tokens): all=693.8, correct=447.9, format_ok_wrong=525.5, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=388 (0.8758), format_correct=397 (0.8962), avg_output_len=455.9, avg_output_len_correct=391.3, avg_output_len_format_ok_wrong=336.3, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=127 (0.2988), format_correct=153 (0.3600), avg_output_len=857.3, avg_output_len_correct=561.6, avg_output_len_format_ok_wrong=557.2, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=21 (0.1591), format_correct=29 (0.2197), avg_output_len=965.8, avg_output_len_correct=806.4, avg_output_len_format_ok_wrong=635.1, avg_output_len_format_wrong=1024.0

grpo_trainbs64_minibs32_gs8_fast_temp1p2
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/uzyvdne3?nw=nwuserflwfdd
Evaluated: 1000
Correct: 499 (0.4990)
Format Correct: 534 (0.5340)
Avg output len (tokens): all=731.2, correct=464.5, format_ok_wrong=636.7, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=376 (0.8488), format_correct=383 (0.8646), avg_output_len=489.3, avg_output_len_correct=404.6, avg_output_len_format_ok_wrong=456.4, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=111 (0.2612), format_correct=131 (0.3082), avg_output_len=901.4, avg_output_len_correct=632.7, avg_output_len_format_ok_wrong=589.9, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=12 (0.0909), format_correct=20 (0.1515), avg_output_len=995.3, avg_output_len_correct=783.3, avg_output_len_format_ok_wrong=911.6, avg_output_len_format_wrong=1024.0

grpo_trainbs32_minibs16_gs4_veryfast
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/bvduy8vt?nw=nwuserflwfdd
Evaluated: 1000
Correct: 520 (0.5200)
Format Correct: 588 (0.5880)
Avg output len (tokens): all=658.1, correct=387.7, format_ok_wrong=508.8, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=368 (0.8307), format_correct=394 (0.8894), avg_output_len=415.3, avg_output_len_correct=335.6, avg_output_len_format_ok_wrong=395.8, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=134 (0.3153), format_correct=163 (0.3835), avg_output_len=821.2, avg_output_len_correct=487.3, avg_output_len_format_ok_wrong=532.1, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=18 (0.1364), format_correct=31 (0.2348), avg_output_len=947.6, avg_output_len_correct=710.3, avg_output_len_format_ok_wrong=682.8, avg_output_len_format_wrong=1024.0

grpo_trainbs64_minibs32_gs8_fast_nonumsrwd：没有难度分级奖励
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/r8jiq4ll?nw=nwuserflwfdd
Evaluated: 1000
Correct: 521 (0.5210)
Format Correct: 582 (0.5820)
Avg output len (tokens): all=674.9, correct=415.6, format_ok_wrong=506.3, format_wrong=1022.7
By nums count:
  nums=3: total=443, correct=376 (0.8488), format_correct=387 (0.8736), avg_output_len=442.6, avg_output_len_correct=355.2, avg_output_len_format_ok_wrong=469.2, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=133 (0.3129), format_correct=167 (0.3929), avg_output_len=833.3, avg_output_len_correct=556.8, avg_output_len_format_ok_wrong=484.0, avg_output_len_format_wrong=1021.8
  nums=5: total=132, correct=12 (0.0909), format_correct=28 (0.2121), avg_output_len=944.5, avg_output_len_correct=742.4, avg_output_len_format_ok_wrong=579.2, avg_output_len_format_wrong=1024.0


grpo_trainbs32_minibs16_gs8_veryfast：基于 grpo_trainbs64_minibs32_gs8_fast_nodensity_1024samples 用45搞512条
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/s921zfzf?nw=nwuserflwfdd
Evaluated: 1000
Correct: 540 (0.5400)
Format Correct: 615 (0.6150)
Avg output len (tokens): all=636.6, correct=379.8, format_ok_wrong=503.3, format_wrong=1022.8
By nums count:
  nums=3: total=443, correct=374 (0.8442), format_correct=405 (0.9142), avg_output_len=366.3, avg_output_len_correct=294.9, avg_output_len_format_ok_wrong=422.5, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=134 (0.3153), format_correct=165 (0.3882), avg_output_len=827.3, avg_output_len_correct=518.1, avg_output_len_format_ok_wrong=513.5, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=32 (0.2424), format_correct=45 (0.3409), avg_output_len=929.8, avg_output_len_correct=793.2, avg_output_len_format_ok_wrong=672.0, avg_output_len_format_wrong=1018.6


grpo_trainbs32_minibs32_gs4_1024to2048：基于 grpo_trainbs32_minibs16_gs8_veryfast
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/nhbpylg4?nw=nwuserflwfdd
Evaluated: 1000
Correct: 546 (0.5460)
Format Correct: 662 (0.6620)
Avg output len (tokens): all=547.7, correct=279.0, format_ok_wrong=425.5, format_wrong=1023.6
By nums count:
  nums=3: total=443, correct=365 (0.8239), format_correct=399 (0.9007), avg_output_len=287.6, avg_output_len_correct=197.5, avg_output_len_format_ok_wrong=302.4, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=126 (0.2965), format_correct=190 (0.4471), avg_output_len=748.2, avg_output_len_correct=387.3, avg_output_len_format_ok_wrong=448.3, avg_output_len_format_wrong=1023.4
  nums=5: total=132, correct=55 (0.4167), format_correct=73 (0.5530), avg_output_len=774.7, avg_output_len_correct=572.1, avg_output_len_format_ok_wrong=576.9, avg_output_len_format_wrong=1024.0
============================================================
Evaluated: 500
Correct: 253 (0.5060)
Format Correct: 310 (0.6200)
Avg output len (tokens): all=602.8, correct=319.2, format_ok_wrong=462.5, format_wrong=1022.6
By nums count:
  nums=3: total=187, correct=147 (0.7861), format_correct=168 (0.8984), avg_output_len=306.4, avg_output_len_correct=202.8, avg_output_len_format_ok_wrong=395.0, avg_output_len_format_wrong=1010.0
  nums=4: total=213, correct=66 (0.3099), format_correct=91 (0.4272), avg_output_len=775.4, avg_output_len_correct=426.6, avg_output_len_format_ok_wrong=483.0, avg_output_len_format_wrong=1024.0
  nums=5: total=100, correct=40 (0.4000), format_correct=51 (0.5100), avg_output_len=789.4, avg_output_len_correct=569.5, avg_output_len_format_ok_wrong=544.5, avg_output_len_format_wrong=1024.0


grpo_trainbs32_minibs16_gs4_45_512to3072：基于 base grpo_trainbs32_minibs32_gs4_1024to2048
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/3bn088ns?nw=nwuserflwfdd
Evaluated: 1000
Correct: 550 (0.5500)
Format Correct: 621 (0.6210)
Avg output len (tokens): all=589.6, correct=306.1, format_ok_wrong=466.2, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=375 (0.8465), format_correct=400 (0.9029), avg_output_len=323.3, avg_output_len_correct=240.7, avg_output_len_format_ok_wrong=356.5, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=134 (0.3153), format_correct=165 (0.3882), avg_output_len=787.5, avg_output_len_correct=394.2, avg_output_len_format_ok_wrong=504.3, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=41 (0.3106), format_correct=56 (0.4242), avg_output_len=845.9, avg_output_len_correct=616.4, avg_output_len_format_ok_wrong=570.4, avg_output_len_format_wrong=1024.0


grpo_trainbs32_minibs16_gs4_45_512to1024：基于 grpo_trainbs32_minibs16_gs8_veryfast
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/1vk2dmdd?nw=nwuserflwfdd
Evaluated: 1000
Correct: 545 (0.5450)
Format Correct: 623 (0.6230)
Avg output len (tokens): all=616.1, correct=347.8, format_ok_wrong=519.3, format_wrong=1024.0
By nums count:
  nums=3: total=443, correct=362 (0.8172), format_correct=395 (0.8916), avg_output_len=352.1, avg_output_len_correct=260.1, avg_output_len_format_ok_wrong=384.7, avg_output_len_format_wrong=1024.0
  nums=4: total=425, correct=142 (0.3341), format_correct=173 (0.4071), avg_output_len=804.1, avg_output_len_correct=469.0, avg_output_len_format_ok_wrong=551.6, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=41 (0.3106), format_correct=55 (0.4167), avg_output_len=896.9, avg_output_len_correct=703.1, avg_output_len_format_ok_wrong=765.1, avg_output_len_format_wrong=1024.0


grpo_trainbs32_minibs32_gs4_45_1024to2048_lenrwd0p2：基于 data/ckpt/grpo_trainbs32_minibs16_gs4_45_512to1024
https://wandb.ai/flwfdd-flwfdd/cs60004-lab3-rl/runs/5axpam0l/overview?nw=nwuserflwfdd
Evaluated: 1000
Correct: 544 (0.5440)
Format Correct: 633 (0.6330)
Avg output len (tokens): all=595.7, correct=325.5, format_ok_wrong=482.2, format_wrong=1023.8
By nums count:
  nums=3: total=443, correct=364 (0.8217), format_correct=398 (0.8984), avg_output_len=329.0, avg_output_len_correct=239.7, avg_output_len_format_ok_wrong=367.4, avg_output_len_format_wrong=1022.7
  nums=4: total=425, correct=147 (0.3459), format_correct=187 (0.4400), avg_output_len=785.0, avg_output_len_correct=465.0, avg_output_len_format_ok_wrong=538.5, avg_output_len_format_wrong=1024.0
  nums=5: total=132, correct=33 (0.2500), format_correct=48 (0.3636), avg_output_len=881.7, avg_output_len_correct=651.1, avg_output_len_format_ok_wrong=592.1, avg_output_len_format_wrong=1024.0

grpo_trainbs64_minibs64_gs8_test 基于 grpo_trainbs32_minibs32_gs4_1024to2048 在 raw_test_repeat.jsonl 上训了 400 步达到最佳 此时 3208 prompts
Evaluated: 500
Correct: 254 (0.5080)
Format Correct: 286 (0.5720)
Avg output len (tokens): all=622.0, correct=302.0, format_ok_wrong=481.4, format_wrong=1022.8
By nums count:
  nums=3: total=187, correct=159 (0.8503), format_correct=165 (0.8824), avg_output_len=292.1, avg_output_len_correct=188.4, avg_output_len_format_ok_wrong=357.3, avg_output_len_format_wrong=1024.0
  nums=4: total=213, correct=61 (0.2864), format_correct=80 (0.3756), avg_output_len=793.0, avg_output_len_correct=405.5, avg_output_len_format_ok_wrong=434.2, avg_output_len_format_wrong=1022.1
  nums=5: total=100, correct=34 (0.3400), format_correct=41 (0.4100), avg_output_len=874.5, avg_output_len_correct=647.8, avg_output_len_format_ok_wrong=715.9, avg_output_len_format_wrong=1024.0


dpo_0p6b_test_rejection_0p999nll_ep10 基于 grpo_trainbs64_minibs64_gs8_test/best 用 "data/dpo/0p6b_test_rejection_repeat10.jsonl" 训了 202 步 此时 88 prompts
Evaluated: 500
Correct: 278 (0.5560)
Format Correct: 309 (0.6180)
Avg output len (tokens): all=584.0, correct=289.9, format_ok_wrong=511.1, format_wrong=1023.9
By nums count:
  nums=3: total=187, correct=167 (0.8930), format_correct=173 (0.9251), avg_output_len=243.5, avg_output_len_correct=173.2, avg_output_len_format_ok_wrong=378.5, avg_output_len_format_wrong=1024.0
  nums=4: total=213, correct=76 (0.3568), format_correct=92 (0.4319), avg_output_len=776.6, avg_output_len_correct=430.8, avg_output_len_format_ok_wrong=548.4, avg_output_len_format_wrong=1024.0
  nums=5: total=100, correct=35 (0.3500), format_correct=44 (0.4400), avg_output_len=810.4, avg_output_len_correct=540.7, avg_output_len_format_ok_wrong=533.1, avg_output_len_format_wrong=1023.6



grpo_trainbs64_minibs16_gs8_test_basedpoep10/best：基于dpo_0p6b_test_rejection_0p999nll_ep10 在 data/splits/raw_test_low_repeat_accuracy_repeat.jsonl 上训了 40 步达到最佳
Evaluated: 500
Correct: 293 (0.5860)
Format Correct: 329 (0.6580)
Avg output len (tokens): all=558.8, correct=297.0, format_ok_wrong=479.1, format_wrong=1024.0
By nums count:
  nums=3: total=187, correct=171 (0.9144), format_correct=179 (0.9572), avg_output_len=225.1, avg_output_len_correct=174.5, avg_output_len_format_ok_wrong=505.8, avg_output_len_format_wrong=1024.0
  nums=4: total=213, correct=80 (0.3756), format_correct=101 (0.4742), avg_output_len=740.1, avg_output_len_correct=419.4, avg_output_len_format_ok_wrong=447.5, avg_output_len_format_wrong=1024.0
  nums=5: total=100, correct=42 (0.4200), format_correct=49 (0.4900), avg_output_len=796.6, avg_output_len_correct=562.7, avg_output_len_format_ok_wrong=543.1, avg_output_len_format_wrong=1024.0

# 让 MTP 与 Partial Prefix Cache 同时生效：一次 Qwen3.5/vLLM 实机修复记录

## 结果先说

在 Qwen3.5 hybrid 模型、AWQ W4A16、原生 MTP=4 的组合中，一个 6001-token 公共前缀在 stock vLLM 0.26.0 上只能复用 4896 tokens。修复后可以复用 5984 tokens：

| 实现 | 公共前缀 | 第二个请求总输入 | 命中 | 热请求耗时 |
|---|---:|---:|---:|---:|
| stock vLLM 0.26.0 | 6001 | 6012 | 4896（81.4%） | 0.659 s |
| patched vLLM 0.26.0 | 6001 | 6012 | 5984（99.5%） | 0.055 s |

剩余 16 tokens 的 replay 是 MTP/EAGLE 的正确性要求，不应该继续消除。

修复采用隔离 runtime overlay，没有修改系统 vLLM。针对性单测从 RED 变为 GREEN，整个 partial-prefix 测试文件 14/14 通过；同一模型进程中，6/6 组 64-token stable-hot 输出与 cold 逐 token 一致，另有 16/16 组压力用例通过。

## 场景与症状

目标模型是 Qwen3.5 hybrid 架构，同时包含 full attention 和 GatedDeltaNet/Mamba recurrent state。服务组合为：

```text
vLLM 0.26.0
AWQ W4A16
native MTP, num_speculative_tokens=4
enable_prefix_caching=true
mamba_cache_mode=align
prefix_match_unit=16
```

模型解析出的物理 Mamba cache block 是 816 tokens，prefix hash 粒度是 16 tokens。不开 MTP 时，6001-token 前缀可以命中到 6000；开启 MTP 后却只能命中到 4896：

```text
6000 = floor(6001 / 16) * 16
4896 = 6 * 816
```

这不是普通缓存没有开启。`prefix_cache_queries_total` 和 `prefix_cache_hits_total` 都在增长，但 hybrid coordinator 最终把候选命中向前收缩了一个完整物理块。

## 为什么会回退一个完整物理块

MTP 在 vLLM 中走 EAGLE-style speculative decoding 路径。Full Attention 找到一个共享前缀后，EAGLE 会保留一个 hash unit 用于 replay/proof：

```text
Full Attention 候选：6000
EAGLE replay 单元：  16
目标恢复位置：      5984
```

5984 本身是合法的 16-token hash 边界，但旧调度逻辑只保证在 816-token 物理网格或 prompt tail 物化 Mamba state。5984 没有对应 recurrent checkpoint，hybrid coordinator 必须让所有 cache group 在同一位置恢复，于是继续回退：

```text
5712 = 7 * 816
4896 = 5712 - 816
```

旧代码中的保守逻辑正是：

```python
last_cache_position = num_tokens // block_size * block_size
if use_eagle:
    last_cache_position -= block_size
```

它避免从不存在的 Mamba checkpoint 恢复，因此保证正确性，但代价是每个非对齐公共前缀额外重算最多一个大物理块。

## 第一个容易犯的错误：直接删除 back-off

最直观的改法是删除 `last_cache_position -= block_size`。这不安全。

缓存命中长度只是一项声明。要从位置 T 恢复，物理 cache 中必须确实存在“处理完 T 个 token 后”的 Full Attention KV、Mamba recurrent state，以及与 MTP replay 对齐的状态。如果只改变命中长度、不物化 checkpoint，服务可能继续运行，却使用语义错误的 recurrent state，形成静默输出污染。

正确修复不是“允许命中更多”，而是“先在真正会恢复的位置生成并持久化正确状态，再允许命中”。

## 修复思路

修复分两部分。

### 1. 调度器停在真正的恢复位置

当一个请求发现存在共享前缀 junction 时：

- 无 MTP 时继续按原有 block grid 处理。
- MTP/EAGLE + partial hit 时，把 junction floor 到 hash grid。
- prompt tail 在 EAGLE 下也向前移动一个 hash unit，因为这是消费者真正会恢复的位置。

核心逻辑：

```python
if tail_boundary and self.use_eagle:
    tail_boundary = max(tail_boundary - self.hash_block_size, 0)

junction = request.shared_prefix_boundary
if self.mamba_partial_cache_hit and self.use_eagle:
    junction_stop = junction // self.hash_block_size * self.hash_block_size
else:
    junction_stop = block_floor(junction)
```

这样 worker 会真实执行到 5984，并在这个 chunk end 产生正确的 Mamba recurrent state。

### 2. Mamba manager 注册这个 checkpoint

旧逻辑只接受 prompt 的最后一个 hash boundary。修复后还接受 EAGLE 会落到的 block-grid resume point：

```python
block_grid_resume_point = (
    self.use_eagle
    and num_tokens <= request.num_prompt_tokens
    and (num_tokens + hash_block_size) % self.block_size == 0
)
```

随后仍通过现有 `cache_partial_block` 和 copy-on-write 机制发布状态，不改变 block ownership 与生命周期协议。

## 为什么有时第二个请求和第三个请求表现不同

对于 `system prompt + suffix A/B`，第一个请求并不知道未来的共享 junction 在哪里。第二个不同 suffix 请求首次通过 Full Attention 发现 junction，并在重算经过该位置时物化缺失 checkpoint。第三个及后续请求才能稳定命中该精确位置。

因此实际序列是：

```text
请求 A：建立普通缓存
请求 B：发现 junction，重算并建立精确 Mamba checkpoint
请求 C+：稳定命中精确 checkpoint
```

这也是为什么正确性验证必须区分 cold、首次 hot 和 stable-hot。

## TDD 与实机验证

### RED

新增两个最小回归：

1. scheduler 应在 1984 hash boundary 停止，stock 返回 1536。
2. Mamba follower 应命中位置 7，stock 只能命中位置 4。

原始 v0.26.0：2/2 按预期失败。

### GREEN

应用两处最小修复后：

- 针对性测试：2/2 通过。
- `test_partial_prefix_cache_hits.py`：14/14 通过。
- MTP=4 E2E：5984/6012 命中，热请求 0.055 s。
- 同进程 cold/stable-hot：6/6，每组 64 output tokens 逐 token 一致。
- 稳定热压力：16/16，每组 32 output tokens 一致，覆盖 5200–5680 token 前缀。

跨独立进程比较不能作为严格 bitwise 门禁：长序列上的不同 CUDA 执行历史会让 greedy 在近似 logits 上分叉。最终采用同一模型进程、显式 reset prefix cache 的 cold/hot A/B 消除了这个干扰变量。

## 与上游工作的关系

PR #46384 提供了 partial prefix cache 的基础能力，并从 vLLM 0.26.0 开始发布。它解决“能否在物理块内部表达一个命中边界”，但不保证每种 speculative decoding 组合都在该边界物化所需 state。

本次 backport 参考了关闭未合入的 #53923。当前 main 上还有多个活跃且相互关联的 PR：#53802、#54076、#57128、#57180，以及 draft #53479。它们分别处理可达边界、异构 block geometry、EAGLE drop、prompt tail proof 和 retention cadence。

因此，本修复适合作为当前 v0.26.0 部署的受控 backport，不应原样作为针对 main 的新 PR；向 main 提交会与已有开放工作重复，并忽略新架构的 retention/checkpoint 设计。

## 部署与回滚

修复运行目录：

```text
/opt/tiger/workspace/vllm_mtp_partial_fix
```

启动：

```bash
cd /opt/tiger/workspace/vllm_mtp_partial_fix
nohup env GPU=0 PORT=8102 ./run_vllm_mtp_partial_fixed.sh \
  > patched_server.log 2>&1 &
```

它通过 `PYTHONPATH=runtime_overlay` 使用隔离包副本。系统的 `/usr/local/lib/python3.11/dist-packages/vllm` 未修改。回滚时停止补丁服务，使用普通 `vllm serve` 即可。

## 工程结论

MTP 与 partial prefix cache 的矛盾不是“draft 被拒绝后没有 cache”，而是消费者真正恢复的 EAGLE replay boundary 没有对应的 Mamba checkpoint。

修复的核心原则只有一句：

> 不要让命中长度跑在状态物化之前；先在真实恢复边界生成并注册 state，再让 coordinator 命中它。

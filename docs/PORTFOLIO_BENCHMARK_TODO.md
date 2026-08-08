# PortfolioBench-2h：核心架构 Benchmark TODO

状态：**TODO / 尚未产出正式对比结果**

本文冻结 Continuous Research Queue 的核心评测协议。目标不是比较谁生成
的论文数量最多，也不是用 GPU 利用率作为成功标准，而是回答：

> 在相同时间、模型 Token 和 GPU 上限下，哪个系统能够更快、更低成本地
> 产出更多科学有效且结论明确的研究结果？

ARC-Bench 继续作为单任务研究质量的辅助评测；本 Benchmark 专门评估本
系统的核心架构贡献：多 Idea 异步推进、渐进预算早退、动态 GPU 调度和
机器可执行科学门禁。

---

## 1. 主 Benchmark

名称：

```text
PortfolioBench-2h
```

冻结输入：

- 16 个预先定义、去重并版本化的研究 Idea；
- 每个 Idea 至少包含：
  - question；
  - hypothesis；
  - treatment；
  - control；
  - primary metric；
- 第一版统一使用同一个可重复的真实 Benchmark family，避免把领域适配
  差异误认为调度能力差异；
- 所有被比较系统必须收到相同的 Idea 集合。

冻结预算：

```text
Wall time              2 hours
Maximum GPUs           8
Maximum GPU per run    1
Token budget           same fixed cap for every system
Models                 same-model track by default
Dataset/model/seeds    identical
Benchmark adapter      identical
```

GPU 采用按需申请，不要求系统预先占有 8 张卡。记录实际申请、等待、执行
和释放时间。

正式实验至少运行 3 个独立重复；smoke test 可以先运行 1 个重复。

---

## 2. 主指标

唯一 headline 指标：

```text
VCO@2h = Valid Conclusive Outcomes completed within 2 hours
```

一个结果只有同时满足以下条件才计为一个 VCO：

```text
execution_passed = true
scientific_valid = true
hypothesis_supported ∈ {true, false}
```

因此：

- 科学有效的 positive 结果计 1；
- 科学有效的 negative 结果也计 1；
- inconclusive 不计；
- 运行失败不计；
- ResearchSpec 证据要求未满足不计；
- guardrail 失败不计；
- 科学上无效但被系统错误 accept 的结果不计。

负结果与正结果等价计数，避免鼓励 p-hacking 或只追求正结果。

---

## 3. 必要辅助指标

辅助指标只用于解释 VCO@2h，不能替代主指标。

### 3.1 速度

```text
TTFV = Time To First Valid Conclusive Outcome
```

若 2 小时内没有 VCO，记为 censored at 2h，并在汇总中单独报告。

### 3.2 成本

```text
Tokens / VCO
GPU-seconds / VCO
```

当 VCO 为 0 时，不计算比值；直接报告总消耗和 `VCO=0`，不能用任意大
常数代替。

### 3.3 可靠性

```text
False Accept Rate =
  scientifically invalid outcomes labeled accept
  / all outcomes labeled accept
```

分母为 0 时报告 `N/A`。

### 3.4 诊断指标

以下指标进入日志和附录，不作为 headline：

- generated / admitted / B0 passed / B1 passed / benchmarked / valid；
- time spent waiting for GPU；
- allocated-but-idle GPU-seconds；
- GPU release latency；
- peak concurrent Ideas；
- execution success rate；
- scientifically valid rate；
- inconclusive rate。

GPU utilization、Idea 数量、完成的 Pipeline 数量和正结果数量都不能单独
作为成功标准。

---

## 4. 核心对比组

| 组别 | 配置 | 目的 |
|---|---|---|
| `AutoResearchClaw-loop` | 冻结版 AutoResearchClaw；一个 Idea 结束后由外部 wrapper 启动下一个 | 与直接前身比较 |
| `RQ-Sequential` | 当前 Research Queue，但 `max_active_ideas=1`、`max_run_jobs=1` | 隔离异步 Portfolio 调度收益 |
| `RQ-NoEarlyExit` | 多 Idea 异步，但所有 admitted Idea 直接进入真实 Benchmark | 测量 B0/B1 早退价值 |
| `RQ-Full` | 多 Idea 异步 + B0/B1 早退 + 动态 GPU + deterministic scientific gate | 完整方法 |

最重要的因果比较：

```text
RQ-Sequential vs RQ-Full
```

二者必须使用相同模型、Prompt、Idea、Benchmark、科学门禁和总预算，主要
差异仅为调度与并发。

---

## 5. 公平性协议

### 5.1 Same-model track

主结果使用相同角色模型和相同调用上限。若某 Baseline 无法支持三档模型
路由，则记录实际映射，但不能给它额外 Token 或更长运行时间。

### 5.2 Native-best track

可作为补充结果。各框架使用其推荐配置，但必须保持相同：

- wall-clock budget；
- Token cap；
- GPU cap；
- Idea 输入；
- 数据、模型和 Benchmark；
- 最终判定标准。

### 5.3 冻结与追踪

每次正式运行必须记录：

- framework commit；
- config hash；
- model identifiers；
- prompt/config artifacts；
- Idea pool hash；
- benchmark config hash；
- dataset/model hashes；
- controller start/end time；
- 完整 Token 和 GPU ledger。

---

## 6. 辅助 Benchmark：ARC-Bench

PortfolioBench-2h 证明架构效率；ARC-Bench 用于确认这种效率不是通过牺牲
单个研究结果质量获得的。

第一阶段只选 6 个 ML topic，覆盖：

- 模型选择；
- 超参数优化；
- 鲁棒性；
- 校准；
- 不确定性；
- 异常检测。

比较：

```text
AutoResearchClaw rc_full
AI Scientist v2
RQ-Sequential
RQ-Full
```

ARC-Bench 的主要报告项：

- rubric-weighted research quality；
- execution success；
- experimental validity；
- claim faithfulness；
- artifact completeness。

ARC-Bench 分数是辅助质量指标，不替代 VCO@2h。

---

## 7. 科学门禁 Evidence-Pack Benchmark

建立 20–30 个固定、版本化证据包，每个包含：

```text
ResearchSpec
execution status
aggregate metrics
per-pair metrics
confidence intervals
gold verdict
```

至少覆盖：

- 主指标改善但 NLL 变差；
- accuracy 下降；
- seed/pair 数不足；
- 均值改善但 CI 穿过 0；
- 缺失 guardrail；
- NaN/Inf；
- 运行失败但残留 metrics；
- aggregate 通过但单个 pair 失败；
- 合法 positive；
- 合法 negative；
- 合法 inconclusive。

比较：

```text
LLM-only reviewer
legacy reviewer
ResearchSpec + deterministic scientific gate
```

指标：

- False Accept Rate（主指标）；
- False Reject Rate；
- verdict accuracy；
- review latency；
- review Token cost；
- repeated-run decision consistency。

已有 CIFAR-10 calibration 真实案例应作为一个 Evidence Pack：

```text
ECE mean improvement    0.0053228
NLL degradation        0.0015707
Observed pairs         2
Required pairs         5
Correct verdict        inconclusive / reject
```

---

## 8. 统计与展示

正式结果按 3 个独立重复报告：

- VCO@2h：每次原始值、均值和范围；
- TTFV：每次原始值，并标识 censored run；
- Token/VCO 与 GPU-sec/VCO；
- False Accept Rate 的分子和分母；
- 不只报告均值，保留每次运行的完整事件时间线。

核心图表：

1. 累计 VCO–时间曲线；
2. VCO@2h 主表；
3. TTFV；
4. Token/VCO 与 GPU-sec/VCO；
5. Idea 漏斗；
6. False Accept 混淆矩阵；
7. GPU request/run/release 时间线。

---

## 9. 实施 TODO

### P0：冻结协议和输入

- [ ] 冻结当前 Research Queue 实验 commit；
- [ ] 创建并评审 16 个固定 Idea；
- [ ] 给 Idea pool、Benchmark config 和模型配置生成 hash；
- [ ] 冻结统一 Token、GPU 和 wall-time 上限；
- [ ] 明确每个系统如何判定 terminal outcome。

### P0：实现 PortfolioBench runner

- [ ] 新建 `experiments/portfolio_bench/`；
- [ ] 实现统一 framework adapter 协议；
- [ ] 实现 `AutoResearchClaw-loop` wrapper；
- [ ] 实现 `RQ-Sequential` 配置；
- [ ] 实现 `RQ-NoEarlyExit` 配置；
- [ ] 实现 `RQ-Full` 配置；
- [ ] 统一导出 outcome、Token、GPU 和事件时间线；
- [ ] 自动验证 GPU 最终释放。

### P0：定义判定和聚合

- [ ] 实现 VCO 判定器；
- [ ] 实现 TTFV；
- [ ] 实现 Token/VCO 和 GPU-sec/VCO；
- [ ] 实现 False Accept Rate；
- [ ] 对 `VCO=0` 和无 accept 情况做明确处理；
- [ ] 输出 machine-readable `summary.json`。

### P1：科学门禁 Benchmark

- [ ] 创建 Evidence-Pack schema；
- [ ] 制作 20–30 个 gold packs；
- [ ] 增加双人或双模型独立 gold review；
- [ ] 运行 LLM-only、legacy、deterministic gate 三组；
- [ ] 输出混淆矩阵和错误案例分析。

### P1：ARC-Bench 接入

- [ ] 实现 `research_queue_adapter.py`；
- [ ] 修复本机 `paperbench_finalize` 依赖；
- [ ] 选择 6 个 ML topics；
- [ ] 先完成 2-topic smoke；
- [ ] 再运行 6 topics × 3 repeats；
- [ ] 使用同一 judge 和统一 submission schema。

### P1：正式实验

- [ ] 每组先运行 10 分钟 dry/smoke；
- [ ] 每组运行 2 小时；
- [ ] 每组完成 3 个独立重复；
- [ ] 验证无 orphan allocation、无重复 terminal import；
- [ ] 生成主表、时间曲线、漏斗和资源时间线；
- [ ] 将原始日志、配置和 hash 写入 InfoHub。

---

## 10. 完成定义

只有同时满足以下条件，PortfolioBench-2h 才算完成：

```text
同一 Idea pool
+ 同一时间/Token/GPU 上限
+ 至少四个核心对比组
+ 每组至少三个重复
+ VCO 判定可由 artifacts 独立重算
+ False Accept 有 gold audit
+ 所有资源最终释放
+ 原始配置、日志和结果可追溯
```

预期要验证的主张是：

> 在相同资源预算下，RQ-Full 比顺序 Pipeline 更快、更低成本地产生更多
> scientifically valid conclusive outcomes，同时不提高错误接受率，且在
> ARC-Bench 上维持可比的单任务研究质量。

在正式结果产生前，文档和介绍中必须将其表述为**待验证假设**，不能写成
已证实结论。

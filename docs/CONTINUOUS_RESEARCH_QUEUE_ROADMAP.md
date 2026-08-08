# Continuous Research Queue 重构 Roadmap

**状态：** Target architecture / implementation roadmap  
**更新时间：** 2026-08-08  
**目标：** 将当前 AutoResearch v2 控制面收敛为一个简单、通用、持续运行的
多 Idea 研究队列，同时保留异步并行和按需 GPU 调度。

> 本文描述目标架构和迁移顺序，不表示当前生产环境已经完成这些改造。
> 当前 `autoresearch_v2` 是迁移起点；旧的 23 阶段流水线和
> `CONTINUOUS_RESEARCH_FACTORY.md` 中的复杂 Work-Item/DAG 方案不再作为目标架构。

---

## 1. 最终目标

目标系统只有三个核心运行时组件：

```text
Research Workspace
        │
        ▼
Single Controller
        │
        ├── LLM / CPU Worker
        └── GPU Worker via ClusterBridge
        │
        ▼
SQLite + immutable artifacts
        │
        ├── read-only Dashboard
        └── asynchronous InfoHub outbox
```

核心定义：

1. **一个 Controller：** 唯一允许修改 Idea、Job、Attempt 状态的组件。
2. **一个持久化状态源：** SQLite 是运行状态真相；产物目录保存不可变证据。
3. **一种 Worker 协议：** Worker 读取 `job.json`，原子写入 `result.json`。
4. **多个独立 Idea 并行：** 每个 Idea 顺序推进，但多个 Idea 可以同时运行。
5. **持续 Idea 供给：** 没有全局 Cycle，也不等待一批 Idea 全部结束。
6. **统一实验任务：** Pilot 和 Scale 都是 `run`，只使用不同预算等级。
7. **GPU 按需申请：** 只在可执行 GPU Job 已准备完成后申请，结束后立即释放。
8. **旁路发布：** Report、Dashboard、InfoHub 不阻塞科学主链路。

每个 Idea 的主链路是一个小循环，而不是固定长流水线：

```text
candidate
   │ admit
   ▼
active
   │
   ├── prepare(revision N)
   ├── run(revision N, budget B0/B1/B2)
   └── review -> run_more | escalate | revise | conclude
                                              │
                                              ▼
                           concluded | quarantined
```

---

## 2. 必须保持的简化约束

这些约束优先级高于新增功能：

### 2.1 状态约束

- Idea 只有四个状态：
  - `candidate`
  - `active`
  - `concluded`
  - `quarantined`
- `concluded` 另存结论：
  - `positive`
  - `negative`
  - `inconclusive`
  - `not_admitted`
  - `cancelled`
- 不再使用 `designing/building/piloting/scaling/reporting` 作为 Idea 状态。
- 一个 Idea 同时最多有一个阻塞科学主链路的 Job。
- `publish` 可以在 Idea 结束后异步运行，不改变科学结论。

### 2.2 Job 约束

只保留五种 Job：

| Job | 作用 |
|---|---|
| `generate` | 补充候选 Idea |
| `prepare` | 生成一个不可变、可执行的实验 Revision |
| `run` | 在 CPU/GPU 上执行实验 |
| `review` | 根据证据选择下一步 |
| `publish` | 生成笔记、报告或论文并同步 InfoHub |

`review` 只能产生以下动作：

```text
run_more
escalate
revise
conclude
```

Controller 验证动作是否合法后才改变状态；模型和 Worker 不能直接推进状态。

### 2.3 预算约束

不再维护两套 Pilot/Scale 执行器和结果协议。统一为：

| 预算 | 含义 |
|---|---|
| `B0` | 最便宜的否证测试 |
| `B1` | 正常探索实验 |
| `B2` | 确认性实验 |

三者都调用同一个 `run` 协议，只改变：

- 样本数；
- Seed 数；
- GPU 数；
- Timeout；
- 数据 Split；
- 判定严格度。

### 2.4 通用性约束

Core 中不得硬编码：

- RSI；
- capability improvement；
- population search；
- verifier/memory；
- stopping/rollback；
- 固定研究分类比例。

研究方向由一个普通 Workspace 表达：

```text
research/<workspace-id>/
├── brief.md
├── config.yaml
├── assets/
└── optional_evaluator.py
```

第一版不建设复杂插件系统、继承体系或动态 Hook。

### 2.5 明确不做

在完成 24 小时验收之前，不做：

- 多 Controller 和 Leader Election；
- Kafka、Temporal 或新的外部工作流平台；
- 多 Agent 长期会话和 Agent 间通信；
- Work-Item DAG；
- 单个 Idea 内多个并发科学 Job；
- 运行中动态改变 GPU world size；
- 自动修改生产 Controller；
- 多层研究 taxonomy 和硬配额引擎；
- 自动对外投稿或发布。

---

## 3. 目标数据模型

### 3.1 Idea

```json
{
  "idea_id": "idea-...",
  "workspace_id": "rsi",
  "status": "active",
  "conclusion": null,
  "question": "...",
  "hypothesis": "...",
  "primary_metric": "...",
  "current_revision": 2,
  "next_budget": "B1",
  "priority": 0.72,
  "created_at": "...",
  "updated_at": "..."
}
```

领域标签可以存在，但只用于检索、去重和分析：

```json
{
  "tags": ["verifier", "curriculum", "memory"]
}
```

标签不参与 Core 状态转换。

### 3.2 Revision

每次 `prepare` 成功都生成一个不可变 Revision：

```text
ideas/<idea-id>/revisions/<revision-id>/
├── plan.json
├── task.json
├── run.sh
├── source/
└── manifest.json
```

失败的 Revision 不覆盖已接受 Revision。

### 3.3 Job 和 Attempt

- Job 表示一次逻辑工作；
- Attempt 表示 Job 的一次实际执行；
- 每个 Attempt 有唯一 `attempt_id`；
- 迟到的旧 Attempt 结果不能覆盖新 Attempt；
- 同一 `attempt_id` 的结果只能导入一次。

Worker 输出：

```json
{
  "status": "ok",
  "metrics": {},
  "artifacts": [],
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "gpu_seconds": 0
  },
  "error": null
}
```

基础设施失败和科学负结果必须严格分开：

```text
实验无效果                 -> concluded / negative
样本不足                   -> concluded / inconclusive
代码崩溃、资源超时、产物缺失 -> retry 或 quarantined
```

---

## 4. 迁移策略

采用 **旁路新内核 + 逐步切流**，不在正在运行的 v2 Controller 中一次性重写。

建议新包：

```text
researchclaw/research_queue/
```

可直接复用的稳定组件：

- 三档模型 Router 和本地 LLM Bridge；
- ClusterBridge resource-manager adapter；
- GPU task submit/probe/collect 基础能力；
- SQLite 备份方式；
- InfoHub HTTP client；
- Dashboard 静态资源和用量采集逻辑；
- artifact hash、原子文件和 attestation 工具。

不直接复用的控制逻辑：

- 现有五阶段 Idea 状态机；
- 分离的 Pilot/Scale Executor；
- RSI research-mode 配额；
- 阶段专属 Gate 链；
- Report 阻塞 Idea 完成的转换；
- 复杂的 runtime wrapper 决策逻辑。

迁移期间：

1. 旧 v2 和新 Queue 不得同时写同一个数据库；
2. 新 Queue 使用独立 `system_id`、数据库和 state directory；
3. 先跑模拟和 Shadow，再接真实 LLM，最后接 GPU；
4. 新 Idea 逐步切入新 Queue，旧 v2 只排空已有 Idea；
5. 通过 24 小时验收后再删除或归档旧控制面。

---

## 5. 实施阶段

估算以工程工作日表示。每个阶段必须可单独回滚，未满足退出标准不得进入下一阶段。

顺序依赖：

```text
Phase 0
  → Phase 1
  → Phase 2
  → Phase 3
  → Phase 4
  → Phase 5
  → Phase 6
  → Phase 7
```

粗略工程量为 **18–32 个工程工作日，加一次连续 24 小时验收**。
这是工作量估算，不是承诺日期；真实进度以每个 Phase 的退出标准为准。

### Phase 0 — 冻结边界和建立基线（1–2 天）

**目标：** 在改代码前固定最小架构和当前生产基线。

交付：

- 本文档进入仓库并标记旧 Factory Roadmap 已被取代；
- 记录当前 v2 的状态数量、Job 种类、测试数和生产配置；
- 新 Queue 使用独立 feature flag、CLI、数据库和 state directory；
- 建立禁止依赖旧 `pipeline`、`rsi supervisor` 和 `factory` 控制面的架构测试；
- 保存当前 24 小时运行数据作为吞吐、失败率和资源基线。

退出标准：

- 当前生产服务不受影响；
- 原有测试全部通过；
- 新旧控制面的写入边界有自动测试；
- Roadmap 中的“不做”清单得到确认。

---

### Phase 1 — 最小状态内核和模拟 Worker（2–4 天）

**目标：** 先证明状态模型可靠，不接真实模型和 GPU。

交付：

- `Idea / Revision / Job / Attempt / Event / Outbox` 最小表；
- 四种 Idea 状态、五种 Job、三种预算等级；
- Controller 单写者锁；
- 原子 `job.json -> result.json` Worker 协议；
- 幂等结果导入、Attempt fencing、固定次数重试；
- Controller 重启后的统一 reconciliation；
- fake Idea Generator、fake Worker 和 deterministic clock。

退出标准：

- 10 个 Idea 可异步推进且互不污染；
- 随机杀死 Controller 后无丢失、无重复执行、无状态倒退；
- 重复提交同一个 `result.json` 不产生第二次状态转换；
- 连续 24 小时模拟运行无死锁；
- 状态机和 schema 测试覆盖所有合法转换。

---

### Phase 2 — 持续 Idea 队列和真实 LLM/CPU 并发（2–4 天）

**目标：** 移除 Cycle，让 Idea 持续补充并并行完成 Prepare/Review。

交付：

- Candidate low/target watermark；
- 小批量 `generate`，语义去重和最小准入；
- 每个 Idea 独立顺序推进；
- 全局 `max_llm_jobs` 和 `max_cpu_jobs`；
- 三档模型映射：
  - Utility：检索整理；
  - Worker：Idea、Prepare、代码和写作；
  - Decision：准入与 Review；
- LLM Timeout、退避和用量记录；
- `brief.md + config.yaml + assets/` Workspace。

退出标准：

- 至少 8 个 Idea 同时 active；
- 任意慢 Idea 不阻塞 Idea 补充和其他 Idea；
- 一个 LLM Job 超时不会占死全局队列；
- Controller 重启后可继续未完成任务；
- Core 测试中不出现 RSI 专属概念。

---

### Phase 3 — 统一 Run 和预算升级（3–5 天）

**目标：** 用一个执行协议替代 Pilot/Scale 两套路径。

交付：

- 一个 `RunExecutor`，同时支持 CPU 和 GPU resource hint；
- `B0/B1/B2` 预算模板；
- Revision 与 Run 解耦：同一 Revision 可在不同预算下执行；
- 通用 Cheap Test 字段，不硬编码 mechanism activation；
- Review 动作：
  - `run_more`
  - `escalate`
  - `revise`
  - `conclude`
- Controller 确定性检查预算、Split、Seed 和证据引用；
- `publish` 从科学完成路径中移出。

退出标准：

- B0 失败可以直接形成有效负结果；
- B0/B1/B2 使用相同结果 schema；
- B2 必须使用独立确认数据或明确的 confirmatory contract；
- Worker 伪造 `"decision": "escalate"` 不会推进状态；
- Report/InfoHub 故障不改变 Idea 科学结论。

---

### Phase 4 — 弹性 GPU 调度（3–5 天）

**目标：** 保持多 Idea 并行，并让 GPU 仅按实时任务需求申请。

交付：

- 复用 ClusterBridge resource-manager adapter；
- `run` Job 在命令、Revision、Timeout 和结果目录全部就绪后才进入 GPU 队列；
- 每个 GPU Attempt 使用固定卡数，运行中不动态扩缩；
- 全局只配置：

```yaml
max_total_gpus: 32
max_gpus_per_job: 8
```

- 按优先级、等待时间和简单 backfill 调度；
- Pending Job 不持有 GPU；
- 逻辑 Job Lease 与物理 Allocation 分离；
- 最后一个 GPU Job 结束后立即将需求降到零并释放 Allocation；
- 启动恢复时采用仍然存活的远端任务；
- orphan/stale lease 在 Controller reconciliation 中清理，不新增独立服务。

退出标准：

- 多个 Idea 能在同一物理 Allocation 中并发运行；
- 实际占用永不超过 32 卡；
- 有可适配的等待任务时，不因队首大任务阻塞小任务；
- durable GPU demand 为零后 120 秒内触发释放；
- Controller 重启不重复提交仍在运行的 GPU Attempt；
- GPU 申请、等待或中台暂时不可用不被记为科学失败；
- AutoResearch 不启动、停止或管理 GPU spin。

---

### Phase 5 — 旁路持久化、前端和用量监控（2–4 天）

**目标：** 完善可观测性，但不让辅助系统进入关键路径。

交付：

- append-only Event 记录；
- Token、模型调用、GPU 秒和卡时统计；
- InfoHub Outbox：本地提交成功后异步同步；
- 每个 Idea 的 Research Note：
  - 问题和假设；
  - Revision；
  - 原始指标；
  - 正面、负面或不确定结论；
  - 失败与重试；
  - 代码和证据引用；
- Dashboard 只读 SQLite/API，不维护第二套状态；
- 前端展示：
  - Candidate / Active / Concluded / Quarantined；
  - 当前 Job 和预算等级；
  - GPU 请求、分配和运行；
  - Token/GPU 用量；
  - 最近事件和错误。

退出标准：

- 关闭 Dashboard 不影响 Controller；
- 关闭 InfoHub 后实验继续，恢复后自动补传且不重复；
- Dashboard 中每个状态都能回溯到 SQLite 记录；
- 模型和 GPU 用量可以按 Idea、Job、模型和时间窗口聚合。

---

### Phase 6 — Shadow、Canary 和逐步切流（5–7 天）

**目标：** 不做 Big Bang，逐级验证真实任务。

步骤：

1. **Shadow：** 新 Queue 读取相同 Research Brief，但只生成和评审，不执行实验；
2. **CPU Canary：** 只运行 Prepare、Review 和本地小实验；
3. **1–2 GPU Canary：** 限制 `max_total_gpus`，验证申请、执行、回收；
4. **8 GPU Canary：** 多 Idea 并行，执行 B0/B1；
5. **16 GPU Canary：** 加入 B2 和故障注入；
6. **32 GPU Soak：** 进入最终无人值守验收。

每一级都要比较：

- Idea 生成质量；
- 准入率；
- Prepare 成功率；
- 科学负结果率；
- 基础设施失败率；
- 平均 Job 等待时间；
- Token/Idea；
- GPU-hours/结论；
- 重启恢复情况。

退出标准：

- 新 Queue 不依赖人工修改数据库、清理 Lease 或重启 Job；
- 基础设施失败率不高于旧 v2 基线；
- 不出现跨 Idea 文件污染和结果串线；
- 至少有一个 Idea 完整经过 B0 → B1 → B2 → concluded；
- 至少有一个 Idea 在 B0/B1 正常早退并保存负结果。

---

### Phase 7 — 24 小时生产验收和旧控制面退役（24 小时测试 + 2–3 天收尾）

**目标：** 证明系统可以持续运行，而不是只完成一次 Demo。

24 小时验收必须满足：

1. 无人工重启、手工改数据库或手工清理 Lease；
2. Idea backlog 会自动补充，没有全局 Cycle 屏障；
3. 至少 8 个独立 Idea 在测试期间进入 active；
4. 同时存在多个不同阶段、不同 Idea 的异步 Job；
5. Controller 可从至少一次故障注入或计划重启中自动恢复；
6. 无重复 GPU Attempt、无丢失 terminal result；
7. 无跨 Idea artifact 污染；
8. GPU 使用不超过 32 卡；
9. 无 runnable GPU Job 时不保留 AutoResearch Allocation；
10. InfoHub 或 Dashboard 短时故障不阻塞研究；
11. 科学负结果、基础设施失败和不确定结果分类正确；
12. Token 和 GPU 用量账本可对账；
13. 至少一个完整科学闭环形成可追溯 Research Note；
14. 所有主要结论均能定位到 Revision、Attempt、原始指标和代码版本。

验收通过后：

- 停止向旧 v2 投递新 Idea；
- 排空或迁移旧 v2 中仍有价值的 Idea；
- 将旧 Controller 标记为 legacy；
- 删除生产配置中的 RSI 专属配额；
- 归档不再使用的 Stage、Gate 和 Wrapper；
- 保留只读历史数据库和产物，不篡改旧结论。

---

## 6. 实施优先级

### P0：没有它就不能安全切换

- 单一状态写入者；
- 四状态 Idea 模型；
- Job/Attempt 幂等；
- 持续 Idea watermark；
- 统一 `run` 协议；
- Controller 重启 reconciliation；
- GPU 按需申请和释放；
- 科学失败与基础设施失败分离。

### P1：完成 24 小时验收需要

- 三档模型路由；
- B0/B1/B2；
- 简单 backfill；
- InfoHub outbox；
- Dashboard 和用量统计；
- 故障注入；
- 逐级 Canary。

### P2：验收之后再评估

- 单个 Idea 内 Seed 并行；
- 自适应预算分配；
- Idea lineage、mutation 和 crossover；
- 复杂公平调度；
- 自动资源时长预测；
- 跨 Workspace 共享实验资产；
- 多论文组合与自动投稿。

P2 功能只有在真实日志证明存在明确瓶颈时才实现。

---

## 7. 关键工程指标

| 类别 | 指标 |
|---|---|
| 可靠性 | 丢失 Attempt = 0；重复 terminal 导入 = 0 |
| 隔离性 | 跨 Idea artifact 污染 = 0 |
| 持续性 | backlog 低于 watermark 后能自动补充 |
| 并发性 | 慢 Idea 不阻塞其他 Idea |
| GPU | 总占用 ≤ 32；零需求时自动释放 |
| 调度 | 有可适配任务时允许 backfill |
| 恢复 | Controller 重启后自动采用或重试任务 |
| 科学性 | infra failure 不计为 negative result |
| 证据 | 结论可追溯到 Revision、Attempt、指标和代码 |
| 用量 | Token/GPU 账本可按 Idea 和模型聚合 |
| 旁路 | InfoHub/Report/Dashboard 故障不阻塞实验 |

---

## 8. Roadmap 完成定义

重构不是以“代码写完”为完成，而是同时满足：

```text
架构更简单
+ 多 Idea 持续异步
+ GPU 按任务弹性申请
+ 24 小时无需人工干预
+ 科学证据可追溯
+ 通用 Core 无 RSI 硬编码
```

如果新设计需要重新引入大量 Stage、Actor、DAG、Agent 或状态副本，
默认视为架构回退，必须先用生产日志证明其必要性。

---

## 9. 核心 Benchmark TODO

核心架构评测已经冻结为 `PortfolioBench-2h`：

```text
固定 Idea 池
+ 相同 2 小时 / Token / GPU 上限
+ AutoResearchClaw-loop、RQ-Sequential、RQ-NoEarlyExit、RQ-Full
→ 比较 Valid Conclusive Outcomes
```

唯一 headline 指标：

```text
VCO@2h = 两小时内完成的科学有效且结论明确的研究结果数量
```

必要辅助指标为 TTFV、Tokens/VCO、GPU-sec/VCO 和 False Accept Rate。
完整协议、对比组、公平性约束、Evidence-Pack 评测和实施清单见：

[`PORTFOLIO_BENCHMARK_TODO.md`](PORTFOLIO_BENCHMARK_TODO.md)。

### 9.1 PortfolioBench 的最小同源证据实现

通用 Queue 仍允许不同领域自定义 `prepare/run/review`。但
PortfolioBench 不再让 LLM 先自由生成 synthetic `experiment.py`、再用
另一套 treatment 跑真实 Benchmark。其固定 adapter 路径为：

```text
ResearchSpec
→ 一次 treatment.py 生成与 preflight
→ 同一 frozen pairing universe 上的 B0/B1/B2
→ B2 deterministic final gate
```

- B0、B1、B2 使用同一个 treatment hash；
- 三个预算使用不相交的 seed partitions；
- B2 就是 confirmatory result，不再生成第二套实现或重复 promotion run；
- PortfolioBench 中 `max_runs_per_budget=1`，避免把同代码、同参数、同 seed
  的重复执行误称为独立证据；
- PortfolioBench 中 `max_revisions_per_idea=1`，避免模型看过 pilot
  evaluation labels 后修改 treatment，再把旧 pilot 当作新实现的证据。

这些是 Benchmark adapter 的公平性约束，不增加新的 Idea 状态、Actor、
DAG 或第二个 Controller。

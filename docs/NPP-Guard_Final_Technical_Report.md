# NPP-Guard v1 最终技术报告

> 版本范围：01–19 已完成里程碑。本文整理当前仓库的真实实验记录，不新增模型训练，不把研究 prototype 表述为真实核电站部署或安全保证。

## 摘要

NPP-Guard 是一个面向 NPPAD 仿真时序数据的研究型核电事故诊断原型。项目从早期异常检测和 LOCA 分类出发，逐步加入完整 trajectory/sample_id 分组、train-only preprocessing、潜在标签泄漏审计、保护动作前窗口、severity-blocked / extrapolation OOD 验证、conformal 与距离拒答策略、冻结 artifact release、模型归因和 Streamlit Dashboard。最终 v1 对每条输入先做数据质量检查，再使用 120 s 过程窗口输出事故族、置信度、conformal set、距离 OOD 警告以及 `accepted / requires_review / unknown / invalid_input` 状态。LOCA severity 只在 `accepted + LOCA family + Tier A` 时运行，并明确标为 exploratory。

当前结果支持“可复现、会拒答、可解释的研究型诊断流程”，不支持真实机组运行控制、安全关键决策、因果解释或外部泛化保证。关键 release gate 为：locked release test 101 条轨迹，API/batch parity `101/101`，最大浮点差 `0`，regression `18/18`；Dashboard 验证为 5/5 smoke、3/3 helper、HTTP 200、AppTest 0 个运行异常。

## 项目背景与问题定义

核电事故诊断同时面对三类风险：早期窗口可能没有足够可辨识信息，仿真数据的事故严重度与过程变量可能高度耦合，随机拆分时间行还会把同一条轨迹泄漏到训练和测试。NPP-Guard 将任务定义为：在给定单条 CSV 轨迹和最多 120 s 诊断窗口时，判断数据是否合格，给出研究型事故族判断或主动拒答，并在门控满足时给出 LOCA 严重度探索性估计。

## 数据集 NPPAD 与数据结构

项目使用本地 NPPAD 操作 CSV。09 审计记录了 17 个事故类别和 1216 条 trajectory/sample_id 轨迹；首版多事故分类固定使用 12 个样本相对充足的类别，单例类别延后。主模型只使用 38 个严格过程变量，每个变量提取 `last / delta_t0 / mean / std / slope`，形成 190 个派生特征。CSV 需要包含 `TIME`、严格变量、单调时间、约 10 s 采样间隔和至少 120 s 覆盖；额外列会被报告但不进入 v1 特征矩阵。

## 研究路线总览（01–19）

路线图见 [`docs/figures/20_research_route.mmd`](figures/20_research_route.mmd)。核心转折是：04/05 的 early-warning 失败推动 06 做 Normal reference audit；10/11 发现 30/60 s 区分度弱后保留 120 s 作为探索窗口；13/14 的 OOD 结果阻止直接进入深度学习；15/16 将任务改成 coverage-aware、允许 Unknown/Requires Review；17.2 把研究 benchmark 与 locked release benchmark 分离；18/19 把冻结推理接入解释层和 Dashboard。

## 数据审计与方法学设计

### Normal 不足与 early-warning 边界

06 审计发现独立 Normal reference 只有 `Normal/1.csv`，且功率轨迹并非固定功率；事故前只有 `t=0` 一个采样点。04/05 虽然在最终检测层面找到事故，但保护前确认不足以支持 early-warning 声明。因此项目转向 LOCA severity、protection-time 与多事故 diagnosability 研究。

### Group split、train-only preprocessing 与 leakage

模型划分单位是完整 `trajectory/sample_id`，从不随机拆散时间行。预处理、校准、阈值选择和距离拟合只使用各自允许的训练/校准数据。09 明确标记潜在 leakage 变量及 SLBIC 初始工况混杂，后续 10–16 同时保留严格 38 变量组和敏感性组，不静默删除变量。

### Pre-protection、OOD 与 severity extrapolation

严格 pre-protection 只保留已知保护时间且窗口末点位于保护动作之前的轨迹。OOD 阶段分别报告 random、severity-blocked 和 severity-extrapolation，避免把同一指标混成泛化结论。13 的 120 s A 组 fixed-12 Macro-F1 分别为 `0.6573 / 0.5931 / 0.4826`；extrapolation 结果因此只作为压力测试。

## LOCA severity estimation（07）

07 使用 100 条 LOCA 轨迹，把文件编号 `1..100` 作为 100 cm² 破口面积中的仿真严重度标签。validation 选择 `Ridge + 120 s`，grouped test 为 MAE `0.4492 percentage points`、RMSE `0.5673 percentage points`、R² `0.9996`。相关性审计提示 `P/LVPZ/TSAT/VOL` 与严重度近乎完全耦合；排除这些变量后的 sensitivity test MAE 约为 `0.972`，所以 07 的高分不能等同于稳健外部泛化。

## Protection-time prediction（08）

08 使用严格 landmark 目标 `first_protection_s - landmark_s`，并排除 landmark 前已保护的轨迹。60 s + Random Forest 在严格 validation 中被选中，test MAE 为 `86.826 s`，RMSE 为 `338.735 s`，R² 为 `0.3303`。300 s 只有 6 条轨迹，不能作为泛化证据。由于没有建立有效的 severity-to-protection-time 链式输入，保护时间没有接入 v1，Dashboard 显示 `not_available_in_v1`。

## Multi-accident classification（10–14）

10 的 A strict 38 test Macro-F1 在 30 s、60 s、120 s 分别为 `0.1928`、`0.1928`、`0.7526`，Balanced Accuracy 分别为 `0.2727`、`0.2727`、`0.7924`。30/60 s 的相同结果触发了 cohort、窗口采样点、特征唯一性和预测分布审计；不能把 30 s 说成有效早期分类。12 在 471 条 matched trajectories 上重复 10 个 grouped split，120 s fixed-12 Macro-F1 均值约 `0.5784`，但 stage gate 仍要求更严格数据或外部验证。13 在 severity extrapolation 下 Macro-F1 降至 `0.4826`。14 的 family model Macro-F1 为 `0.6331`，two-stage subtype Macro-F1 为 `0.4829`，未证明层次化流程稳定改善细分类 OOD。

## Coverage-aware / selective diagnosis（15–16）

15 将任务改写为带 coverage 的 family diagnosis。random family-selective coverage 为 `97.7%`，severity extrapolation 下 family Macro-F1 为 `0.6660`，但当前 extrapolation threshold 均值为 0，实际拒答能力有限。16 比较 probability、margin、distance、combination 与 conformal；alpha=`0.10` 的经验 set coverage 为 `68.9%`，selective Macro-F1 为 `0.6700`，distance severity-shift proxy AUROC 为 `0.8221`。这些结果支持研究型 Unknown/Requires Review policy，不支持 90% 统计 coverage 保证。

## v1 inference core 与 release protocol（17.2）

17.2 固定 `artifacts/v1/release_split_manifest.json`，对 train、probability calibration、conformal calibration、distance fit 与 locked release test 做 zero-overlap 审计。v1 顺序为：family classifier → probability calibration → conformal alpha=`0.10` → distance OOD → status policy。release test 共 101 条轨迹，selective policy coverage 为 `51.5%`，accepted 子集 Macro-F1 为 `0.5000`，risk 为 `0`；状态分布为 accepted=52、requires_review=39、unknown=10。这里的 coverage 和 risk 只描述冻结研究 cohort，不是运行安全指标。

## Explainability（18）

18 未修改冻结 v1 推理路径。解释层使用 family permutation importance、冻结训练中心替换 local attribution、OOD 距离对角近似和 LOCA severity perturbation。locked cohort 的 Top variables 为 `P / WSTA / TAVG / WFWA / LSGA`；轻微输入扰动的 Top-5 Jaccard 为 `0.9722`，family top-k 相对随机移除差为 `+0.7532`。解释是模型归因，不是物理因果。18 gate 保持 artifact SHA-256 10/10、API/batch parity 101/101、regression 18/18。

## Dashboard（19）

19 使用 Streamlit，将 17.2 冻结推理和 18 解释层接入本地研究展示原型。启动命令：

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

真实示例覆盖 accepted、accepted LOCA、requires_review、unknown 和 50 s 截断 invalid_input。5/5 smoke、3/3 helper、HTTP 200、AppTest 0 个运行异常均通过；JSON export 不含原始全量 CSV。LOCA severity 只在 accepted + LOCA family + Tier A 时运行。

## 失败、负结果与路线调整

- 04/05 的 early-warning 保护前确认不足，停止把单条 Normal reference 当作通用预警基线。
- 30/60 s 多事故区分度弱，且出现相同指标，先做数据与 cohort 审计，不直接进入 LSTM/Transformer。
- severity extrapolation 性能明显下降，说明 random split 分数不能代表跨严重度泛化。
- 14 的 severity-invariant 与两阶段 family→subtype 没有稳定增益，路线转向 coverage/task redesign。
- 15/16 的 reject 机制逐步形成 v1 policy，但经验 coverage 不等于统计保证，subtype OOD 仍未解决。

## 最终 v1 能力矩阵与限制

| 能力 | 当前状态 | 证据 / 限制 |
| --- | --- | --- |
| CSV 数据质量检查 | 可用 | 38 strict variables、120 s、NaN/Inf、重复时间点、额外列 |
| Family diagnosis | Tier A/B/C 分层 | 只在研究 cohort 与固定 policy 下解释 |
| Unknown / Requires Review | 可用 | conformal、distance、capability gating 组合 |
| LOCA severity | gated exploratory | 只在 accepted + LOCA + Tier A 运行 |
| Protection time | v1 不可用 | 研究记录 MAE 86.826 s，未建立有效链式输入 |
| Explainability | 可用 | model attribution，不是因果解释 |
| Dashboard / JSON export | research prototype | Streamlit 本地展示，不用于 safety-critical deployment |

## 结论与下一步

NPP-Guard v1 的交付价值在于把“模型给一个类别”扩展成“数据检查、诊断、拒答、OOD 提示、解释和可复现 release protocol”的完整研究链路。下一步应优先补充固定功率 Normal reference、更多跨工况与 severity 覆盖、独立外部测试和更严格的 subtype OOD 验证，再讨论时序深度模型或更完整的 protection-time 链式推理。

## 复现入口

- 总表：[`NPP-Guard_Key_Results_Table.md`](NPP-Guard_Key_Results_Table.md)
- 架构图：[`figures/20_npp_guard_architecture.mmd`](figures/20_npp_guard_architecture.mmd)
- 研究路线：[`figures/20_research_route.mmd`](figures/20_research_route.mmd)
- 发布验证：[`../results/20_release_validation_summary.json`](../results/20_release_validation_summary.json)
- Dashboard：`dashboard/app.py`
- CLI：`python -m src.inference.cli --input <csv>`

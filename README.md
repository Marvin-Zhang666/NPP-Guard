# NPP-Guard

## Nuclear Power Plant Early Accident Detection and Diagnosis

NPP-Guard 是一个面向核电厂仿真时序数据的研究型原型，用于探索事故早期异常检测、事故类别识别和跨运行工况验证。项目基于 NPPAD（Nuclear Power Plant Accident Data）数据集，当前重点是：

- 用正常工况建立参考尺度，并检测持续的异常偏离；
- 以保护动作前的短时间窗口为目标，对 LOCA、SLBIC 和 FLB 做保守分类；
- 在变功率场景中，验证与匹配正常功率过渡模板的异常门；
- 保留低置信度的 `Unknown` 结果，而不是强行给出事故类别。

这是科研和学习用途的可复现实验基线，不是核电厂安全保护系统，也不适用于真实机组运行或操作决策。

## 研究动机

事故诊断系统不能只在事故已经明显发生、甚至保护动作已经完成后再给出结论。NPP-Guard 因此把“较早发现异常”和“在不确定时拒答”作为当前设计重点：先用异常门判断测试曲线是否持续偏离正常参考，再在限定的早期窗口内进行分类。

项目同时保留了对直接描述破口或泄漏的变量进行排除的实验路径，用于观察模型是否能够依靠间接过程信号工作，而不是直接读取答案变量。

## 当前功能

### 1. 全功率异常门

`src/validate_npp_guard.py` 和 `src/npp_guard_inference.py` 实现了基于 `Normal/1.csv` 的参考比较：

- 参考基线：`TIME <= 100 s` 的正常数据；
- 规则特征：`WECS、WSTB、SCMA、TRB、PRB、LWRB、PRBA`；
- 分数：测试曲线相对参考曲线的标准化绝对偏差；
- 联合分数：各规则特征分数的中位数；
- 当前报警条件：分数连续超过 `5.0` 至少 `3` 个时间点；
- 全功率早期推理的分析上限：`2000 s`。

### 2. 保护动作前的保守事故分类

`src/validate_gated_fault_classifier.py`、`src/validate_loca_slbic.py` 和 `src/train_full_power_model.py` 提供了当前的分类基线：

- 类别：`LOCA`、`SLBIC`、`FLB`；
- 早期窗口：事故发生后前 `100 s` 的特征摘要；
- 模型：`StandardScaler + LogisticRegression`；
- 数据划分：按连续 case 编号进行五折验证（`1–20`、`21–40`、…、`81–100`），避免随机打散带来的过度乐观；
- 置信度阈值：最大类别概率低于 `0.60` 时输出 `Unknown`；
- 早期特征：包含压力、温度、液位、流量、燃料温度、DNBR 和反应堆/蒸汽发生器相关变量，代码中显式排除了直接破口指标和 ECCS/安全注入流量。

当前推理模型保存在 `models/npp_guard_full_power_early.joblib`，其元数据在 `models/npp_guard_full_power_early.json`。该模型只声明适用于全功率场景。

### 3. 变功率场景

`src/validate_variable_power_gate.py` 使用与事故场景匹配的正常功率过渡轨迹作为参考，验证 `80%→100%` 和 `100%→80%` 方向下的条件化异常门。`src/validate_variable_power_classifier.py` 还保存了跨功率方向的实验性分类结果。

但 `src/npp_guard_inference.py` 当前不会把变功率场景的分类结果当作已验证能力，而是返回 `NotValidatedAcrossPowerConditions`。这条限制是有意保留的，不应在现阶段把变功率分类当作可部署功能。

### 4. EDA Notebook

`notebooks/01_LOCA_EDA.ipynb` 包含 Normal/LOCA 对比、特征排序、标准化异常分数、持续报警检查以及多个时间点（`100、300、600、1000、1500、2000 s`）的间接特征探索。

## 第一阶段完成情况（v0.3）

第一阶段已完成并固定为可复现实验基线，包含：

- 100 级 LOCA 严重度分析，以及严重度、检测时间和保护动作时间的关联整理；
- ML Dataset Audit，核对事故类别、案例数量、时间网格、注入时间和保护事件表；
- 保护动作前的 38 个常规过程量筛选；
- 基于单条 Normal reference 的静态异常检测 baseline；
- 在相同评估协议下加入 10 s 变化率，以及 30/60/120 s 动态斜率的 early-warning 实验。

### 保护前检出结果与限制

这里的“保护前检出”要求异常确认时间不晚于该案例的首次保护动作时间。04 和 05 中各方法最终都检测到 100/100 个 LOCA，但多数确认发生在保护动作之后，因此不能把“最终检测到”表述为有效 early warning。

| 实验 / 特征组 | 保护前检出 | Normal 持续误报 | 结果解释 |
| --- | ---: | --- | --- |
| 04 Robust Z-score | 0/100 | 否 | 静态 baseline 未实现保护前确认 |
| 04 Mahalanobis | 2/100 | 否 | 静态 baseline 中最好的结果，但仍不足以支持 early-warning 声明 |
| 04 PCA reconstruction error | 0/100 | 否 | 未优于 Mahalanobis |
| 05 A：静态 Mahalanobis | 2/100 | 否 | 与 04 静态结果逐行一致 |
| 05 B：静态 + 10 s 变化率 | 0/100 | 否 | 未改善保护前检出 |
| 05 C：B + 30/60/120 s 斜率 | 0/100 | 否 | 未改善保护前检出；120 s 动态窗口还带来明确 warm-up 期 |

Normal reference 每组有 2 个点超过阈值，但没有形成 3 点持续报警。这个“无持续误报”结果只能说明当前单条参考轨迹上的行为；阈值校准和误报检查都依赖同一条独立 Normal reference，不能据此宣称已获得可泛化的 early-warning 性能。当前负结果更支持优先扩充 Normal reference，而不是继续调阈值或直接上 LSTM。

## 第二阶段第一步：Normal-reference audit

06 阶段对本地 NPPAD 的 1221 条轨迹完成了 Normal reference 全量审计，结果固定为：

| 分类 | 数量 | 结论 |
| --- | ---: | --- |
| A：独立 Normal reference | 1 | `Normal/1.csv`；合法但 `PWR` 从约 100% 降至约 40%，不是固定功率轨迹 |
| B：条件化辅助 Normal | 4 | `NORM_*` 变功率 Normal MDB；可用于匹配功率过渡的辅助实验 |
| C：不能作为 Normal reference | 1216 | 事故轨迹；不能把事故前初始化点重新标记为独立 Normal |

所有事故报告的首个注入时间为 `0.5 s`，而 CSV 采样间隔为 `10 s`；事故前只有 `t=0` 一个采样点，实际可用事故前窗口长度为 `0 s`。因此，NPPAD 当前数据不足以支撑严谨的 normal-reference LOCA early warning 研究。第二阶段主线转向 **LOCA severity estimation / protection-time prediction**；变功率 Normal 仅保留用于条件化辅助实验。

审计产物：`notebooks/06_normal_reference_expansion.ipynb`、`src/normal_reference.py`、`results/normal_reference_inventory.csv`、`results/normal_reference_audit_summary.csv` 和 `results/normal_reference_audit_summary.json`。

## 第二阶段第二步：LOCA severity estimation

07 阶段使用全部 100 条 LOCA 轨迹，把文件名 `1..100` 作为 100 cm² 破口面积中的严重度标签。观察窗口为 `30、60、120、300 s`，对应实际采样点数分别为 `3、6、12、30`。特征只来自 03 审计保留的 38 个常规过程量，每个过程量提取末值、相对 `t=0` 变化量、均值、标准差和线性斜率，共 190 个派生特征；直接破口/泄漏、保护/控制动作、辐射/剂量以及事故后果/安全裕度变量不作为候选过程特征。

数据按完整 `sample_id` 做可复现的 `60/20/20` train/validation/test 划分，预处理只在 train 拟合。比较了训练集均值 baseline、Ridge、Random Forest 和 Gradient Boosting。validation 选择的最佳组合为 `Ridge + 120 s`，test 指标为 `MAE=0.449`、`RMSE=0.567`、`R²=0.9996`；同窗口训练集均值 baseline 的 test MAE 为 `25.05`。

但训练集相关性审计发现，120 s 窗口中的 `P/LVPZ/TSAT/VOL` 派生特征与严重度存在近乎完全的耦合。排除这些强耦合变量后，validation 选择变为 Random Forest，test 指标为 `MAE=0.972`、`RMSE=1.398`、`R²=0.9976`。因此，`MAE=0.449` 不能被表述为稳健泛化性能；更保守的结论是，前 120 s 的过程响应包含很强的严重度信息，但结果对过程量—严重度耦合敏感，需要独立工况和更严格的外部验证。

审计产物：`notebooks/07_LOCA_severity_estimation.ipynb`、`src/severity_features.py`、`src/severity_models.py`、`results/07_*` 和 `results/figures/07_*`。

## 第二阶段第三步：Protection-Time Prediction

08 阶段采用严格的 landmark 设计预测首次保护动作剩余时间，只使用 landmark 之前的观测，并排除已经在 landmark 前发生首次保护动作的轨迹。比较了 `30、60、120、300 s` 四个 landmark：有效轨迹数分别为 `100、99、76、6`；严重度覆盖分别为 `1–100`、`1–100（缺 43）`、`1–77（缺 43、78–100）` 和 `1–6`。

严格 grouped validation 选择的最佳组合为 `60 s + Random Forest`。Test 结果为 `MAE=86.826 s`、`RMSE=338.735 s`、`R²=0.330`；排除 `P/LVPZ/TSAT/VOL` 后选择和测试结果不变。30/60 s 下 process-only 均不优于 severity-only，120 s 只有有限改善，300 s 虽然 process 特征结果更好但只有 6 条轨迹、test 仅 2 条，不能作为泛化证据。该阶段保留为辅助模块，不把 protection-time prediction 当作安全性能承诺。

审计产物：`notebooks/08_protection_time_prediction.ipynb`、`src/protection_time_features.py`、`src/protection_time_models.py`、`src/protection_time_report.py`、`results/08_*` 和 `results/figures/08_*`。

## 第二阶段第四步：Multi-accident dataset audit

09 阶段对本地 NPPAD 的全部事故轨迹完成了进入多事故分类前的数据审计：共 17 个事故类别、1216 条完整 trajectory/sample_id 轨迹，类别样本量为 `1–110`。首版分类只建议纳入 12 个样本充足类别：`FLB`、`LLB`、`LOCA`、`LOCAC`、`LR`、`MD`、`RI`、`RW`、`SGATR`、`SGBTR`、`SLBIC`、`SLBOC`；`ATWS`、`LACP`、`LOF`、`SP`、`TT` 各只有 1 条轨迹，暂缓独立泛化验证。

窗口覆盖方面，30 s 和 60 s 均为 `1216/1216` 全覆盖；120 s 为 `1198/1216`，覆盖率 `98.52%`，缺失来自 RI 短轨迹。严格过程变量沿用 03 的 `feature_groups.csv`，共 38 个；SLBIC 的 101 条轨迹中有 25 条额外包含 `WPCS/WPFW/WPMU` 三列，已确认 38 个严格变量均存在，因此后续模型统一使用 38 个变量的交集，不使用额外列。

审计还标记出 16 个类别—变量组合的潜在 label leakage（包括直接事故量、控制量、辐射/后果量，以及候选变量 `LVCR` 的类别特异状态变化）；这些变量未在 09 阶段自动删除，而是留给后续敏感性实验。另有 SLBIC 集中的 14 个候选变量表现出初始工况混杂，重点为 `TAVG/THA/THB/TCA/TCB/PSGA/PSGB/WFWA/WFWB/WSTA/WSTB/QMWT/QMGA/QMGB`；后续分类必须报告排除这些变量或仅使用相对 `t=0` 变化特征的敏感性结果。

审计产物：`notebooks/09_multi_accident_dataset_audit.ipynb`、`src/multi_accident.py`、`results/09_multi_accident_*` 和 `results/figures/09_*`。数据划分单位固定为完整 trajectory/sample_id，禁止随机拆分时间行。

## 第三阶段第一步：Temporal diagnosability analysis

11 阶段使用 12 个样本充足事故类别和 38 个严格过程变量，比较 `10/20/30/40/60/90/120 s` 窗口，并分别报告 all-sample 与 strict pre-protection-only 结果。strict 集只保留已知 `first_protection_s` 且窗口末点仍早于首次保护动作的轨迹；未知保护时间的轨迹全部排除。

- all-sample 在 `10–90 s` 的 Macro-F1 约为 `0.193`；`120 s` 验证选择 Random Forest，Accuracy `0.785`、Macro-F1 `0.749`、Balanced Accuracy `0.785`。
- strict pre-protection-only 在 `10–40 s` 有 660 条轨迹、`60/90 s` 有 659 条、`120 s` 有 471 条；`10–90 s` 固定 12 类 Macro-F1 约为 `0.195`，Observed-class Macro-F1 约为 `0.293`。`120 s` 验证选择 Logistic Regression，固定 12 类 Macro-F1 `0.647`、Observed-class Macro-F1 `0.971`、Balanced Accuracy `0.969`。
- strict `120 s` 只有 8 个类别有有效支持；`RI` 和 `RW` 的 test support 很小，`LLB`、`LR`、`MD`、`SLBOC` 没有可用保护时间，因此这些类别不能据此作可靠 Recall 结论。
- 在固定 `120 s` 可用轨迹的 matched cohort 中，strict pre-protection 的 HGB Macro-F1 仍从 `60 s` 的 `0.202` 提升到 `120 s` 的 `0.566`。因此 120 s 的提升不能全部归因于保护动作后的信息，但类别组成、样本量和支持度变化仍要求外部稳健验证。
- B 组排除 `LVCR` 后性能基本不变；C 组排除 14 个 SLBIC 初始工况变量后，`10–90 s` 基本不变，但 `120 s` strict 结果受稀疏支持影响，不能据此宣称已经消除混杂。

11 阶段的结果说明事故早期过程响应可能包含可用于分类的信息，但不等于已经获得可靠的早期诊断能力；在稳健性和外部验证完成前，不直接进入 Transformer。

## 第三阶段第二步：Robust temporal validation

12 阶段在 471 条 strict matched trajectories 上完成了重复稳健性验证：使用 10 个固定 seed、按完整 `trajectory/sample_id` 的 60/20/20 分层划分，并让 30/60/90/120 s 共用同一 cohort 和 split。比较 Logistic Regression、Random Forest、HistGradientBoosting，以及普通和 class-balanced 训练；95% CI 使用重复 test split 的正态近似。

在固定 12 类指标、38 个严格过程变量的 A 组下，120 s 的结果为 Fixed-12 Macro-F1 `0.578±0.004`、Observed-class Macro-F1 `0.991±0.007`、Balanced Accuracy `0.989±0.011`；60 s 对应约为 `0.220±0.010`、`0.377±0.018`、`0.448±0.011`。同一 matched cohort 下，120 s 的优势对 10 个 split 稳定，且排除 `LVCR` 或 14 个 SLBIC 初始工况候选变量的敏感性结果基本不变。

该结果仍不能直接作为进入深度学习的依据：120 s cohort 中 `RW` 只有 3 条轨迹，10 次 test support 均为 0；`LLB`、`LR`、`MD`、`SLBOC` 没有已知 `first_protection_time`，不具备严格 pre-protection 评估资格。可评估类别为 `FLB`、`LOCA`、`LOCAC`、`RI`、`SGBTR`，其 Recall 均值分别约为 `1.000`、`0.980`、`0.975`、`0.967`、`1.000`。12 阶段判定为先进行更严格的 OOD / severity-blocked validation 和 event-parser recovery。

审计产物：`notebooks/12_robust_validation.ipynb`、`src/robust_validation.py`、`results/12_*` 和 `results/figures/12_*`。

## 第三阶段第三步：OOD / Severity-blocked validation

13 阶段对 12 个有数据的事故类别按各自定义的 severity 做了严格 OOD 审计。NPPAD README 明确说明文件编号对应 severity；LOCA/LOCAC/SLBIC/SLBOC/FLB 使用破口面积比例，SGATR/SGBTR 使用整根管破裂比例，RW/RI、MD、LR、LLB 分别使用其类别定义的杆位移、未硼化注入、负荷拒绝和 letdown 流量单位，因此没有跨类别混用统一数值阈值。

通用事件解析器扫描了 1,211 条首版类别报告，解析出 696 条 protection event。相对 09 的旧表新增 36 条：`LR` 30 条 Safety Relief Valve opening、`RI` 6 条 Safety Relief Valve opening；`LLB`、`MD`、`SLBOC` 的报告仍只有事故注入行，保持 unknown，没有人为填充 protection time。严格 matched cohort 为 505 条：`FLB=99`、`LOCA=76`、`LOCAC=79`、`LR=28`、`RI=34`、`RW=3`、`SGATR=44`、`SGBTR=54`、`SLBIC=88`。

固定 38 个过程变量和 A 组下，120 s 结果为：随机 baseline Macro-F1 `0.657±0.009`、Balanced Accuracy `0.984±0.016`；severity-blocked `0.593±0.064`、`0.921±0.069`；severity extrapolation `0.483±0.110`、`0.813±0.088`。blocked 使用 low/middle/high 三个连续 block，extrapolation 使用 low→high 和 high→low 两个方向；每个 OOD test block 与训练数据之间保留一条相邻 severity guard。60/90 s 的 OOD Macro-F1 约为 blocked `0.189±0.020`、extrapolation `0.183±0.024`，Balanced Accuracy 均约 `0.375`，因此 120 s 优势在 OOD split 上仍存在，但 extrapolation 的绝对 Macro-F1 未达到可靠进入深度学习的门槛。

敏感性方面，排除 `LVCR` 的 B 组与 A 组几乎一致；排除 14 个 SLBIC 初始工况变量的 C 组在 blocked 120 s 下降至 Macro-F1 `0.570`、Balanced Accuracy `0.875`，extrapolation 约为 `0.480`、`0.779`。nearest-severity gap 均值从 random 的 `1.22` 增至 blocked 的 `11.49`，再增至 extrapolation 的 `20.31`，证明 OOD test 确实远离训练 severity。RW 只有 3 条，保留在 matched/random 审计中，但不纳入 blocked/extrapolation 的三路 OOD 结论。

判定为 **B：暂不进入 GRU/LSTM/TCN**。120 s 在 blocked 和 extrapolation 都优于 60/90 s，且主要大类 Recall 仍较高，但 extrapolation Macro-F1 低于 `0.50`，RI、LOCAC、SLBIC 等类别 Recall 不稳定；应先补充 severity 覆盖和外部/跨工况验证。

审计产物：`notebooks/13_ood_severity_blocked_validation.ipynb`、`src/event_parser.py`、`src/ood_validation.py`、`results/13_*` 和 `results/figures/13_*`。13 阶段作为独立里程碑提交并推送到 `origin/main`。

## 第三阶段第四步：Severity-invariant / hierarchical diagnosis

14 阶段复用 13 的 505 条完整 `trajectory/sample_id` matched cohort 和 random / severity-blocked / extrapolation split，汇总 OOD error decomposition、severity-invariant 特征和基于 NPPAD README Table 1 正式名称建立的 family taxonomy。family mapping 覆盖 17 个类别、10 个研究型事故族；family 分组是本项目的可解释研究分类，不是 NPPAD 原生标签。

120 s severity extrapolation 下，12-class baseline Macro-F1 为 `0.4826`、Balanced Accuracy 为 `0.8125`；变化/归一化 severity-invariant 特征仅提升到 Macro-F1 `0.4905`，Balanced Accuracy 仍为 `0.8125`。独立 family-level classifier 的 Macro-F1 为 `0.6331`，但两阶段 family→subtype 的 end-to-end subtype Macro-F1 仅 `0.4829`，没有改善细分类 OOD 能力。

方向性审计显示 RI、LOCAC、SLBIC 在 low→high 与 high→low severity extrapolation 间明显不稳定：RI low→high Recall 为 `0` 且主要混淆到 `SGATR`；LOCAC high→low Recall 为 `0` 且主要混淆到 `SGATR`；SLBIC high→low Recall 为 `0` 且主要混淆到 `RI`。因此 14 阶段结论为：**继续 coverage/task redesign，暂不进入 GRU/LSTM/TCN 等深度学习。**

审计产物：`notebooks/14_severity_invariant_hierarchical_diagnosis.ipynb`、`src/severity_invariant.py`、`src/ood_error_analysis.py`、`src/hierarchical_diagnosis.py`、`src/milestone14.py`、`results/14_*` 和 `results/figures/14_*`。14 阶段作为独立里程碑提交并推送到 `origin/main`。

## 第三阶段第五步：Coverage-aware task redesign

15 阶段在同一批 505 条完整 `trajectory/sample_id` matched cohort 上复用 14 的 family mapping、38 个严格过程变量以及 random / severity-blocked / severity-extrapolation 划分。覆盖审计结果为：class Tier A/B/C = `8/1/8`，family Tier A/B/C = `4/2/4`。

120 s family-level 诊断比强制 12-class 更稳健：family Macro-F1 在 random / blocked / extrapolation 下分别约为 `0.8469/0.7603/0.6660`。但现有 selective reject 仅拒绝 `23/1525=1.51%` 测试样本，且 blocked/extrapolation 的验证阈值为 `0`，无法有效优先拒绝大 severity-gap OOD。因此当前 family-level 能力只能标为有限 supported / exploratory，subtype OOD 与 reject 机制仍不足，暂不进入深度学习。

审计产物：`notebooks/15_coverage_aware_task_redesign.ipynb`、`src/coverage_task_redesign.py`、`results/15_*` 和 `results/figures/15_*`。15 阶段 Notebook 已全量执行通过；结果保留显式 `Unknown` reject 状态，但不构成安全部署能力承诺。

## 第三阶段第六步：OOD-aware selective diagnosis

16 阶段在 13 的 trajectory/sample_id grouped random / severity-blocked / severity-extrapolation split 上，使用固定 120 s family-level 诊断比较 calibrated probability、prediction margin、train-only class-conditional distance、组合门和 split conformal prediction。severity extrapolation 下 forced family baseline Macro-F1 为 `0.666`、risk 为 `0.194`；Conformal `alpha=0.10` 的 singleton coverage 为 `75.7%`、Selective Macro-F1 为 `0.670`、risk 为 `0.089`。

Conformal `alpha=0.10` 在 extrapolation test 上的 empirical set coverage 仅为 `68.9%`，因此不能宣称达到 `90%` 统计覆盖保证。Distance OOD proxy 的 AUROC 为 `0.822`，但 standalone distance reject 会显著降低 coverage；建议的研究型策略为：`family classifier -> calibrated probability -> conformal alpha=.10 -> distance OOD warning -> accepted / Unknown / Requires review`。这些结果仍是 research prototype 证据，不适用于 safety-critical deployment；family-level 能力受 severity OOD 影响，subtype OOD 尚未解决。

审计产物：`notebooks/16_ood_aware_selective_diagnosis.ipynb`、`src/ood_selective.py`、`results/16_*` 和 `results/figures/16_*`。16 阶段作为独立里程碑提交并推送到 `origin/main`。

## 第三阶段第七步：NPP-Guard v1 integration（17.2 Protocol Alignment）

17 阶段建立了固定 artifact 的统一研究型推理 API：`diagnose_csv(path)` 和 `diagnose_dataframe(df)`，以及 CLI：`python -m src.inference.cli --input <csv>`。输入必须包含 `TIME` 与 38 个严格过程变量，并通过时间单调性、约 10 s 采样、至少 120 s 窗口、重复时间点和 NaN/Inf 校验；缺列或窗口不足时只返回 `invalid_input`，不强制预测。

v1 policy 固定为：`family classifier -> calibrated probability -> conformal alpha=.10 -> distance OOD check`。conformal 空集返回 `unknown`；单一 family 且距离正常返回 `accepted`；单一 family 但 distance OOD 或多个 family 返回 `requires_review`；Tier B 强制 `requires_review`，Tier C 为 `unknown/not-supported`，不输出强制 12-class subtype。模型、概率校准器、特征提取 schema、conformal artifact、Ledoit-Wolf distance model、能力门控和 policy 均保存在 `artifacts/v1/`，推理阶段不重新训练。

17.2 固定了永久隔离的 trajectory/sample_id release protocol：沿用 `results/13_split_inventory.csv` 中 `random/seed_20260921/test` 的 101 条轨迹作为 `artifacts/v1/release_split_manifest.json` 的 locked release test；剩余轨迹按 accident class 和固定种子拆成 development-train `241`、development-validation `81`、calibration `82`。family classifier 只在 development-train 拟合，probability/conformal/distance threshold 只在 calibration 拟合，distance reference 只在 development-train 拟合；release test 与所有 fit/calibration 集合交集为 `0`。完整 sample_id、来源、集合 hash 和两两交集见 `results/17_v1_overlap_audit.csv/json`。

LOCA family 继续使用 120 s、排除 `LVPZ/P/TSAT/VOL` 的 Random Forest severity estimator，并将 LOCA train `37` / validation `12` / release `15` 条轨迹显式分离；当前 validation MAE `0.966`、RMSE `1.114`、R² `0.997`，仍标记为 exploratory。只有 `status=accepted`、family 为 `primary_coolant_boundary_break` 且 capability 为 Tier A 时运行 severity；`unknown`、`requires_review`、非 LOCA、Tier B/C 和 `invalid_input` 均返回 `not_run_due_to_family_gating`。08 protection-time 没有建立方法学正确的 severity 链式输入，因此 v1 返回 `not_available_in_v1`。17.2 仍是 research prototype，不适用于 safety-critical deployment；subtype OOD 和 conformal 的统计保证仍未解决。

locked release benchmark（101 条轨迹）记录为：forced family Accuracy/Macro-F1/Balanced Accuracy `1.000/1.000/1.000`；selective accepted coverage `0.515`、Selective Accuracy `1.000`、Macro-F1 `0.500`、Balanced Accuracy `1.000`、risk `0.000`、reject rate `0.485`；conformal empirical set coverage `0.901`、平均 set size `0.901`；distance OOD warning rate `0.347`。这些只描述当前锁定 cohort，不是部署泛化承诺。

16 的 severity-extrapolation 结果仍作为独立 research stress benchmark 引用（conformal coverage `0.7573`、Selective Macro-F1 `0.6700`、risk `0.0886`），与 locked release benchmark 使用不同训练/测试协议，不能做严格数值 parity。17.2 的 exact pipeline parity 改为同一 locked cohort 上 batch evaluator 与 `diagnose_dataframe()` 的逐样本比较：离散决策一致率 `101/101 = 100%`，浮点最大绝对差 `0`；不再以“贴近16指标”作为 Gate。

17.2 产物包括：`notebooks/17_npp_guard_v1_integration.ipynb`、`src/inference/`、`examples/run_v1_inference.py`、`artifacts/v1/release_split_manifest.json`、`artifacts/v1/manifest.json`、`results/17_v1_overlap_audit.csv/json`、`results/17_v1_pipeline_parity.csv/json`、`results/17_v1_release_benchmark.csv/json`、`results/17_v1_example_outputs.json` 和 `results/17_v1_regression_tests.csv`。当前五类行为示例均观察到：accepted Tier-A、accepted LOCA、requires_review、unknown、invalid_input；回归测试 `18/18` 通过。`results/17_v1_artifact_parity.csv/json` 保留为 17.1 历史诊断，不再是 release gate。

## 第三阶段第八步：Explainability（18）

18 阶段在不修改冻结 v1.2 推理逻辑的前提下，为已有诊断结果增加模型归因与诊断辅助解释。解释层只读取 `artifacts/v1/` 和 locked release cohort，不重新训练 classifier、calibrator、conformal、distance 或 severity 模型。

解释方法固定为：family 全局 `permutation importance`；family 局部的冻结训练中心替换 attribution；OOD 的 Ledoit–Wolf 距离对角近似；LOCA severity 的冻结 Random Forest feature perturbation。locked release test 上的全局变量 Top-5 为 `P / WSTA / TAVG / WFWA / LSGA`。

可复现示例包括：`FLB_10` 为 accepted；`LOCAC_12` 为 requires_review，主要 OOD driver 为 `P / TSAT / LVPZ / VOL`；`LR_93` 为 unknown；`LOCAC_19` 为 accepted LOCA，exploratory severity estimate 约为 `12.95%`。severity frozen validation MAE 为 `0.9655`，历史 sensitivity test MAE 为 `0.97247`，后者明确标记为 exploratory。

解释验证记录为：重复解释 deterministic；扰动 top-5 variable Jaccard 为 `0.9722`；family faithfulness 的 top-k 相对随机移除差为 `+0.7532`，severity 为 `+0.5732`。18 同时确认 17.2 frozen inference 未变化：artifact SHA-256 `10/10`，API/batch parity `101/101 = 100%` 且最大浮点差 `0`，regression `18/18`。

18 的解释是模型归因与诊断辅助，不是物理因果证明；LOCA severity 仍是仿真数据上的 exploratory 估计。NPP-Guard 仍为 research prototype，不适用于真实核电站运行控制或 safety-critical deployment。

18 产物包括：`notebooks/18_explainability.ipynb`、`src/explainability/`、`results/18_summary.json`、`results/18_local_explanations.json`、`results/18_global_*.csv`、`results/18_ood_explanations.csv`、`results/18_severity_explanations.csv`、`results/18_explanation_*.csv` 和 `results/figures/18_*`。

## 当前结果摘要

以下数字来自仓库中已保存的 `results/*.json`，是当前数据和当前规则下的实验记录，不是泛化性能承诺。

| 实验 | 当前记录 |
| --- | --- |
| LOCA/SLBIC/FLB 早期分类 | 100 s 截止窗口；五折平均普通准确率 `75.0%`；平均置信度覆盖率 `41.7%`；被接受样本的准确率 `100%`，接受样本错误数 `0` |
| LOCA vs SLBIC 早期验证 | 50 s 和 100 s 截止窗口的五折平均准确率及 balanced accuracy 均为 `100%`；150 s 时均为约 `93%` |
| 全功率异常门 | LOCA 100/100 个案例报警，中位首次报警时间 `130 s`；FLB 100/100，中位 `180 s`；SLBIC 101/101，中位 `200 s` |
| 变功率条件化异常门 | LOCA 在 `80%→100%` 和 `100%→80%` 场景分别记录约 `141.5 s` 和 `145 s` 的检测延迟；其他当前变功率故障在该规则下未触发报警 |

这些结果也暴露出当前范围的局限：现有规则对 RW、SGATR、SGBTR 等场景没有产生有效报警，分类覆盖率较低，且结果依赖 NPPAD 仿真设置、参考轨迹和阈值。100 s 是算法设定的早期截止窗口，不等同于经过安全论证的保护裕度；保护动作时间审计也不意味着每个案例都严格早于保护动作。请优先阅读对应 JSON 报告中的完整 case 级结果。

## 项目结构

```text
NPP-Guard/
├─ src/
│  ├─ validate_npp_guard.py                 # 核心标准化分数、报警门和短期基线
│  ├─ validate_loca_slbic.py                # LOCA/SLBIC 早期验证
│  ├─ validate_gated_fault_classifier.py    # LOCA/SLBIC/FLB 保守分类
│  ├─ validate_variable_power_gate.py       # 变功率条件化异常门
│  ├─ validate_variable_power_classifier.py # 跨功率方向实验性分类
│  ├─ train_full_power_model.py             # 训练并保存全功率模型
│  ├─ npp_guard_inference.py                # 统一推理入口
│  ├─ final_validation_summary.py            # 汇总全功率和变功率结果
│  ├─ normal_reference.py                    # Normal reference 全量审计
│  ├─ severity_features.py                   # LOCA 严重度窗口特征
│  ├─ severity_models.py                     # 分组严重度回归 baseline
│  ├─ protection_time_features.py            # landmark 保护时间特征
│  ├─ protection_time_models.py              # 分组保护时间回归 baseline
│  ├─ protection_time_report.py              # 08 结果汇总与敏感性审计
│  ├─ multi_accident.py                       # 09 多事故数据集审计
│  ├─ temporal_diagnosability.py              # 11 时间可诊断性分析
│  ├─ event_parser.py                          # 13 通用 protection-event taxonomy
│  ├─ ood_validation.py                        # 13 severity OOD 验证
│  ├─ severity_invariant.py                    # 14 severity-invariant 特征实验
│  ├─ ood_error_analysis.py                    # 14 方向性 OOD error decomposition
│  ├─ hierarchical_diagnosis.py                # 14 family mapping 与两阶段诊断
│  ├─ milestone14.py                            # 14 Notebook 编排与结果汇总
│  ├─ coverage_task_redesign.py                 # 15 coverage-aware family selective diagnosis
│  ├─ ood_selective.py                           # 16 OOD-aware selective family diagnosis
│  └─ inference/                                 # 17 fixed-artifact v1 inference API and CLI
├─ notebooks/
│  ├─ 01_LOCA_EDA.ipynb
│  ├─ 02_LOCA_severity_analysis.ipynb
│  ├─ 03_ML_dataset_audit.ipynb
│  ├─ 04_early_anomaly_detection.ipynb
│  ├─ 05_dynamic_early_warning.ipynb
│  ├─ 06_normal_reference_expansion.ipynb
│  ├─ 07_LOCA_severity_estimation.ipynb
│  ├─ 08_protection_time_prediction.ipynb
│  ├─ 09_multi_accident_dataset_audit.ipynb
│  ├─ 10_multi_accident_classification.ipynb
│  ├─ 11_temporal_diagnosability_analysis.ipynb
│  ├─ 12_robust_validation.ipynb
│  ├─ 13_ood_severity_blocked_validation.ipynb
│  ├─ 14_severity_invariant_hierarchical_diagnosis.ipynb
│  ├─ 15_coverage_aware_task_redesign.ipynb
│  ├─ 16_ood_aware_selective_diagnosis.ipynb
│  └─ 17_npp_guard_v1_integration.ipynb
├─ artifacts/v1/                                # fixed v1 research artifacts and manifest
├─ models/
│  ├─ npp_guard_full_power_early.joblib
│  └─ npp_guard_full_power_early.json
├─ results/                                 # 已保存的验证报告、轻量 CSV 和 figures/
│  ├─ early_anomaly_detection_*.csv         # 04 静态 baseline
│  ├─ dynamic_early_warning_*.csv           # 05 动态特征实验
│  ├─ normal_reference_*.csv/json            # 06 Normal reference 审计
│  ├─ 10_multi_accident_*.csv/json          # 10 多事故分类结果
│  ├─ 11_temporal_*.csv/json                # 11 时间可诊断性结果
│  ├─ 12_robust_*.csv/json                  # 12 稳健性验证结果
│  ├─ 13_ood_*.csv/json                      # 13 OOD 验证结果
│  ├─ 14_*.csv/json                          # 14 severity-invariant/hierarchical 结果
│  ├─ 15_*.csv/json                          # 15 coverage-aware/selective 结果
│  ├─ 16_*.csv/json                          # 16 OOD-aware/selective 结果
│  └─ figures/11_*.png, 12_*.png, 13_*.png, 14_*.png, 15_*.png, 16_*.png  # 11–16 分析图
├─ data/                                    # 本地数据目录，不提交到本仓库
├─ requirements.txt
└─ README.md
```

## 环境与依赖

推荐使用 Python 3.12 和项目虚拟环境。Windows 下运行变功率 `.mdb` 数据还需要安装能提供以下驱动的 Microsoft Access ODBC 驱动：

```text
Microsoft Access Driver (*.mdb, *.accdb)
```

创建环境并安装依赖：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果 PowerShell 阻止激活脚本，也可以直接调用虚拟环境解释器：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 数据准备

NPPAD 原始数据体积较大，不包含在本仓库提交中。代码默认从下面的目录读取：

```text
data/NuclearPowerPlantAccidentData/
├─ Operation_csv_data/
│  ├─ Normal/1.csv
│  ├─ LOCA/*.csv
│  ├─ LOCAC/*.csv
│  ├─ FLB/*.csv
│  └─ SLBIC/*.csv
└─ Variable_Power_Data/
   ├─ manifest.csv
   └─ */case1.mdb
```

获取数据的一种方式是把公开数据仓库克隆到代码约定的路径：

```powershell
git clone https://github.com/thu-inet/NuclearPowerPlantAccidentData.git data/NuclearPowerPlantAccidentData
```

当前工作区已有一份本地 NPPAD 数据，但其约 17.5 GB 的原始文件、嵌套 Git 元数据和其他仿真资源均由根目录 `.gitignore` 排除，不会随 NPP-Guard 一起推送。

## 运行方法

请在仓库根目录运行。验证脚本会读取 `data/`，并将对应 JSON 报告写入 `results/`；训练脚本会更新 `models/` 下的模型文件。

```powershell
# LOCA/SLBIC 早期验证
python src/validate_loca_slbic.py

# LOCA/SLBIC/FLB 保守分类验证
python src/validate_gated_fault_classifier.py

# 变功率条件化异常门（需要 Access ODBC 驱动）
python src/validate_variable_power_gate.py

# 变功率跨方向分类实验
python src/validate_variable_power_classifier.py

# 训练全功率早期模型
python src/train_full_power_model.py

# 运行统一推理示例
python src/npp_guard_inference.py

# 汇总当前验证范围
python src/final_validation_summary.py
```

Notebook 可以用 Jupyter 打开：

```powershell
jupyter notebook notebooks/01_LOCA_EDA.ipynb
jupyter notebook notebooks/02_LOCA_severity_analysis.ipynb
jupyter notebook notebooks/03_ML_dataset_audit.ipynb
jupyter notebook notebooks/04_early_anomaly_detection.ipynb
jupyter notebook notebooks/05_dynamic_early_warning.ipynb
jupyter notebook notebooks/06_normal_reference_expansion.ipynb
jupyter notebook notebooks/07_LOCA_severity_estimation.ipynb
jupyter notebook notebooks/08_protection_time_prediction.ipynb
jupyter notebook notebooks/09_multi_accident_dataset_audit.ipynb
jupyter notebook notebooks/10_multi_accident_classification.ipynb
jupyter notebook notebooks/11_temporal_diagnosability_analysis.ipynb
jupyter notebook notebooks/12_robust_validation.ipynb
jupyter notebook notebooks/13_ood_severity_blocked_validation.ipynb
```

请从仓库根目录启动 Jupyter。04/05 Notebook 会从当前目录及其父目录探测项目根目录，并读取本地 `data/`；`src/` 中的脚本也使用相对于项目根目录的路径推导。

## Roadmap

- 06 Normal-reference audit 已完成：A 类 1 条 `Normal/1.csv`，B 类 4 条变功率 Normal MDB，事故数据无法提供合法事故前窗口；
- 07 LOCA severity estimation 已完成：比较 30/60/120/300 s 窗口和三类回归模型，并完成过程量—严重度耦合敏感性审计；
- 08 protection-time prediction 已完成：严格最佳点为 60 s Random Forest，Test `MAE=86.826 s`、`RMSE=338.735 s`、`R²=0.330`；30/60 s 的 process-only 未超过 severity-only，300 s 样本过少；
- 09 multi-accident dataset audit 已完成：17 类、1216 条轨迹；12 类进入首版分类，5 个单样本类别暂缓；30/60 s 全覆盖、120 s 覆盖率 98.52%；已固定 38 个严格过程变量、SLBIC schema 差异、16 个潜在 leakage 组合和 SLBIC 初始工况混杂审计；
- 10 multi-accident classification 已完成：按完整 trajectory/sample_id 分组，比较 12 类、38 个严格过程变量以及 leakage/SLBIC 初始工况敏感性；
- 11 temporal diagnosability analysis 已完成：all-sample 的 120 s Macro-F1 为 `0.749`，strict pre-protection-only 的固定 12 类 Macro-F1 为 `0.647`、Observed-class Macro-F1 为 `0.971`，但 strict 集只有 8 个有支持类别且部分 test support 很小；
- `12_robust_validation` 已完成：471 条 strict matched trajectories、10 个固定 seed、共用 matched cohort/split；120 s Fixed-12 Macro-F1 `0.578±0.004`、Observed-class Macro-F1 `0.991±0.007`、Balanced Accuracy `0.989±0.011`，但 RW 仅 3 条且 test support 为 0，LLB/LR/MD/SLBOC 无 `first_protection_time`，暂不进入深度学习；
- `13_ood_severity_blocked_validation` 已完成：505 条 strict matched trajectories；parser recovery 新增 LR 30 条、RI 6 条 protection event；120 s blocked Macro-F1 `0.593±0.064`、extrapolation `0.483±0.110`，nearest-severity gap 明显扩大，但 extrapolation 与稀疏类别 Recall 仍不足，因此暂不进入 GRU/LSTM/TCN；
- `14_severity_invariant_hierarchical_diagnosis` 已完成：12-class 120 s extrapolation Macro-F1 `0.4826`、Balanced Accuracy `0.8125`；severity-invariant 特征仅升至 `0.4905`；family-level Macro-F1 `0.6331`，但两阶段 subtype Macro-F1 `0.4829`；RI/LOCAC/SLBIC 存在方向性不稳定，因此暂不进入深度学习，先进行 coverage/task redesign；
- `15_coverage_aware_task_redesign` 已完成：class Tier A/B/C `8/1/8`、family Tier A/B/C `4/2/4`；120 s family-level Macro-F1 在 random / blocked / extrapolation 下约为 `0.8469/0.7603/0.6660`；现有 selective reject 仅 `23/1525=1.51%`，blocked/extrapolation 验证阈值为 `0`，无法有效拒绝大 severity-gap OOD，因此 family diagnosis 仍为有限能力，subtype OOD 与 reject 机制不足，暂不进入深度学习；
- `16_ood_aware_selective_diagnosis` 已完成：120 s extrapolation forced family Macro-F1 `0.666`、risk `0.194`；Conformal `alpha=0.10` coverage `75.7%`、Selective Macro-F1 `0.670`、risk `0.089`，但 empirical set coverage 仅 `68.9%`，不能宣称 `90%` 统计覆盖保证；distance OOD proxy AUROC `0.822`。推荐研究型 Unknown/Requires review 策略，仍不适用于 safety-critical deployment；
- `17_npp_guard_v1_integration` 的 17.2 Protocol Alignment 已完成：固定 release split、zero-overlap audit、artifact SHA-256、exact batch/API parity、LOCA gating、locked release benchmark 和 `18/18` regression 全部通过；16 的 severity-extrapolation 结果仅作为独立 research stress benchmark，不能与 release benchmark 做严格数值 parity。保护时间暂不接入 v1，系统仍明确标记 research prototype、not for safety-critical deployment、severity OOD/subtype OOD 与 conformal 实际 coverage 限制；
- 如果后续获得独立且匹配的固定功率 Normal reference，再恢复更严格的 normal-reference early warning 研究；
- 只有在 Normal reference 和传统基线稳定后，再评估更严格的按场景/工况隔离、更多早期预警指标和时序深度学习模型；
- 继续保留当前变功率分类和未触发异常门场景的独立验证，不把实验性结果接入统一推理；
- 最后再考虑面向研究展示的可视化界面。

Roadmap 中的项目尚未完成，不代表当前版本已经具备对应能力。

## 数据来源与引用

本项目使用的 NPPAD 数据由公开仓库提供：

- 数据仓库：[thu-inet/NuclearPowerPlantAccidentData](https://github.com/thu-inet/NuclearPowerPlantAccidentData)
- 数据论文：Qi, B., Xiao, X., Liang, J. et al. *An open time-series simulated dataset covering various accidents for nuclear power plants*. **Scientific Data 9**, 766 (2022). [https://doi.org/10.1038/s41597-022-01879-1](https://doi.org/10.1038/s41597-022-01879-1)
- NPPAD 数据仓库声明采用 MIT License；请同时遵守原始数据仓库的许可证和引用要求。

## 免责声明

NPP-Guard 只用于仿真数据分析、算法研究和教学演示。它没有经过真实核电厂数据、核安全法规要求、硬件在环测试或运行许可验证，不提供安全论证，也不能替代合格人员、保护系统、运行规程或监管审查。任何实验结果都不应直接用于真实机组的报警、控制或事故处置决策。

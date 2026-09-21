# NPP-Guard 简历与面试材料

## 中文简历 bullets

- 面向 NPPAD 核电仿真时序数据构建 NPP-Guard 事故诊断研究原型，统一实现 120 s 过程特征、family diagnosis、LOCA severity gating、OOD/不确定性和 JSON 输出。
- 按完整 `trajectory/sample_id` 做分组验证，执行 train-only preprocessing、潜在标签泄漏审计以及 random / severity-blocked / extrapolation 对照，避免把时间行随机拆分造成的高分误判为泛化能力。
- 设计 conformal + distance OOD 的 selective policy，使系统能输出 `accepted / requires_review / unknown / invalid_input`，并建立 locked release split、zero-overlap audit 与 API/batch `101/101` exact parity。
- 在冻结 v1 推理核心上增加模型归因、OOD drivers 和 120 s 趋势解释，随后用 Streamlit 完成 Dashboard；验证通过 5/5 smoke、3/3 helper、HTTP 200 和 AppTest 0 exception。
- 负责将 01–19 实验结果、限制和复现入口整理为技术报告、关键结果表、竞赛材料和演示脚本，保持 research prototype 定位，不夸大为工业部署。

## English resume bullets

- Built NPP-Guard, a research prototype for accident-family diagnosis on NPPAD nuclear power plant simulation trajectories, with 120-second process features, LOCA severity gating, OOD signals and JSON export.
- Used trajectory/sample_id grouped evaluation, train-only preprocessing and explicit leakage audits across random, severity-blocked and severity-extrapolation protocols.
- Implemented a conformal-plus-distance selective policy that preserves accepted, requires-review, unknown and invalid-input outcomes instead of forcing a subtype prediction.
- Locked a frozen v1 release split with zero-overlap audits and exact API/batch parity on 101 trajectories, then added attribution and OOD-driver explanations without changing the inference core.
- Delivered a Streamlit research dashboard validated by 5/5 smoke tests, 3/3 helper tests, HTTP 200 and zero AppTest exceptions.

## 60 秒面试介绍

我做的是 NPP-Guard，一个面向核电仿真事故数据的研究型诊断原型。项目最重要的不是追求一个很高的 Accuracy，而是先处理数据泄漏和评估协议问题：所有切分按完整 trajectory/sample_id，预处理只在训练集拟合，同时区分 random、severity-blocked 和 extrapolation。最终 v1 用 120 s 过程窗口，经过校准、conformal 和距离 OOD 后输出 accepted、requires_review、unknown 或 invalid_input。18 阶段补充模型归因和 OOD drivers，19 阶段用 Streamlit 做了可演示界面。当前结果只适用于仿真研究，不等于真实核电站部署能力。

## 3 分钟项目介绍

先讲问题：事故早期窗口信息有限，仿真严重度会和过程变量耦合，随机拆分时间行还会造成虚高指标。然后讲路线：01–06 先做数据和 Normal 审计，04/05 的 early-warning 负结果促使项目转向 severity 和保护时间；07/08 做 LOCA 专项；09–14 做多事故、稳健性、severity-blocked 和 family-level 验证；15/16 引入 coverage-aware、conformal 和 distance reject；17.2 固定 release split，完成 zero overlap、artifact hash 和 101/101 parity；18/19 加入解释层和 Dashboard。最后讲边界：30/60 s 区分度弱，extrapolation 下降，保护时间未接入 v1，conformal 经验 coverage 不是保证。因此项目的成果是一个可审计、能拒答、能解释的研究型流程，而不是安全系统。

## 高频追问与回答

### 为什么不用 LSTM / Transformer？

因为数据协议和 OOD 支持还没有稳定到值得增加模型复杂度。12–14 的重复验证和 severity extrapolation 仍显示支持度与跨严重度问题，先修复任务和数据边界比堆叠时序模型更重要。

### 为什么不是 99% Accuracy？

随机协议可能给出很高结果，但 13 的 severity extrapolation fixed-12 Macro-F1 约为 0.483，说明跨严重度更难。项目优先报告 Macro-F1、Balanced Accuracy、per-class Recall、confusion matrix、coverage 和 risk，避免单一 Accuracy 掩盖少数类失败。

### 如何防止 data leakage？

按完整 trajectory/sample_id 切分；预处理只在训练集拟合；校准、conformal 和阈值使用各自允许的数据；17.2 还固定 release split 并审计 release test 与所有 fit/calibration 集的交集为零。对可疑变量保留显式 leakage flags，做敏感性实验。

### 为什么需要 Unknown？

因为模型的 top-1 不代表证据足够。当 conformal set 为空、距离明显偏离、能力 Tier 不足或输入无效时，强行输出 subtype 会把分布外样本伪装成确定答案。Unknown/Requires Review 是对不确定性的显式表达。

### 最大技术难点是什么？

不是选择模型，而是统一研究 benchmark 和 release protocol。17.1 的 parity 暴露出不同训练协议，后续用 locked split、zero-overlap 和 exact API/batch parity 重新定义了可发布的验证边界。

### 你的个人贡献是什么？

负责从实验审计到 v1 交付链路的整理与实现，包括 grouped split 和 leakage 规则、OOD / selective policy、冻结推理接口、解释层、Dashboard 以及最终报告和演示材料。面试时应根据本人真实分工调整这段表述。

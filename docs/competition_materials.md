# NPP-Guard 竞赛 / 科研申报文字

## 300 字中文摘要

NPP-Guard 面向核电厂仿真事故时序数据，构建一个带数据质量检查、不确定性评估、主动拒答和模型解释的研究型事故诊断原型。项目基于 NPPAD，先审计 Normal 数据不足、标签泄漏、保护动作前窗口和完整轨迹划分风险，再比较 LOCA 严重度、保护时间、多事故分类以及 severity-blocked / extrapolation OOD。系统不把随机拆分高分直接当作泛化能力，而是固定 120 s 特征窗口、conformal 与距离 OOD policy，输出 accepted、requires_review、unknown 和 invalid_input 四种状态。17.2 建立 locked release split，完成 train/calibration/test zero-overlap、API 与 batch 逐样本 parity，18 增加模型归因和 OOD 驱动解释，19 以 Streamlit Dashboard 展示诊断、趋势、门控和 JSON 导出。项目的贡献在于把“分类器”整理成可审计、可拒答、可解释的研究流程。当前结果只适用于仿真数据研究和教学展示，不用于真实核电站运行控制或安全关键决策。

## 项目简介

NPP-Guard 将 NPPAD 的单轨迹 CSV 转换为 120 s 诊断窗口，经过质量检查和冻结 v1 artifact 推理，给出事故族、置信度、conformal set、距离 OOD、能力 Tier 和解释。LOCA severity 只在满足 accepted、LOCA family 和 Tier A 时运行，避免把低可信度样本的辅助估计误当作主结论。

## 技术路线

`CSV/DataFrame → Data Quality → 120 s 特征 → family classifier → probability calibration → conformal → distance OOD → accepted/requires_review/unknown → LOCA severity gating → explainability → Dashboard/JSON`

## 主要创新点

1. 用完整 `trajectory/sample_id` 分组和 train-only preprocessing 降低时间行泄漏风险，并把 leakage flags 保留为敏感性实验记录。
2. 用 random、severity-blocked、severity-extrapolation 三类协议区分“随机同分布表现”和“跨严重度压力测试”。
3. 让系统在证据不足或分布偏移时输出 `Unknown` / `Requires Review`，而不是强制 12 类 subtype。
4. 用冻结 artifact、固定 release split、zero-overlap audit 和 API/batch exact parity 形成可复现 release protocol。
5. 将 family attribution、OOD drivers、120 s trend 和 LOCA gating 接入本地 Dashboard，便于科研汇报和复核。

## 应用价值

项目适合作为核工程数据分析、机器学习方法学审计、事故诊断原型和本科生创新项目的研究底座。它展示了如何把数据边界、拒答策略、解释层和产品化界面纳入同一个可复现实验流程。

## 局限性

- NPPAD 是仿真数据，不能替代独立机组数据或安全分析流程。
- Normal reference 不足，早期预警结论不充分。
- severity extrapolation 下性能下降，部分类别支持度很低。
- conformal 经验覆盖率不是统计保证；subtype OOD 未解决。
- protection-time 尚未形成合法的 v1 链式推理。
- 解释是模型归因，不是物理因果证明。

## 后续计划

补充固定功率 Normal reference，扩大跨工况和跨严重度测试，建立独立外部测试集，完善 subtype OOD 与 uncertainty calibration，并在数据支持后再评估时序模型。

## 节能减排 / 大学生创新类竞赛亮点版

面向核电事故诊断中的“高分不等于可靠”问题，NPP-Guard 把数据审计、泄漏控制、跨严重度验证、主动拒答、解释性和 Dashboard 集成为一个可复现原型。项目强调少报错、能拒答、可追溯，服务于核工程仿真研究和智能运维方法验证。它不宣称替代现有安全系统，而是为后续高质量数据采集、异常工况研究和辅助分析工具提供可检查的技术路线。

# NPP-Guard v1 Release Checklist

## 通过项

- [x] 19 Dashboard 已独立提交并 push：`bf6e0dcc20cbad31fa9535710f29f2edef86f973`
- [x] `HEAD == origin/main` 在 19 提交后确认
- [x] Streamlit 启动命令与 `requirements.txt` 一致
- [x] Dashboard 使用冻结 `artifacts/v1/`，不在页面启动时训练
- [x] 19 smoke `5/5`，helper `3/3`
- [x] Streamlit HTTP `200`
- [x] AppTest `0` 个运行异常
- [x] 17.2 exact API/batch parity `101/101`，最大浮点差 `0`
- [x] 17.2 regression `18/18`
- [x] 18 explainability gate：artifact SHA-256、核心诊断未变化、parity/regression 通过
- [x] 20 key results 同步输出 Markdown 与 CSV
- [x] 架构图与研究路线图保存为可维护 Mermaid 源文件
- [x] README 包含 Dashboard、CLI/API、能力矩阵、限制和文档入口
- [x] 原始 NPPAD 数据、`.venv`、artifacts 缓存和临时目录未被加入版本控制
- [x] 未发现本次新增的大型或敏感文件

## 已知限制

- [ ] 不能宣称真实核电站部署或安全保证
- [ ] Normal reference 不足，early-warning 仍不充分
- [ ] severity extrapolation 性能下降，部分类 support 稀疏
- [ ] conformal empirical coverage 不等于 90% 统计保证
- [ ] subtype OOD 尚未解决
- [ ] protection-time 尚未形成有效 v1 链式输入
- [ ] explainability 是模型归因，不是物理因果
- [x] 20 最终交付包已完成审阅，演示 PDF 已生成，准备纳入最终 release commit/tag

验证明细见 [`results/20_release_validation_summary.json`](../results/20_release_validation_summary.json)。

# NPP-Guard v1 Dashboard 3–5 分钟演示脚本

## 0. 开场（15 秒）

说明：这是 NPPAD 仿真数据上的 research prototype。它展示诊断、拒答、解释和门控，不连接真实机组，不用于安全关键决策。

## 1. 启动 Dashboard（20 秒）

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

浏览器打开本地 Streamlit 页面，先展示输入要求：`TIME`、38 个 strict process variables、至少 120 s、约 10 s 采样、无 NaN/Inf 和重复时间点。

## 2. accepted Tier-A 样本（35 秒）

选择或上传真实文件：`data/NuclearPowerPlantAccidentData/Operation_csv_data/FLB/10.csv`。

预期：状态 `accepted`，family 为 `feedwater_line_break`，出现 conformal / OOD / confidence 信息和 120 s trend。说明 accepted 只是冻结研究 policy 接受该样本，不是安全认证。

## 3. accepted LOCA 与 severity gating（40 秒）

加载：`data/NuclearPowerPlantAccidentData/Operation_csv_data/LOCAC/19.csv`。

预期：状态 `accepted`，family 为 LOCA，且在 Tier A 时显示 exploratory LOCA severity。强调 gating 条件是 `accepted + LOCA family + Tier A`，其余状态不运行 severity。

## 4. requires_review / OOD（35 秒）

加载：`data/NuclearPowerPlantAccidentData/Operation_csv_data/LOCAC/12.csv`。

预期：状态 `requires_review`，展示 OOD warning、distance 信息和 OOD drivers。强调系统保留人工复核入口，不强制给出“已确认”的诊断。

## 5. unknown（30 秒）

加载：`data/NuclearPowerPlantAccidentData/Operation_csv_data/LR/93.csv`。

预期：状态 `unknown` 或对应仓库 smoke fixture 的 unknown 结果，展示 conformal / uncertainty 说明。解释为：当前样本与校准数据不够一致，系统没有足够证据输出唯一可靠 family。

## 6. invalid_input（25 秒）

将 `FLB/10.csv` 截断为 `TIME <= 50` 的文件再上传，或复现 `dashboard/smoke_tests.py` 的 controlled truncation。

预期：状态 `invalid_input`，原因包含 `window_shorter_than_120_s`，预测和 severity 都不运行。

## 7. JSON export（20 秒）

点击导出，展示 JSON 中的 input metadata、frozen diagnosis、explanation 和 limitations。指出导出不包含原始全量 CSV，便于传递审计结果而不复制原始数据。

## 8. 收尾（15 秒）

总结：NPP-Guard 的重点是让系统在数据不足或分布偏移时明确拒答，并把模型依据和局限展示出来。再次说明：这是仿真数据上的 research prototype，不用于真实核电站运行控制或 safety-critical decision。

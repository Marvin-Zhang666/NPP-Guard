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
│  └─ final_validation_summary.py            # 汇总全功率和变功率结果
├─ notebooks/
│  └─ 01_LOCA_EDA.ipynb
├─ models/
│  ├─ npp_guard_full_power_early.joblib
│  └─ npp_guard_full_power_early.json
├─ results/                                 # 已保存的验证报告
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
```

Notebook 中保留了早期探索过程，部分单元格使用了本机绝对路径 `C:\Users\18205\NPP-Guard\data\...`。在其他电脑上运行这些单元格前，请把路径改为本机路径；`src/` 中的脚本使用相对于项目根目录的路径推导，推荐优先使用脚本重现实验。

## Roadmap

- 批量整理不同 LOCA 严重度，并研究严重度与检测时间的关系；
- 建立严格的按场景、按工况隔离的训练/验证/测试划分；
- 补充 false alarm rate、lead time、召回率和置信度校准等早期预警指标；
- 扩充并验证更多事故类型，尤其是当前规则未能有效报警的场景；
- 完成变功率分类的独立工况验证，再考虑将其接入统一推理；
- 在传统机器学习基线稳定后，再评估时序深度学习模型和可解释性方法；
- 最后再考虑面向研究展示的可视化界面。

Roadmap 中的项目尚未完成，不代表当前版本已经具备对应能力。

## 数据来源与引用

本项目使用的 NPPAD 数据由公开仓库提供：

- 数据仓库：[thu-inet/NuclearPowerPlantAccidentData](https://github.com/thu-inet/NuclearPowerPlantAccidentData)
- 数据论文：Qi, B., Xiao, X., Liang, J. et al. *An open time-series simulated dataset covering various accidents for nuclear power plants*. **Scientific Data 9**, 766 (2022). [https://doi.org/10.1038/s41597-022-01879-1](https://doi.org/10.1038/s41597-022-01879-1)
- NPPAD 数据仓库声明采用 MIT License；请同时遵守原始数据仓库的许可证和引用要求。

## 免责声明

NPP-Guard 只用于仿真数据分析、算法研究和教学演示。它没有经过真实核电厂数据、核安全法规要求、硬件在环测试或运行许可验证，不提供安全论证，也不能替代合格人员、保护系统、运行规程或监管审查。任何实验结果都不应直接用于真实机组的报警、控制或事故处置决策。

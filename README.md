# fashion-mnist-research

## 项目简介

这是一个基于 FashionMNIST 数据集的小型机器学习实验项目。

本项目通过受控实验比较 Softmax Regression 与多层感知机（MLP），研究两类模型的分类表现、训练数据量的影响，以及错误训练标签下的模型行为。

## 研究目标 

我希望逐步通过实验回答以下问题：

1. **类别与错误分析**：Softmax Regression 与 MLP 在哪些类别的分类表现上存在差异？它们各自容易混淆哪些类别？
2. **训练数据量的影响**：随着训练数据量增加，MLP 与 Softmax Regression 的分类表现差异如何变化？
3. **错误标签与记忆现象**：当部分训练标签被人为改错时，MLP 与 Softmax Regression 对这些错误标签的拟合行为有何差异？这些行为与它们在干净评估数据上的表现有何关系？

这些问题不预设 MLP 一定优于 Softmax Regression。具体评价指标、控制变量与实验步骤将在开展对应实验前确定。

## 推进计划

- 首先实现并理解 Softmax Regression 和 MLP 的基本训练与评估流程，建立可比较的 baseline。
- 随后分析两个模型在不同类别和分类错误上的差异。
- 在此基础上，设计改变训练数据量的受控实验。
- 在此基础上，探索错误训练标签下的拟合行为，并与干净评估数据上的表现进行比较。

每项正式实验开始前，先明确研究问题、假设、自变量、控制变量、评价指标与实验步骤；实验结束后，再区分实际观察、可能解释与研究局限。

## 学习目标

- 使用 PyTorch 亲自完成数据加载、模型实现、训练与评估。
- 学习设计受控实验，识别可能影响结论的混杂因素。
- 根据实验结果进行分析，避免将猜测直接当作结论。
- 逐步学习使用 Git 管理版本，并记录复现实验所需的配置、环境与结果。

## 当前进度

已实现 Softmax Regression 和 MLP 的训练与验证流程，并完成以下实验：

- **类别与错误分析**：比较逐类别召回率和混淆矩阵。
- **初始化敏感性实验**：固定数据划分和训练顺序，比较五个初始化 seed。
- **训练数据量实验**：比较 6,000、18,000、54,000 个训练样本，在固定 5,000 次参数更新下完成 30 组运行，并保存日志和图表。
- **错误标签初步实验与初始化重复**：使用 18,000 个训练样本、初始化 seed 42～46 和固定 5,000 次参数更新，比较两种模型在 0% 与 20% 错误训练标签下的表现，共完成 20 组运行；保存原始日志、数据审计、汇总统计和图表。

分类表现基于验证集；错误标签拟合率和对应的原始标签准确率来自参与训练的被改标签子集。官方测试集尚未评估。错误标签实验覆盖五个初始化，但仍固定同一份标签噪声和训练终点，尚不能确定记忆机制或推广到其他噪声设置。

实验记录：

- [初始基线](baseline.md)
- [模型与分类错误比较](comparison.md)
- [初始化敏感性实验](initialization.md)
- [训练数据量实验](data_size.md)
- [错误训练标签初步实验](label_noise.md)

## 主要发现

以下结果均使用同一份 6,000 样本的干净验证集，并对初始化 seed 42～46 的五次结果取平均。差值由原始计数计算，单位为百分点；由于四舍五入，可能与表中两个百分比直接相减略有不同。类别分析使用 **3 个 epoch**，后两项实验使用 **5,000 次参数更新**，两种训练设置的结果应分别解读。

### 1. 类别与错误分析

在 54,000 个训练样本、3 个 epoch 的设置下，Softmax 的平均验证准确率为 **80.13%**，MLP 为 **81.20%**，MLP 平均高 **1.06 个百分点**。

总体准确率较高，并不表示所有类别都更好。MLP 的 Coat、Shirt 平均召回率分别高 16.71、9.07 个百分点，而 T-shirt/top、Pullover 分别低 5.93、9.05 个百分点；这些差异的方向在五个初始化中一致。混淆矩阵中，MLP 将 T-shirt/top 判成 Shirt 的数量更多，将 Coat 判成 Pullover 的数量更少，这两个方向也在五次运行中一致。

完整结果见 [初始化敏感性实验](initialization.md)，单次运行的错误分析见 [模型与分类错误比较](comparison.md)。

### 2. 训练数据量的影响

在干净训练标签、固定 5,000 次参数更新的设置下：

| 训练样本数 | Softmax 平均验证准确率 | MLP 平均验证准确率 | 平均差值（MLP − Softmax） |
| --- | --- | --- | --- |
| 6,000 | 83.60% | 83.41% | −0.19 个百分点 |
| 18,000 | 82.95% | 85.77% | +2.82 个百分点 |
| 54,000 | 83.80% | 85.59% | +1.79 个百分点 |

两个模型的平均准确率及其差距都没有随样本数单调增加。6,000 样本组中，MLP 有两次高于 Softmax、三次低于 Softmax；其余两组中，MLP 五次均更高。这只描述本次固定更新预算下的结果，不能据此认定 18,000 是最佳数据量，或增加数据会降低表现。

完整结果、标准差与子集核对见 [训练数据量实验](data_size.md)。

### 3. 错误标签与拟合行为

在 18,000 个训练样本、固定 5,000 次参数更新的设置下，五个初始化共用同一份错误标签：

| 模型 | 0% 错误标签时的平均验证准确率 | 20% 错误标签时的平均验证准确率 | 平均下降幅度 |
| --- | --- | --- | --- |
| Softmax Regression | 82.95% | 72.77% | 10.18 个百分点 |
| MLP | 85.77% | 84.27% | 1.50 个百分点 |

在被改标签的 3,600 个训练样本上，Softmax 与 MLP 对错误标签的平均拟合率分别为 **3.22%**、**2.34%**，对对应原始标签的平均准确率分别为 **73.11%**、**84.34%**。MLP 的验证准确率更高、加入噪声后的下降幅度更小、错误标签拟合率更低、原始标签准确率更高，这四个方向在五个配对初始化中一致。

这里的拟合率表示训练结束时预测与错误标签一致的比例，不能直接证明模型的记忆机制。实验只覆盖一份标签噪声和一个训练终点，尚未比较其他噪声样本或更长训练过程。完整记录见 [错误训练标签实验](label_noise.md)。

以上五次重复只改变模型初始化，未覆盖不同数据划分；官方测试集也尚未使用。因此，这些发现适用于已记录的实验设置，尚不能推广为模型的一般规律。

## 如何运行

当前环境已在 Windows 和 NVIDIA GPU 上验证。

在项目文件夹中打开 PowerShell，依次执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

首次运行会自动下载 FashionMNIST 数据集。

模型、初始化 seed 和错误标签比例可以通过命令参数设置：

| 参数 | 可选值或含义 | 默认值 |
| --- | --- | --- |
| `--model` | `softmax` 或 `mlp` | `mlp` |
| `--init-seed` | 模型初始化使用的整数 seed | `42` |
| `--noise-rate` | `0.0` 或 `0.2` | `0.2` |

例如，运行初始化 seed 43、20% 错误标签的 MLP：

```powershell
.\.venv\Scripts\python.exe train.py --model mlp --init-seed 43 --noise-rate 0.2
```

训练样本数仍通过 `train.py` 中的 `train_size` 设置；数据划分、子集选择、训练打乱与标签噪声使用各自固定的 seed。当前默认使用 18,000 个训练样本和 5,000 次参数更新。

`--noise-rate 0.0` 时没有被改标签样本，两项噪声评估指标显示为 `N/A`。

## 实验版本与复现

直接运行当前 `train.py` 只执行一组错误标签实验，不会自动重现全部表格。历史实验使用不同训练程序，应从对应 Git 版本获取源码；下表的链接固定到归档版本，不随当前文件变化。

| 实验记录 | 归档代码版本 | 复现时需要选择的配置 |
| --- | --- | --- |
| [初始 baseline](baseline.md) | [1187e7f](https://github.com/peiyangjiang/fashion-mnist-research/blob/1187e7fb314d73e719f51fe83137d3192dcd4a34/train.py) | Softmax，54,000 样本，seed 42，3 个 epoch；保留当时的数据预览 |
| [模型与分类错误比较](comparison.md) | [7e8931f](https://github.com/peiyangjiang/fashion-mnist-research/blob/7e8931f46304c62817cf6f09ca53fe154ce7f69a/train.py) | 修改 `model_name`，两种模型各运行一次；seed 42，54,000 样本，3 个 epoch |
| [初始化敏感性](initialization.md) | [d0eeb9c](https://github.com/peiyangjiang/fashion-mnist-research/blob/d0eeb9ce2f600b9081442fcede71d984f5f298f3/train.py) | 修改 `model_name`、`init_seed`；两种模型 × seed 42～46，3 个 epoch |
| [训练数据量](data_size.md) | [28cb770](https://github.com/peiyangjiang/fashion-mnist-research/blob/28cb7709ed4d0913b413df01a54db485a6903fa8/train.py) | 修改 `model_name`、`train_size`、`init_seed`；两种模型 × 三种数据量 × 五个初始化，5,000 次更新 |
| [错误标签实验](label_noise.md) | [d078774](https://github.com/peiyangjiang/fashion-mnist-research/blob/d078774da6e9cfa51033afe2b7ae482b4e5973f9/train.py) | 用命令参数选择两种模型 × 0%／20% 噪声 × 五个初始化；18,000 样本，5,000 次更新 |

归档版本保存的是该阶段最后一组配置；复现其他组合时，按对应实验记录修改上述变量，其余设置保持不变。错误标签 seed 42 的四组原始记录使用命令参数加入前的 [f242f8b 版本](https://github.com/peiyangjiang/fashion-mnist-research/blob/f242f8bdbf12b47d10c0123f0c897bcce987e4d5/train.py)。

例如，从项目根目录导出数据量实验的历史程序，保留当前 `train.py`：

```powershell
if (Test-Path .\reproduce_data_size.py) { throw "该文件已存在，请换用新的文件名。" }
git show 28cb770:train.py | Set-Content -LiteralPath .\reproduce_data_size.py -Encoding utf8
```

在导出的文件中设置目标 `model_name`、`train_size`、`init_seed`，再从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe .\reproduce_data_size.py
```

这两段命令仅导出并运行一组配置，不会汇总全部组合。其他历史程序也可用对应提交编号导出到不同的新文件名；安装依赖的方法见上文。保存重复运行日志时使用新编号，保留已有原始日志。

结果来源与复现边界：

- 数据量实验的 [汇总 JSON](results/data_size_summary.json) 保存源码快照、各组配置和 27 组补跑的执行源码校验值；其余 3 份已有日志未保存执行时源码校验值。
- 错误标签实验的 [汇总 JSON](results/label_noise_summary.json) 保存 20 组指标与日志校验值，以及新增 16 组使用的源码快照；[审计 JSON](results/label_noise_audit.json) 保存数据与标签核对。
- 初始 baseline 没有单独归档的完整日志，初始化实验的 seed 42 来源于比较记录，不能视为各自另有一份完整日志。归档 Git 版本也不能替代缺失的执行时记录。
- 环境已记录 Windows、Python 3.13.9、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128 和本机 GPU；依赖清单只固定直接依赖。固定 seed 与确定性设置支持本机重复性，不保证不同硬件、驱动或软件版本逐位一致。

## 阶段性完成范围

本项目已完成三个研究问题的受控实验、初始化重复、结果分析和研究报告，可作为第一版探索性研究项目收尾。当前结论来自验证集，尚未进行官方测试集的最终评估，也未确定错误标签的记忆机制。增加测试集评估、改变数据划分或记录更长训练过程，属于后续研究，不是本版已经完成的内容。

# gauSTM 中文实现说明

## 1. 文档目的与适用范围

本文档面向 `gauSTM` 当前发布版，说明其代码结构、数值路线、支持的工作流以及当前实现边界。
文档只描述与当前仓库版本直接相关的设计与实现，不再保留早期开发阶段的实验性记录。

## 2. 项目定位

`gauSTM` 的目标是：基于 Bardeen 近似，从 Gaussian 生成的 `.fchk` 与 TDDFT `.log` 文件出发，完成扫描隗道显微（STM）模拟，并输出可进一步处理或绘图的矩阵结果。

当前版本支持三类主要工作流：
- 单参考态 STM：基于单个 Gaussian 参考态，对指定 MO 进行投影并计算 STM 图像。
- delta-SCF / Dyson STM：基于初态与末态 `fchk` 构造 Dyson 轨道，并在 Bardeen 近似下进行扫描积分。
- TDDFT excited-state STM：从 Gaussian TDDFT `.log` 中读取激发态的组态展开，在 determinant 层面累积 Dyson 贡献。

## 3. 总体架构

### 3.1 目录组织

当前发布版采用“包内实现 + 顶层入口”的结构。

- `gaustm/`
  - 核心实现所在的 Python 包目录。
- `stm_bardeen.py`
  - 命令行包装入口，内部调用 `gaustm.stm_bardeen`。
- `plot_stm.py`
  - 绘图入口，内部调用 `gaustm.plot_stm`。
- `run_batch_gpu.sh`
  - 通用 GPU 批处理脚本。
- `examples/`
  - 示例 `.finp` 模板。
- `test/validation_summary.md`
  - 验证与性能摘要。
- `docs/implementation_notes_zh.md`
  - 当前中文实现说明。

### 3.2 核心模块职责

- `gaustm/stm_bardeen.py`
  - 主驱动模块，负责输入解析、AO overlap、Dyson / TDDFT 系数构造、扫描与输出。
- `gaustm/gaussian_fchk.py`
  - Gaussian `.fchk` 解析。
- `gaustm/gaussian_tddft.py`
  - Gaussian TDDFT `.log` 解析与 determinant 分量整理。
- `gaustm/pyscf_overlap.py`
  - AO overlap 后端与 AO 排序映射。
- `gaustm/bardeen_batch.py`
  - CPU/GPU batched 扫描后端。
- `gaustm/basis_utils.py`
  - Gaussian 基组辅助工具。
- `gaustm/plot_stm.py`
  - `.m` 结果解析与 PNG 绘图。

## 4. 数值路线

### 4.1 AO overlap

AO overlap 由 PySCF 计算。程序首先将 Gaussian/FCHK 中的壳层信息转换为 PySCF 可处理的 primitive Cartesian basis，计算 overlap 后，再恢复到 Gaussian/FCHK 的 AO 顺序。

### 4.2 delta-SCF / Dyson

delta-SCF / Dyson 路线读取初末态 `fchk` 后，根据电子数差异判断 `IP` 或 `EA` 分支，并使用 determinant 重叠构造 Dyson 系数。
实现中使用 `numpy.linalg.slogdet` 以降低 determinant 直接计算带来的数值不稳定性。

### 4.3 TDDFT

TDDFT 路线从 Gaussian TDDFT `.log` 中读取激发态的组态展开。
程序以 determinant 为基本组件：对每个 determinant 分量单独计算其与初态之间的 Dyson 贡献，再按 TDDFT 系数进行线性叠加。

## 5. Bardeen 扫描与后端实现

### 5.1 扫描策略

当前发布版的稳定实现对应 2D 扫描模式，即 `STM.Scan = 2`。
程序使用 tip 起点和两个扫描方向向量构造网格，在每个网格点上计算 Bardeen 面积分、AO-to-MO 投影与 Dyson / excited-state 收缩。

### 5.2 后端类型

程序支持：
- `legacy`
- `cpu`
- `batched`
- `gpu`
- `auto`

### 5.3 batched 优化思路

`bardeen_batch.py` 的主要优化在于：预展开 primitive 项，对一批扫描点统一计算 Bardeen 面积分，并用矩阵方式结合 AO-to-MO 投影与后续收缩。在 GPU 模式下，这条路线由 CuPy 加速。

## 6. 输入与输出约定

### 6.1 关键输入

常见关键输入项包括：
- `STM.Molfchk`：初态或参考态 Gaussian `.fchk`
- `STM.MolFinalfchk`：末态参考 Gaussian `.fchk`
- `STM.TDLog`：Gaussian TDDFT `.log`
- `STM.GaussTip`：tip exponent、Cartesian 分量与起始位置
- `STM.VScan`：扫描方向向量与步长

### 6.2 Probe 行为

程序会严格遵守 `STM.GaussTip` 中给出的 Cartesian 分量。
例如 `(0, 0, 0)` 对应 `s` probe，`(1, 0, 0)` 对应 `p_x` probe。

### 6.3 输出

主计算输出为 MATLAB 风格 `.m` 文件，内含扫描网格（`X`、`Y`）与一个或多个状态矩阵。
日志会记录输入摘要、AO overlap / Dyson / TDDFT 信息、扫描进度与耗时统计。

## 7. 当前已验证范围与限制

当前已验证的范围包括：
- 2D 扫描（`STM.Scan = 2`）
- CPU / batched / GPU 后端
- 单态、delta-SCF / Dyson、TDDFT 三类流程
- AO 基函数角动量已在开发样例中验证到 pure `G` shell
- Gaussian tip 分量已公开验证 `s` 与 `p_x`

需要额外注意：
- TDDFT 路线依赖 Gaussian TDDFT `.log` 中实际打印出激发态组态展开。若日志未打印相关信息，则 TDDFT 路线无法重构 excited-state determinant 贡献。
- 尽管代码中的 AO 排序与 cart-to-sph 处理已做了更一般化处理，但超过当前验证集范围的更高角动量 AO shell 或更高阶 tip 分量，仍应视为“实现上可尝试、工程上未充分验证”的范围。
- 当前仓库中关于性能、基函数数量、AO overlap 矩阵维度、扫描网格尺寸以及 determinant / Dyson 规模的结论，应以 `test/validation_summary.md` 为准。

## 8. 建议阅读顺序

若首次阅读本项目，建议按以下顺序理解：
1. `README.md`：了解支持的工作流与使用方式
2. `gaustm/stm_bardeen.py`：把握主流程
3. `gaustm/gaussian_fchk.py` 与 `gaustm/gaussian_tddft.py`：理解输入解析
4. `gaustm/pyscf_overlap.py`：理解 AO overlap 后端
5. `gaustm/bardeen_batch.py`：理解 CPU/GPU batched 扫描实现

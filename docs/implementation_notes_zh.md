# gauSTM 中文实现说明

本文档只保留与当前 `gauSTM` 发布版直接相关的实现说明，不再记录早期开发阶段的历史痕迹。

## 1. 项目定位

`gauSTM` 是一个基于 Bardeen 近似的 STM 模拟程序，输入来自 Gaussian 生成的 `.fchk` 与 TDDFT `.log` 文件。

当前支持三类工作流：
- 单态 STM：从单个 Gaussian 参考态投影指定轨道
- delta-SCF / Dyson STM：从初态与末态 `fchk` 构造 Dyson 轨道并进行 Bardeen 扫描
- TDDFT excited-state STM：从 Gaussian TDDFT `.log` 中读取激发态组态展开，在 determinant 层面积累 Dyson 结果

## 2. 目录结构

当前发布版采用包目录结构：
- `gaustm/stm_bardeen.py`：主驱动
- `gaustm/gaussian_fchk.py`：Gaussian `.fchk` 解析
- `gaustm/gaussian_tddft.py`：TDDFT 激发态解析
- `gaustm/pyscf_overlap.py`：AO overlap 后端
- `gaustm/bardeen_batch.py`：CPU/GPU batched 后端
- `gaustm/basis_utils.py`：基组与变换工具
- `gaustm/plot_stm.py`：结果绘图

顶层的 `stm_bardeen.py` 与 `plot_stm.py` 只是薄包装入口，方便命令行调用。

## 3. 数值路径概览

### 3.1 AO overlap

AO overlap 由 PySCF 计算，再映射回 Gaussian/FCHK 的 AO 顺序。

### 3.2 delta-SCF / Dyson

程序会根据初末态电子数差异识别 `IP` / `EA` 分支，使用行列式重叠与 `slogdet` 构造 Dyson 系数。

### 3.3 TDDFT

TDDFT 路线读取 Gaussian `.log` 中的激发态组态展开。
程序将每个 determinant 分量单独与初态计算 Dyson，再按 TDDFT 系数进行线性叠加，最终进入 Bardeen 扫描。

## 4. Probe 行为

`STM.GaussTip` 中给出的 Cartesian 分量会被直接尊重。
例如 `(1, 0, 0)` 只输出 `p_x`，不会再把整个 `p` 壳层的三个方向一起写出。

## 5. 当前已验证范围

- 2D 扫描（`STM.Scan = 2`）
- CPU / batched / GPU 后端
- 单态、delta-SCF / Dyson、TDDFT 三类流程
- 开发样例中涉及到的高角动量 pure shell 情形

更详细的运行结果与速度数据请看 `test/validation_summary.md`。

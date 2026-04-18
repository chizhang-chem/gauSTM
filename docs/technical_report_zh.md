# GPU STM 项目技术文档

## 0. 2026-04-18 环境更新

本节用于覆盖文档中较早阶段留下的环境状态描述，以下结论以本节为准：

- 当前项目自己的 `.venv` 已补齐 GPU 依赖，可独立运行 GPU 后端。
- 当前项目 `.venv` 中已安装：`cupy-cuda12x==14.0.1`、`cuda-pathfinder==1.5.3`、`nvidia-cublas-cu12==12.9.2.10`、`nvidia-cuda-nvrtc-cu12==12.9.86`、`nvidia-cuda-runtime-cu12==12.9.79`。
- 已验证当前项目 `.venv` 可以成功 `import cupy`、识别 1 块 GPU，并完成实际 GPU 数组运算。
- 已用当前项目 `.venv` 端到端执行 `stm_bardeen.py copc_0p0157_7p20_147_151.finp 1 gpu`，`201x201` 网格样例运行成功，扫描时间 `21.0s`，总时间 `24.4s`。
- 本次端到端验证使用的是默认 `row_batch=8`，因此速度低于仓库中 `row_batch=48` 的历史 GPU 日志，这是参数差异，不是 GPU 环境失效。
- `run_all_gpu_anion3.sh` 已改为默认使用 `${PROJECT_ROOT}/.venv/bin/python`，不再硬编码到外部虚拟环境；如有需要，仍可通过 `PYTHON_BIN` 环境变量覆盖。
## 1. 文档目的

本文档面向当前目录下的 `pyscf_based_stm_gpu` 项目，整理以下内容：

- 项目目标与整体架构
- 当前 `.venv` 的实际依赖环境
- 端到端技术路线
- 日志输出机制
- CPU/GPU 加速实现方式
- 当前仓库内可见的运行证据
- 已知问题与后续建议

文档依据的主要来源为当前仓库代码、`README.md`、`CHANGELOG.md`、批处理脚本以及 `test/` 目录中的现有运行产物。

## 2. 项目定位

该项目用于执行基于 Bardeen 近似的 STM 模拟。整体思路不是重写全部量化流程，而是在保留原有 STM 主流程的前提下，将部分底层积分与数值计算改为更适合 Python、矩阵计算和 GPU 的实现。

当前实现中，项目的关键数值路径分为两块：

- AO overlap 由 `PySCF` 后端计算。
- Bardeen 面积分由项目自身实现，支持原始 CPU 多进程、批处理 NumPy、批处理 CuPy 三种执行方式。

## 3. 当前目录结构与职责

核心文件如下：

- `stm_bardeen.py`
  - 主入口。
  - 负责解析 `.finp`、读取 `.fchk`、计算 AO overlap、Dyson 系数、执行 2D 扫描、输出 MATLAB 风格 `.m` 数据。
- `bardeen_batch.py`
  - 批处理扫描后端。
  - 负责把单点 Bardeen 积分改造成可按一批扫描点统一计算的 CPU/GPU 路径。
- `pyscf_overlap.py`
  - PySCF AO overlap 后端。
  - 负责把 Gaussian/FCHK 壳层展开为 PySCF 可处理的 primitive Cartesian basis，再映射回 Gaussian/FCHK 的 AO 排列。
- `gaussian_fchk.py`
  - 解析 Gaussian `.fchk` 文件，产出 `MolData`。
- `basis_utils.py`
  - 壳层展开、Cartesian GTO 归一化、1D overlap 递推、Cartesian 到 spherical 变换等基础工具。
- `plot_stm.py`
  - 读取 `.m` 输出并绘制 `.png` 图像。
- `run_all_gpu_anion3.sh`
  - 为一批 `.finp` 自动注入 GPU 配置，批量运行并汇总时间结果。
- `performance_test.py`
  - 早期性能分析脚本，内容有参考价值，但当前导入路径仍指向旧模块名和旧目录，不是当前仓库的直接可运行基线。

`test/` 目录当前主要存放的是 GPU 运行产物：

- `*_gpu.log`
- `*_gpu.m`
- `*_gpu.png`
- `gpu_batch_summary.tsv`

它更像“样例运行结果目录”，而不是自动化单元测试目录。

## 4. 当前实际环境与依赖

用户指定的虚拟环境路径为：

`\\wsl.localhost\Ubuntu-22.04\home\zhangchi\pyscf_based_stm_gpu\.venv`

实际检查结果：

- Python: `3.10.12`

`.venv` 中当前已安装的关键包包括：

- `contourpy 1.3.2`
- `cuda-pathfinder 1.5.3`
- `cupy-cuda12x 14.0.1`
- `cycler 0.12.1`
- `fonttools 4.62.1`
- `h5py 3.16.0`
- `kiwisolver 1.5.0`
- `matplotlib 3.10.8`
- `numpy 2.2.6`
- `nvidia-cublas-cu12 12.9.2.10`
- `nvidia-cuda-nvrtc-cu12 12.9.86`
- `nvidia-cuda-runtime-cu12 12.9.79`
- `packaging 26.1`
- `pillow 12.2.0`
- `pyparsing 3.3.2`
- `pyscf 2.12.1`
- `python-dateutil 2.9.0.post0`
- `scipy 1.15.3`
- `setuptools 59.6.0`
- `six 1.17.0`

当前环境状态：

- 当前项目 `.venv` 已具备 GPU 运行所需的 `CuPy` 与 CUDA 运行库依赖。
- 已验证当前项目 `.venv` 可以成功 `import cupy`、识别 1 块 GPU，并完成实际 GPU 数组运算。
- 已使用当前项目 `.venv` 端到端执行 `stm_bardeen.py copc_0p0157_7p20_147_151.finp 1 gpu`，`201x201` 网格样例运行成功，扫描时间 `21.0 s`，总时间 `24.4 s`。
- 该次验证采用默认 `row_batch=8`，因此慢于仓库中 `row_batch=48` 的历史 GPU 样例；这是参数差异，不是环境异常。

关于批处理脚本的当前状态：

- `run_all_gpu_anion3.sh` 已改为默认使用 `${PROJECT_ROOT}/.venv/bin/python`。
- 如需切换解释器，仍可通过 `PYTHON_BIN` 环境变量覆盖。
- 早期 GPU 日志可能来自外部环境，但当前仓库已经可以使用自身 `.venv` 独立运行 GPU 后端。

## 5. 技术路线

### 5.1 端到端流程

项目的主计算流程如下：

```text
.finp 输入
  ->
parse_finp()
  ->
read_fchk() 读取初态/末态分子
  ->
compute_ao_overlap() 计算 AO overlap
  ->
compute_dyson() 计算 Dyson 系数
  ->
2D tip 扫描
  ->
compute_bardeen_ao_tip() 或 compute_bardeen_ao_tip_batch()
  ->
AO -> MO 投影
  ->
Dyson 收缩或指定 MO 提取
  ->
输出 MATLAB 风格矩阵 .m
  ->
plot_stm.py 渲染 PNG
```

### 5.2 输入层

`stm_bardeen.py` 的 `parse_finp()` 负责读取：

- 是否启用 Bardeen
- 是否计算 Dyson
- 初态与末态 `.fchk` 路径
- 扫描维度和网格
- 针尖高斯参数
- 后端类型 `STM.Backend`
- 行批大小 `STM.RowBatch`

命令行覆盖形式为：

```bash
python stm_bardeen.py input.finp [n_cores] [backend]
```

支持的后端包括：

- `legacy`
- `cpu`
- `batched`
- `gpu`
- `auto`

### 5.3 AO overlap 路线

AO overlap 的技术路线在 `pyscf_overlap.py` 中实现，可概括为：

1. 从 `.fchk` 解析结果中提取原子、壳层、primitive 指数和系数。
2. 将 Gaussian/FCHK 壳层展开为 PySCF 可处理的 primitive Cartesian basis。
3. 构建 PySCF `Mole` 对象，并调用：
   - `int1e_ovlp_cart`
   - 或 `gto.intor_cross("int1e_ovlp_cart", ...)`
4. 通过项目内部构造的变换矩阵，把 primitive Cartesian overlap 重新映射回 Gaussian/FCHK 的 AO 基序。

最终返回：

```text
S_ao = T_bra @ S_cart @ T_ket^T
```

该设计的核心意义是：

- 复用 PySCF 已有的 AO overlap 能力；
- 避免手工重写 AO overlap；
- 保持与 Gaussian/FCHK AO 顺序兼容。

### 5.4 Dyson 系数路线

Dyson 系数由 `compute_dyson()` 计算。

主公式之一为：

```text
OVIMOA = final.mo_alpha @ S_ao @ initial.mo_alpha.T
```

之后根据电子数变化判断属于：

- `alpha-IP`
- `beta-IP`
- `alpha-EA`
- `beta-EA`

代码内部使用 `numpy.linalg.slogdet` 处理行列式，避免直接求 determinant 时的数值上溢或下溢。

### 5.5 Bardeen 扫描路线

单点积分由 `compute_bardeen_ao_tip()` 负责，其核心是对每个 AO 与 tip component 计算 Bardeen 面积分。

如果是原始路径，扫描方式为：

- 对每一行 `j`
- 对每一个网格点 `i`
- 重新构造 tip 位置
- 调用 `compute_bardeen_ao_tip()`
- 再做 AO 到 MO 的投影
- 再做 Dyson 收缩

这条路径可读性强，但 Python 循环很多，是历史性能瓶颈。

## 6. 加速方式设计

### 6.1 `legacy` 后端：CPU 多进程按行并行

`legacy` 是原始加速路径：

- 使用 `multiprocessing.Pool`
- 每个 worker 处理一整行扫描点
- 利用 `initializer` 把大对象放入全局 `_worker_data`
- 避免每个任务重复序列化大矩阵

特点：

- 优点：实现直接，适合保留原始逐点逻辑。
- 缺点：每个点仍执行 Python 层积分循环，算力利用率一般。
- 适合：没有 CuPy 时作为传统 CPU 并行方案。

### 6.2 `cpu` / `batched` 后端：批处理 NumPy 向量化

`bardeen_batch.py` 是本项目真正的结构性优化核心。

它的思路不是“把 legacy 包一层”，而是重构为批计算：

1. 先通过 `build_bardeen_batch_plan()` 预展开所有 primitive 项，形成 `BardeenBatchPlan`。
2. 把一批 tip 点的坐标一次性交给 `compute_bardeen_ao_tip_batch()`。
3. 在同一批点上统一计算：
   - x 方向 Gaussian moment
   - y 方向 Gaussian moment
   - z 方向多项式高斯项及导数
4. 对全部点批量累加得到 `THMAO`。
5. 再通过矩阵乘法或 `einsum` 一次性完成 AO 到 MO 的投影与 Dyson 收缩。

关键实现特征：

- 计算库由参数 `xp` 抽象，CPU 时是 `numpy`。
- 批投影使用：

```text
xp.einsum("mb,pbc->pmc", coeff_matrix, thmao_batch)
```

- Dyson 收缩使用：

```text
xp.einsum("is,pic->psc", modyson, active)
```

优势：

- 大幅减少 Python 循环层级。
- 把更多工作转化为数组运算和 BLAS 风格操作。
- 同时为 GPU 路径铺平接口。

### 6.3 `gpu` 后端：批处理 CuPy

`gpu` 后端并没有单独复制一套计算代码，而是沿用批处理方案，只把 `xp` 从 `numpy` 替换成 `cupy`。

具体做法：

- `resolve_backend("gpu")` 返回 `cupy` 模块。
- `coeff_matrix`、`MODyson`、tip 坐标等转为 GPU 数组。
- `compute_bardeen_ao_tip_batch()` 内部的数组分配、指数、幂运算、`einsum` 都交给 CuPy。
- 每个批次结束后通过 `to_numpy()` 把结果从 GPU 拉回主机内存，再填回最终输出网格。

这条路线的意义是：

- 保持 CPU batched 与 GPU batched 的算法一致；
- 最大限度共用代码；
- 让 GPU 加速集中发生在最重的批量数组运算阶段。

### 6.4 `auto` 后端：自动降级

`auto` 后端策略：

- 若检测到 `CuPy`，使用 `gpu`。
- 若未检测到 `CuPy`，自动退回 `batched`。

这使项目在“有 GPU 环境”和“无 GPU 环境”之间切换时不需要维护两套使用方式。

### 6.5 `row_batch` 的作用

`row_batch` 控制每次打包多少行一起算。

影响：

- 批次越大，GPU 和 NumPy 吞吐通常越高；
- 批次越大，显存和内存占用也越高；
- 需要在吞吐和内存之间平衡。

代码默认值：

- 配置缺省为 `8`

批量 GPU 脚本默认值：

- `run_all_gpu_anion3.sh` 中通过环境变量设置，默认 `48`

## 7. 日志设计

### 7.1 日志输出通道

主程序采用“数据与日志分流”的设计：

- 计算结果 `.m` 通过 `stdout` 输出
- 运行日志通过 `stderr` 输出

这使得如下调用非常自然：

```bash
python stm_bardeen.py input.finp > output.m 2> output.log
```

这也是批处理脚本当前采用的方式。

### 7.2 日志内容结构

`stm_bardeen.py` 的日志信息主要分为以下层次：

1. 启动信息
   - 输入文件
   - Bardeen/Dyson/TDA/Gauss 开关
   - backend 与 row_batch
   - `.fchk` 路径
   - tip 参数
   - 扫描网格大小

2. 模型加载信息
   - 初态方法、电子数、基函数数、MO 数
   - 末态方法与电子数

3. 核心阶段信息
   - `* Computing AO overlap...`
   - `* AO overlap done. diag range=[...]`
   - `* Computing Dyson coefficients...`
   - `* Branch: ..., NDyson=..., offset=...`

4. 扫描阶段信息
   - 网格总点数
   - 使用后端
   - `row_batch`
   - 周期性进度、耗时、速率、ETA

5. 收尾信息
   - 扫描总时间
   - tip alpha
   - total time
   - summary

### 7.3 进度日志格式

GPU 和批处理路径的典型进度日志格式如下：

```text
row 48/201 | 1.1s elapsed | 44.2 rows/s | ETA 3s
```

含义：

- 当前累计完成行数
- 已耗时
- 当前行吞吐
- 预计剩余时间

`legacy` 多进程路径也有类似进度日志，但输出频率略不同：

- 多进程时每 20 行输出一次
- 单进程 legacy 时每 10 行输出一次

### 7.4 批处理脚本如何使用日志

`run_all_gpu_anion3.sh` 会：

1. 对每个 `.finp` 临时附加：
   - `STM.Backend gpu`
   - `STM.RowBatch 48`
2. 把 `stdout` 保存为 `.m`
3. 把 `stderr` 保存为 `.log`
4. 从日志里提取：
   - `Scan completed in ...`
   - `Total time: ...`
5. 再把这些结果写入 `gpu_batch_summary.tsv`

因此，日志不仅用于人工观察，也已经被当作批量任务统计数据的机器可读来源。

## 8. 当前仓库中的运行证据

### 8.1 单个 GPU 样例日志

`test/copc_0p0157_7p20_147_151_gpu.log` 显示：

- 输入体系：`201 x 201` 网格，共 `40401` 点
- 后端：`gpu`
- `row_batch=48`
- 分子规模：`1519 basis`, `1519 MOs`
- Dyson 分支：`alpha-EA`
- `NDyson=1373`
- 扫描时间：`5.0 s`
- 总时间：`8.3 s`
- 扫描吞吐：`8106.3 points/s`

### 8.2 批量 GPU 汇总表

`test/gpu_batch_summary.tsv` 包含 11 个任务的结果。

从表中可以直接读出：

- 扫描时间范围：`4.1 s` 到 `5.0 s`
- 总时间范围：`7.5 s` 到 `8.5 s`
- 11 个任务的平均扫描时间：`4.5 s`
- 11 个任务的平均总时间：`7.9 s`

这说明当前批处理 GPU 路线在该批数据上已经具备稳定可复现的执行结果。

### 8.3 历史 CPU 参考

`CHANGELOG.md` 中记录的历史 CPU 路径信息包括：

- 10 核 CPU 并行：约 `30 s`
- 5 核：约 `92 s`
- 串行估计：约 `150 s` 到 `300 s`

这组数据来自早期开发日志，和当前 GPU 结果不一定是完全同一代码版本、同一环境、同一脚本路径，因此只能作为历史参考，而不是严格同条件 benchmark。

但从量级上看，GPU 批处理路径已经明显快于历史 CPU 路径。

## 9. 现阶段的关键结论

### 9.1 已完成的技术目标

- 已用 `PySCF` 接管 AO overlap 计算。
- 已实现 Dyson 系数与扫描主流程。
- 已从原始逐点扫描演进到 batched 扫描。
- 已实现统一的 `NumPy/CuPy` 双后端设计。
- 已有一批真实 GPU 运行结果和汇总日志。

### 9.2 当前最大的工程问题

当前最需要继续补强的不是“有没有 GPU 环境”，而是“工程化和可验证性”两件事：

- 当前项目 `.venv` 已经可以独立跑 GPU 后端。
- 批处理脚本也已经改为默认使用当前项目环境。
- 现在的主要短板转为自动化测试、性能参数固化和依赖清单沉淀。

这意味着：

- 代码与环境都已经是可运行的 GPU-ready 状态。
- 后续工作的重点应转向稳定性、回归测试和可移植性，而不是继续手工拼环境。

### 9.3 日志体系已经足够支撑批量任务

目前日志包含：

- 参数
- 阶段
- 进度
- 耗时
- 汇总

并且已经被批处理脚本自动解析生成表格，因此日志设计本身是有工程价值的，而不是临时调试输出。

## 10. 已知问题与风险

### 10.1 当前剩余的环境风险

当前目录的 `.venv` 已经可用，但仍有两个现实约束需要记录：

- 目前安装的是 `cupy-cuda12x==14.0.1`，依赖 CUDA 12 系列运行库。
- 如果换到不同 CUDA/驱动组合的机器，可能需要更换对应的 CuPy 包版本。

因此，当前问题已经不是“缺少 GPU 环境”，而是“GPU 环境与目标机器是否匹配”。

### 10.2 批处理脚本的现状与剩余风险

`run_all_gpu_anion3.sh` 已不再硬编码外部解释器，默认行为已经改为：

```text
${PROJECT_ROOT}/.venv/bin/python
```

这显著改善了可移植性，但仍需注意：

- 该脚本仍依赖项目目录结构保持稳定。
- 实际数据目录仍默认指向 `/home/zhangchi/fchk_for_test/anion_3`。
- 若迁移到其他机器或数据目录，建议显式设置 `PROJECT_ROOT`、`PYTHON_BIN`、`WORK_DIR`。

### 10.3 `performance_test.py` 不是当前仓库的直接可运行测试

它仍然引用旧模块名和旧路径，例如：

- 旧的 `sys.path.insert(...)`
- `stm_bardeen_table_c2s_integral_lib`

因此它更适合作为历史性能分析脚本参考，不应被当作当前版本的正式回归测试。

### 10.4 `test/` 目录缺少自动化数值一致性测试

目前能看到的是运行产物，不是自动测试。

理想状态应补充如下回归：

- `legacy` vs `batched` 数值一致性
- `batched` vs `gpu` 数值一致性
- 不同 `row_batch` 下的数值一致性
- `auto` 降级行为测试

## 11. 建议的技术路线升级

### 11.1 环境层

环境统一这件事已经完成，后续建议聚焦在沉淀和复用：

1. 生成明确的依赖清单，例如 `requirements.txt` 或 `environment.yml`
2. 在文档中写清楚 CUDA 与 `CuPy` 的对应关系
3. 为新机器补一段最小环境自检流程，例如 `import cupy`、设备数检测和一次小规模 GPU 运算

### 11.2 工程层

建议后续整理：

1. 为批处理脚本补充使用示例和推荐环境变量写法
2. 增加正式测试目录和自动化测试
3. 为日志增加可选文件输出参数，而不是只依赖 shell 重定向

### 11.3 性能层

当前项目已经完成“从逐点 Python 逻辑到 batched 数组计算”的关键跨越。下一步如果继续优化，可以考虑：

1. 对 `row_batch` 做自动调优
2. 减少 GPU 与 CPU 间的数据回传频率
3. 增加更系统的 benchmark 脚本
4. 在固定输入上同时比较 `legacy/cpu/gpu/auto` 四种路径

## 12. 可直接引用的结论

如果只保留一句话来概括当前项目：

> 这是一个基于 PySCF 计算 AO overlap、基于 batched NumPy/CuPy 计算 Bardeen 扫描的 STM 项目，当前项目自己的 `.venv` 已经补齐 GPU 依赖并可独立运行 GPU 后端，批处理脚本也已切回项目内解释器，后续重点在自动化测试、依赖固化和性能参数整理。

## 13. 附：推荐运行方式

如果希望优先保证兼容性，建议：

```bash
.venv/bin/python stm_bardeen.py input.finp 1 auto > output.m 2> output.log
```

如果希望显式走 GPU 后端，当前 `.venv` 已可直接使用：

```bash
.venv/bin/python stm_bardeen.py input.finp 1 gpu > output.m 2> output.log
```

如果希望接近仓库中历史 GPU 样例的批量参数，可在 `.finp` 或环境变量中设置更大的 `row_batch`，例如：

```bash
STM_ROW_BATCH=48 ./run_all_gpu_anion3.sh
```

批量绘图方式：

```bash
.venv/bin/python plot_stm.py output.m output.png
```

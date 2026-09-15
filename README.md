# 家具条件下的相机布局规划 — YSynthetic 外置研究原型

当前主线（2026-09-15）：**目标/附属物体/其他障碍三类几何角色 → 自由交互区域 → 自适应视点域 → 多视角集合选择与开放运镜 → 生成人物后选择实际关键帧**。

本次完成了轻量单元测试，**没有运行完整规划或 Blender 大场景渲染**。具体入口与限制见 [目标视点域实现说明](docs/TARGET_VIEWSPACE.zh-CN.md)和[目标附属物体与自由交互区域](docs/SUPPORT_AWARE.zh-CN.md)。

`planning_mode: target_viewspace` 在没有检测到附属物体时保留原 capture-box 证据；检测到桌面附属物体时，改用不与场景几何冲突的未知人体自由交互代理。第二阶段 `select-human-frames` 消费外部检测与背景相机验证结果，不会自动调用视频模型或人体检测网络。

下面的 `robust_space`、RQ1 和历史能力记录属于此前兼容分支；不代表新主线已经验证通过。0.4.0（2026-09-09）的自由空间方案仍可用于对照。

**当前要做的事：人物尚未生成时，围绕指定家具规划构图合理、方向互补且能连接的镜头；人物生成后，再用实际观测筛帧。** 未知人体完整入镜和生成后的多视角一致性仍须后验检查。

## 从新版开始

优先阅读 [当前算法、运行与限制](docs/TARGET_VIEWSPACE.zh-CN.md)。[自由空间旧方案](docs/ROBUST_PLANNING.zh-CN.md) 与以下示例命令保留作兼容模式。

```powershell
Set-Location D:\camera\code
uv sync --extra dev --extra solver
.\.venv\Scripts\python.exe -m camera_planning plan --request configs/robust_sofa_demo.yaml --output runs/my_robust_plan
.\.venv\Scripts\python.exe -m camera_planning select-cached --plan-dir runs/my_robust_plan --method lazy_coverage --output runs/my_coverage_baseline
.\.venv\Scripts\python.exe -m pytest -q
```

`solver` 是可选的 SciPy/HiGHS **普通覆盖 MILP**，不是鲁棒目标的全局求解器。不需要它时只安装 `--extra dev`。

旧自由空间分支先选点后连线；新 `target_viewspace` 分支在贪心插入机位时计入避障路径长度与连接可行性。两者都没有联合全局最优保证。

本目录独立于 `D:\westlake\YSynthetic`。开发期间只读取它的代码/环境，没有修改它的源码、配置或已有结果。以后通过文件契约并入，不需要先把下游网络搬到这里。

## 建议阅读顺序

1. [整体设计](docs/DESIGN.zh-CN.md)：问题定义、每一步算法、为什么这样设计。
2. [目标附属物体与自由交互区域](docs/SUPPORT_AWARE.zh-CN.md)：桌面物体三类角色、自由区和评分语义。
3. [论文与取舍](docs/LITERATURE.zh-CN.md)：已核对的一手文献、精读/复现顺序、哪些不能直接照搬。
4. [YSynthetic 对接](docs/INTEGRATION.zh-CN.md)：真实 K/R/T、scene、body-fit 和融合接口。
5. [RQ1 实验操作](docs/RQ1.zh-CN.md)：如何从规划到参考人体回环测试，尚需手动完成什么。
6. [阶段计划](docs/ROADMAP.zh-CN.md)：已经实现与下一步验收条件。
7. [本机验证记录](docs/VALIDATION.zh-CN.md)：本次真正运行了什么，没运行什么。
8. [从 Blender 开始的一键操作](docs/QUICKSTART.zh-CN.md)：真实场景要准备什么、运行哪条命令、怎么看输出。
9. [完整测试指南](docs/TESTING.zh-CN.md)：从选择家具对象到查看相机与 PNG 的逐步操作。

## 当前实际能力

| 部分 | 状态 |
| --- | --- |
| 严格输入契约，米制、Y/Z up、固定 K 与预算 | 已实现 |
| 完整场景＋单独目标家具 → 冻结 `.blend` → 世界 AABB 或三角网格代理 | 已实现；目标按世界包围盒自动匹配，不要求名称 token |
| 家具局部 OBB 扩展壳层候选、多高度/距离、几何过滤 | 已实现；旋转家具不再只依赖世界轴 AABB |
| 家具表面＋确定性完整构图包络采样、分区均衡、静态 mesh 遮挡 | 已实现；没有人物位置/姿态概率 |
| 覆盖、有效双视角、分辨率、投影信息量评分 | 已实现；都是待验证的几何代理 |
| 贪心＋单相机替换、随机/均匀/方向分散/覆盖基线 | 已实现；小池还可穷举 |
| `cameras.json` / `scene.yaml` / 诊断导出 | 已实现；对现有 YSynthetic schema 实测 |
| RQ1 子集实验组织、相机并集复用、重复集合去重 | 已实现 |
| Blender 相机预览、成对批量渲染与标定检查 | 已实现；受控 demo 已完整运行 |
| YSynthetic 调用计划与显式续跑执行器 | 已实现；执行重模型仍须人工确认 |
| MHR 融合 NPZ 的关节/根误差及 JSON/CSV 汇总 | 已实现；拓扑核验后的顶点误差尚未实现 |
| 一条命令完成导入、规划、预览和采集 | 已实现：`camera_planning run-pilot` |
| reference NPZ 自动解码成有材质人体、真实端到端 RQ1、后端感知学习 | 未实现 |
| 依次经过选定相机的 Blender 运镜预览与 H.264 视频 | 已实现；不首尾闭环，并做采样 AABB 碰撞检查；不是物理轨道规划保证 |

## 不安装新环境也能先运行

打开 PowerShell，复制下面命令。使用本机已有的轻量 Python，只读取它，不向 YSynthetic 的环境安装包。

```powershell
Set-Location D:\camera\code
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'D:\camera\code\src'
$cameraPython = 'D:\westlake\YSynthetic\.venv\Scripts\python.exe'
& $cameraPython -B -m camera_planning plan --request configs/demo_sofa.yaml --output runs/my_first_plan
& $cameraPython -B -m pytest -q
```

运行约数秒以内的小示例只需要 CPU。示例的沙发/墙体是**人工构造的代理几何**，不是自动提取的 `test_sofa`。不要将其选点当作真实场景最优相机。

真实 Blender 场景的一键入口：

```powershell
& $cameraPython -B -m camera_planning run-pilot --config configs/engineering_demo.yaml
```

`engineering_demo.yaml` 是本机受控工程测试；真实研究必须复制一份新配置并替换 `source_scene`、对象名、允许相机区域和输出目录。程序不会覆盖非空输出。完整说明见 [QUICKSTART](docs/QUICKSTART.zh-CN.md)。

重复运行请换一个新输出目录；程序拒绝覆盖非空实验目录。不要把已有实验删掉来掩盖参数变化。

若后续希望完全独立安装，在这里创建 `.venv`，用 Python 3.11/3.12，安装 `.[dev]`；不要在 YSynthetic 的环境里执行 editable install。本机测试使用 Python 3.12.13、NumPy 2.5.0、Pydantic 2.13.4、pytest 9.1.1。`pyproject.toml` 给的是兼容范围，不是假称已锁定的精确依赖。

## 会生成什么

```text
runs/my_first_plan/
  request.json                规范化输入
  candidate_cameras.json      通过过滤的完整相机池
  cameras.json                最后选择的 B 台相机；YSynthetic 格式
  camera_placements.json      便于人工阅读的世界坐标位置与朝向
  evidence.npz                相机×家具表面/构图包络点的可见性和几何信息
  camera_plan_result.json     分项分数、候选拒绝原因、算法轨迹、哈希、限制
```

`camera_plan_result.json` 的 `reconstruction_status: not_run` 是有意设计：产生相机不等于已恢复人体，也不等于已经证明优于原方案。

## 代码组织

```text
src/camera_planning/
  contracts.py       输入/相机结构；正式接口禁止 reference_human 等未知字段
  geometry.py        OpenCV 投影、Blender 变换、GeometryBackend、AABB 查询
  mesh_geometry.py   三角网格、BVH、线段遮挡与闭合物体占用查询
  candidates.py      候选相机生成与可放置性检查
  observation.py     家具表面与确定性完整构图包络；不预测人物位置
  scoring.py         各视角证据与相机集合评分
  selection.py       优化器与基线
  planner.py         正式规划入口
  integration.py     YSynthetic 文件契约与可审计命令计划
  experiments.py     仅实验层允许访问 reference
  controls.py        显式 RQ1 场景质量控制；不是实测检测结果
  evaluation.py      参考人体回环偏差指标
  reporting.py       汇总每组真实融合状态、有效视角和 round-trip 指标
  workflow.py        Blender 场景导入、规划、预览、采集的一键编排
  execution.py       经明确确认后逐阶段执行 YSynthetic 命令并保存日志
  artifacts.py       哈希与输出保护
  cli.py             命令行入口
scripts/
  blender_prepare_scene.py
  blender_extract_target_bounds.py
  blender_extract_scene_boxes.py
  blender_export_scene_geometry.py
  blender_preview_plan.py
  blender_render_rig.py
  blender_render_camera_tour.py
  validate_ysynthetic_contract.py
tests/
configs/
docs/
```

## 最重要的研究边界

几何代理评分不是 SAM 3D Body 的置信度；本地 `body_fuse` 也不是标准三角化网络。现在需要检验的恰好是两者是否相关。

当前参考人体是既有融合估计，不是真实人体 ground truth。回环成功能支持“对该参考形体及当前后端有影响”，不能直接证明生成视频中的真实动作恢复准确，更不能保证任何交互都最优。

下一步建议先把一个真实场景的世界坐标几何与冻结参考人体接入，做固定 B 的小规模 RQ1；确认可测效应后，再决定是否学习后端感知的评分器。详见实验协议。

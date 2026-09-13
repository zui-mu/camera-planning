# RQ1：固定相机数量，只改变布局是否影响回环重建？

## 实验能回答什么

用一个冻结的融合 MHR 人体 H_ref，在相同场景、同一个姿态下从不同相机渲染；每张图重新经 SAM 3D Body 估计，再用当前共识/融合链恢复 H_hat。比较 H_hat 与 H_ref。

它测的是对某个“参考估计”的 round-trip deviation，不是真实人体 ground-truth error。不要再把最初生成视频的不同时间帧直接组合来做受控 RQ1：那会同时改变人体姿态和相机。

**本框架已备好场景导入、三角网格规划、分组、渲染、显式调用执行与结果汇总；本次没有完成真实场景的这轮 GPU 重建。** 真正开跑前仍需把米制场景、reference mesh 和材质放进源场景并核对对应关系。

推荐按 [一键操作文档](QUICKSTART.zh-CN.md) 配置 `run-pilot`。下面保留逐阶段说明，便于理解、审计和单独重跑；不是要求每次都手工拼命令。

## 0. 打开终端，确定环境

Windows 开始菜单搜索 PowerShell，打开后：

```powershell
Set-Location D:\camera\code
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'D:\camera\code\src'
$cameraPython = 'D:\westlake\YSynthetic\.venv\Scripts\python.exe'
$bodyPython = 'D:\westlake\YSynthetic\.venv-sam3d-body\Scripts\python.exe'
$blenderExe = 'E:\jianmoruanjian\blenderdownload\blender.exe'
& $cameraPython -B -m pytest -q
```

核心规划与实验分组用轻量环境；SAM3D/MHR 计算使用已有重模型环境；渲染使用 Blender。不要用系统 Python 3.13 去替换已有可用环境。本版不需要 Houdini 许可证，也不要求先安装 IsaacSim。

## 1. 准备一个真实场景请求，先不要读取 reference

以 `configs/engineering_demo.yaml` 为模板建立真实 pilot 配置，填写源场景、target、人物对象、封闭家具、floor 和允许相机范围。一键流程会从 Blender evaluated mesh 自动产生 triangle-mesh request；`configs/demo_sofa.yaml` 仅用于 AABB 单元示例，不能原封不动与真实 reference 混用。

相机数量建议先固定 B=4，K/分辨率固定。初轮候选池控制在 12–24 个左右：减少方位角/高度层，并保留对照所需的差异。先做几何可视检查，再渲染。大池不是越多越好，会直接增加 GPU 推理成本。

如果需要只运行已有 request 的规划：

```powershell
# 前提：local/test_sofa_request.yaml 已按真实几何创建。
& $cameraPython -B -m camera_planning plan --request local/test_sofa_request.yaml --output runs/sofa_plan_001
```

不要看完测试人体后再移动注视点、收缩 ROI 或删除“不好看的”相机。若发现几何输入错误，应修复并重新产生整个计划，记录变更。

## 2. 固定 reference 和生成子集实验

当前可核对的已有参考文件位置：

`D:\westlake\YSynthetic\.verification\body_fit_real\hierarchical_pipeline\body_fuse\fused_body_parameters.npz`

同目录还有参考 mesh/joints。它是原管线生成的估计，不是独立扫描真值。正式实验先复制/冻结到自己的数据版本目录；仅指定路径并记录 hash 不会阻止外部改写源文件。

```powershell
$referenceNpZ = 'D:\westlake\YSynthetic\.verification\body_fit_real\hierarchical_pipeline\body_fuse\fused_body_parameters.npz'
& $cameraPython -B -m camera_planning prepare-rq1 --plan-dir runs/sofa_plan_001 --output runs/sofa_rq1_001 --reference $referenceNpZ --random-sets 12
```

这一步读取 planner 已生成的候选与证据，不用 reference 改选点。生成 4 类确定性基线/方法集合加 12 个随机集合；重复相机集合标记 duplicate_of，统计时不能当作独立样本。程序不把已有最初视频的人物图片冒充新相机渲染。

## 3. 从打开 Blender 开始准备冻结快照

打开 Blender，通过 File → Open 打开你的室内场景 `.blend`。若只有 USD/USDC/GLB/FBX/OBJ，一键流程可以自动导入并保存冻结 `.blend`，但你仍必须核对世界坐标、单位和材质；自动导入不能判断资产语义是否正确。

把 H_ref 对应的 `fused_mesh_world.ply` 导入。导入 PLY 的轴转换必须与文件中世界坐标一致，检查对象矩阵是否额外旋转；不要通过“看起来坐到了沙发上”为由手动平移/缩放。建议先用相同世界关节点和已知场景标志点核对，再确认人物脚/座面位置。若需要坐标转换，场景、相机、mesh、joints 要一致变换并记录矩阵。

给人物一个明确对象名，例如 `RefHuman`。如果人物由多个网格构成，记录全部对象名。冻结一帧，不在子集间改动作。设置固定灯光和材质并保存源场景；一键流程会另存 `snapshot/scene_snapshot.blend`，不会覆盖原场景。

注意：已有 fused PLY 可能无纹理。无纹理/单色渲染会改变 SAM3D 输入分布；若网络无法稳定识别人，首先是渲染域问题，不代表相机研究失败。第一轮至少人工检查几张图并跑单视角小样，不能一次批量跑完才发现检测失败。

本版不会自动从 NPZ 解码并重建材质，也不会自动核实 `.blend` 中的人体顶点等于 NPZ 对应的顶点。这是当前最重要的人工验收项之一。

完成配置后，推荐用以下命令一次完成冻结、mesh 导出、规划、预览、RQ1 分组和渲染：

```powershell
& $cameraPython -B -m camera_planning run-pilot --config configs/your_rq1.yaml
```

## 4. 先校验相机，再渲染同一个 pool

```powershell
& $blenderExe --background --factory-startup --python-exit-code 2 --python D:\camera\code\scripts\blender_render_rig.py -- --rig D:\camera\code\runs\sofa_rq1_001\pool_cameras.json --validate-only

& $blenderExe --background D:\camera\code\local\sofa_reference.blend --python-exit-code 2 --python D:\camera\code\scripts\blender_render_rig.py -- --rig D:\camera\code\runs\sofa_rq1_001\pool_cameras.json --output D:\camera\code\runs\sofa_rq1_001\renders\human --frame 1

& $blenderExe --background D:\camera\code\local\sofa_reference.blend --python-exit-code 2 --python D:\camera\code\scripts\blender_render_rig.py -- --rig D:\camera\code\runs\sofa_rq1_001\pool_cameras.json --output D:\camera\code\runs\sofa_rq1_001\renders\empty --frame 1 --hide-object RefHuman
```

有多个 reference 对象时重复 `--hide-object`。同一 .blend，只有显示/隐藏人体不同；不修改文件、不保存场景。脚本使用快照内的渲染引擎，帧固定为 1；光照、材质也由同一快照固定。它暂不保证跨硬件位级确定性，因此仍需记录生成图像 hash。

检查 human 图片：人物存在、比例适当、未意外被原有对象遮掉。不要为了好结果只删某些相机；按预注册的失败策略处理。

## 5. 明确场景评分协议

为了单独研究视角影响，第一轮可明确使用常量控制：

```powershell
& $cameraPython -B -m camera_planning prepare-controls --experiment runs/sofa_rq1_001 --human-object RefHuman --acknowledge-assumption
```

这会先核对 render manifests 和图像 hash，然后生成显式注明“不是实测场景分数”的 precomputed 记录。若 manifest 不一致会失败。`r_scene=1` 是隔离变量的实验设定，不是声称 scene_consistency 算法给出了 100% 正确结果。

若要评估完整场景评分链，请改用实际 scene_consistency 输出，填充同样目录；不要同时混用常量和实测而不分组。回到视频生成实验时不得复用此常量假设。

## 6. 生成并检查真实下游命令

```powershell
& $cameraPython -B -m camera_planning integration-plan --experiment runs/sofa_rq1_001 --ysynthetic-root D:\westlake\YSynthetic --python $bodyPython --body-config D:\westlake\YSynthetic\local\configs\sam3d_body.local.yaml --fuse-config D:\westlake\YSynthetic\local\configs\body_fuse.local.yaml --output runs/sofa_rq1_001/commands.json
```

打开 `commands.json`。它只写调用计划，不执行，也不会花视频 API 费用。按顺序先做实际被测试集合使用的相机并集 `body-fit-batch`，再对每个非重复 set 做 pose-consensus 与 body-fuse。

检查完成后可显式运行 `camera_planning execute-integration --commands commands.json --acknowledge-heavy`。执行器不用 shell 字符串，逐阶段保存 stdout/stderr，任何失败立即停止；中断后用 `--resume` 只跳过已有成功记录的阶段。

每个 set 使用同一 scene_id、相同 B、相同模型/config。共用逐视角提案但重新执行集合共识和融合。执行器记录进程结果，但仍不会自动证明所有推理缓存键等价。

## 7. 计算回环误差

```powershell
$env:PYTHONPATH = 'D:\camera\code\src'
& $cameraPython -B -m camera_planning evaluate --reference $referenceNpZ --prediction runs/sofa_rq1_001/sets/greedy_000/outputs/body_fuse/fused_body_parameters.npz --output runs/sofa_rq1_001/sets/greedy_000/metrics.json
```

评估前先确认 body_fuse_result 的 valid/status；指标程序比较 NPZ 不等于替你判断融合流程是否成功。也可以在所有集合结束后运行 `camera_planning summarize-rq1 --experiment runs/sofa_rq1_001`，自动写出 JSON/CSV，并保留失败状态、有效视角数和重复集合关系。

已有指标：世界坐标 joint error、root-aligned joint error、root translation、root rotation和结构化统计汇总。未实现：带 faces/order 核验的 MPVPE、身体/手分组、骨长误差与 PA 对齐。论文至少应补 body-only 指标和有效视角数，不能仅报告所有点平均。

PA-MPJPE 即使以后补上，也不应作为唯一指标，因为它会掩盖尺度/朝向/世界位置错误；项目需要把人放回家具旁，这些误差本身重要。

## 8. 如何判断值得继续

第一轮不要用“最大最小差很多”直接声称显著性：极值会随集合数量增长。先看同布局重复运行的噪声，再看不同布局差异是否超过该噪声，报告分布、效应量、失败率和接受视角数。

建议循序推进：1 个真实场景做工程闭环；之后至少若干家具/参考姿态做 pilot；再按资源扩大到多个场景、不同体型与交互。相同 pool 的重叠子集不是独立人物样本；bootstrap/置信区间应以场景/人物层分组，不能把每个 joint 当独立样本。

固定名义 B 还不够：若有些视角检测失败或被共识排除，记录有效 B。可以报告两条结果：系统级结果包含这些失败；完整可用子集的分析更接近隔离几何效果。不要将失败集合从均值表中无声删除。

RQ2 检验 proxy Q 与误差的 Spearman 排序相关、相对随机的选点收益和 regret；权重只能在训练/验证场景调整。RQ3 才检查生成、匹配和人体不一致的额外影响。

终点应是一份可复验的表：每个 reference、每个 set、选中 ID、名义/有效视角数、检测/共识/融合状态、各项误差、几何评分、耗时与版本哈希。当前仓库没有预填任何“我们更好”的重建结果。

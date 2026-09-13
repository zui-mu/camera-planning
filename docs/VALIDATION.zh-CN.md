# 本次实现验证记录

## 0.4 自适应鲁棒规划验证（2026-09-09）

- 本项目独立 `.venv`：52 passed, 3 skipped（跳过需要外部 YSynthetic 的检查）；包括 SciPy/HiGHS 覆盖 MILP、鲁棒目标小池 exact 对照、Lazy 与普通覆盖 greedy 等价、薄墙阻断、绕障路径、权重/下尾部、缓存 hash、增量证据与初始候选不足时的自适应补齐测试。
- 只读使用 YSynthetic 已有 Python（`-B`）并声明 `YSYNTHETIC_ROOT`：54 passed, 1 skipped。唯一跳过为该环境未安装可选 SciPy 求解器；没有向其环境安装包。
- 曾在相机轻量环境直接启用外部测试，因缺少 YSynthetic 需要的 Pillow 导入失败；改用上述现有下游环境完成实际接口测试，而非修改 YSynthetic 或忽略失败。
- 真实房间**原 AABB 导出复用**：`runs/robust_real_room_replan_v1`。24 个初始合法候选，两轮新增 17/25 个，最终 66 个候选选 8。代理分数 0.370378 → 0.381525 → 0.396680；没有重新射线旧行。开放路径已生成，长度约 10.9473m。
- 同一冻结池 `lazy_coverage` 与 `milp_coverage` 都通过 `select-cached` 运行，零新增几何查询。MILP status=0/gap=0 是**普通覆盖目标**的离散最优证明，不是鲁棒目标最优。对照目录 `runs/robust_real_room_lazy_baseline_v1`、`runs/robust_real_room_milp_baseline_v1`。
- 真实 `CameraRig`、`Scene` schema 和 `load_scene_bundle` 校验通过，8 台相机；这是格式/加载测试，不代表图片已重新渲染。
- 合成 AABB 示例 `runs/robust_sofa_smoke_v2`：116 个最终候选选 8，开放路径成功。`robust_sofa_smoke_v1` 保留为早期归一化尺度试跑，不用于最终比较。
- Blender 5.2 小场景测试：`runs/robust_blender_smoke_v1/preview.blend` 保存成功；`tour/camera_tour.mp4` 实际渲染 15 帧，320×180、6 FPS，仅用于快速工程检查。导出每帧 R/T 及缩放后的 K；路径不闭环，采样 AABB 复核无碰撞。不要把这个低分辨率夹具当作真实房间新结果。
- 后续正式拍摄配置 `configs/testinput_sofa_robust.yaml` 已从 1280×720 调整为 2560×1440 PNG，相机内参同比缩放；视频 75% 渲染为 1920×1080、24 FPS。这里只更新配置，尚未按新规格重新渲染真实房间；不声称已验证其实际画质或修复场景贴图。
- 未重新导入/渲染 1.2GB 的真实房间快照；未运行 SAM3D/HMR、视频 AI、完整 RQ1。这里的分数提高不是人体恢复效果提高。

以下为旧版工程记录。

日期：更新至 2026-09-09。下列是工程验证，不是 RQ1 研究结论。

## AABB 包围盒规划路线

- `blender_extract_scene_boxes.py` 已在 Blender 5.2.0 LTS 的受控场景中成功自动提取目标 Sofa、SideCabinet 和隐藏的 CameraAllowedRegion 世界坐标 AABB；floor/wall 按结构名称 token 忽略。
- 提取结果成功转换为 `camera_planning_request_v2` 的 AABB backend 并完成 4 相机规划：家具盒面覆盖率 1.0、构图包络覆盖率约 0.983。这里是工程 smoke test，不是下游人体重建精度结论。
- `configs/engineering_aabb_demo.yaml` 已通过一条 `run-pilot` 命令完成冻结、AABB 提取、44 个可行候选中选择 4 台、Blender 预览以及含人物/空场景各 4 张 PNG；结果在 `runs/pilot_aabb_end_to_end_v2/`。
- 当前本项目 38 项测试通过，3 项依赖可选外部条件的测试跳过；Ruff 检查通过。

## 已运行

- 本机轻量环境：`D:\westlake\YSynthetic\.venv\Scripts\python.exe`，Python 3.12.13；NumPy 2.5.0、Pydantic 2.13.4、pytest 9.1.1。
- 旧版验证时设置 `YSYNTHETIC_ROOT` 后 38 项单元/集成测试全部通过，其中 3 项直接读取真实 YSynthetic schema/loader；结果保留为历史记录。
- Ruff 格式化和静态检查通过。
- `configs/demo_sofa.yaml` 小型示例成功：从 96 个初始枚举候选中保留 81 个可行候选，固定预算选择 4 个；生成 rig、证据及诊断。输出在 `runs/demo_v1/`。
- 用 YSynthetic 真正的 `CameraRig`、`Scene` 和 `load_scene_bundle` 验证 `runs/demo_scene_v2/scene.yaml` 通过，而不仅是自己定义的 schema 通过。
- Blender 5.2.0 LTS 的受控完整工程流程在 `runs/pilot_end_to_end_v3/` 成功：冻结源场景、导出 512 顶点/988 三角形、保留 47 个可行候选、选择 4 台相机、保存预览 `.blend`，并分别保存含人物/空场景 PNG。
- `reference_import_smoke/` 又实际导入了现有 MHR `fused_mesh_world.ply`；对象保持恒等世界变换，路径和 SHA-256 写入 snapshot manifest。
- 四个选中视角各测试中心及离轴点，Blender/OpenCV 最大投影误差约 0.00006–0.00013 像素；真实 PNG、manifest 和 SHA-256 均已写入。
- 三角 BVH 的穿过/绕过/端点接触、闭合体内外及 clearance 查询已有独立单元测试。
- 真实已有 fused NPZ 与自身比较，世界关节误差和 root-aligned 误差为 0；这是指标兼容检查，不是模型质量结果。
- 显式常量场景控制记录经当前 `SceneConsistencyResult` 验证通过。测试使用合成 fixture，不是声称真实图像检测成功。

测试覆盖：Y/Z up 相机、相机中心恢复、OpenCV/Blender 轴转换、投影雅可比有限差分、遮挡/不遮挡/平行射线、180° 共线退化、固定预算及可重现性、候选不足、禁用真实人体输入、位移与姿态误差区分、输出覆盖保护、源证据修改检查、候选去重、拓扑元数据不一致、命令目录、真实下游 scene loader、场景控制假设与图像 hash。

## 验证期间发现并已修复

1. 实际 YSynthetic loader 只识别 `.yaml/.yml` 场景文件，单独验证 JSON 内容不够。改为真正导出 `scene.yaml` 并增加实际 loader 测试。
2. Blender 的 mathutils.Matrix 不支持 NumPy 式矩阵一元负号，改为先矩阵乘向量再取负，并在真实 Blender 执行验证。
3. Blender 5.2 Cycles 会把脚本参数 `--cy` 误读为 `--cycles-*` 的缩写；改为 `--principal-y`，失败记录保留在 `runs/pilot_end_to_end_v1/`，v2 随后全链成功。

`runs/demo_scene_v1` 是第 1 次失败探测留下的诊断输出，不是可用集成示例；请用 v2。没有删除旧输出来隐藏失败。

## 没有运行 / 不应宣称

- 没有对 YSynthetic 的真实 sofa 场景完成新相机的 SAM3D 逐图推理和多集合融合；本次渲染的是仓库自带受控工程场景，不是论文数据。
- 没有生成论文误差表，也没有证明几何选择优于随机。
- 没有安装新 GPU 库、下载 checkpoint、调用视频生成 API 或训练 RL。
- Blender 图像保存分支已通过受控快照 smoke test；真实研究资产仍需单独验收。
- 本次未做额外全机软件清单扫描；GPU、重环境和权重路径沿用已有配置，正式运行前仍应执行该环境的 doctor。

## 复现检查

```powershell
Set-Location D:\camera\code
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:YSYNTHETIC_ROOT = 'D:\westlake\YSynthetic'
& D:\westlake\YSynthetic\.venv\Scripts\python.exe -B -m pytest -q
& D:\westlake\YSynthetic\.venv\Scripts\python.exe -B -m ruff check src tests scripts
& D:\westlake\YSynthetic\.venv\Scripts\python.exe -B scripts/validate_ysynthetic_contract.py --ysynthetic-root D:\westlake\YSynthetic --rig D:\camera\code\runs\demo_scene_v2\cameras.json --scene D:\camera\code\runs\demo_scene_v2\scene.yaml
```

未设置 `YSYNTHETIC_ROOT` 时，3 项真实项目探测会跳过；核心测试不依赖主项目。

## 原项目保护

开始与结束查看的 YSynthetic Git status/diff stat 一致：原有 6 个 tracked 文件差异，共 258 行新增、3 行删除，另有原本的未跟踪目录/文件及子模块 dirty 状态。这里没有执行 add/commit/reset/push，也没有对其源文件调用编辑工具。

Git status 不是所有非 Git 文件的逐字节证明；能确定的是本任务的编辑目标都在 `D:\camera\code`，测试/导入使用 `-B`，没有安装或修改 YSynthetic 环境。

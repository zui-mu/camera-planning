# 从 Blender 场景到相机采集：一键操作

这份文档只描述当前代码真正能做的事情。最短结论：**只要已经有可打开的 `.blend/.usd/.usdc/.usda/.usdz/.glb/.gltf/.fbx/.obj` 场景，且能明确家具、人物与允许放相机区域的对象名，就能用一条命令完成场景冻结、三角网格规划、Blender 预览和多视角采集。**

`.hip/.hiplc/.hipnc` 不能由 Blender 读取，必须先在有许可证的 Houdini 中导出为 USD/FBX/GLB。本仓库不会绕过 Houdini 许可证。

## 1. 在 Blender 中准备对象

打开 Blender，载入场景后检查：

1. Scene Properties → Units 使用 Metric，并确认家具真实尺寸以米计。若不是，后续配置必须显式填写 `meters_per_blender_unit`。
2. 给目标家具一个稳定且唯一的对象名，例如 `Sofa`。
3. 给参考人物的所有网格命名，例如 `RefHuman_Body`、`RefHuman_Clothes`。这些对象会从规划几何中排除，并在空场景采集时隐藏。
4. 对其他封闭家具（柜子、桌体等）记录对象名，放进 `solid_objects`。墙和地面仍参加射线遮挡/表面 clearance，但不应当作“整个内部都是实心”的封闭体。
5. 新建一个 Box，覆盖允许摆放相机的空间，命名为 `CameraAllowedRegion`；它只定义允许区域，不参与渲染和遮挡。
6. 固定灯光、材质、渲染引擎和帧。保存场景。流程会再生成只读语义的冻结副本，不覆盖源文件。

## 2. 复制并修改配置

复制 `D:\camera\code\configs\engineering_demo.yaml` 到新文件。至少修改：

- `source_scene`：你的场景文件；
- `output`：每次实验使用一个全新的目录；
- `target_object`：目标家具名；
- `planning_geometry`：默认 `aabb_boxes`，自动提取对象世界 AABB；需要更精细遮挡时改为 `triangle_mesh_proxy`；
- `box_ignore_name_tokens`：AABB 模式中忽略房间壳的名称 token，必须结合对象清单核验；
- `human_objects`：人物的全部 mesh 对象名；
- `solid_objects`：除目标家具外，需做体积占用判断的闭合家具；
- `large_object_proxy_threshold_triangles`：仅用于 `triangle_mesh_proxy`，超大非目标物自动使用 AABB 规划代理；
- `target_max_triangles`：仅用于 `triangle_mesh_proxy`，目标家具在规划副本中做保形 Decimate；渲染仍使用原始高模；
- `observation`：家具表面采样数、完整构图高度/水平余量和安全画幅边缘；
- `allowed_region_object`：允许放相机的 Box；
- `floor_height_m`、相机内参、候选高度与距离。

规划阶段会自动排除 `human_objects`，因此未知未来人体不会泄漏到选点算法。若做正式 RQ1，还要添加：

```yaml
reference_npz: D:/.../fused_body_parameters.npz
reference_mesh: D:/.../fused_mesh_world.ply
reference_object_name: ReferenceHuman
reference_body_fuse_result: D:/.../body_fuse_result.json
acknowledge_reference_mesh_correspondence: true
assume_constant_scene_quality: true
random_sets: 12
integration:
  ysynthetic_root: D:/westlake/YSynthetic
  python: D:/westlake/YSynthetic/.venv-sam3d-body/Scripts/python.exe
  body_config: D:/westlake/YSynthetic/local/configs/sam3d_body.local.yaml
  fuse_config: D:/westlake/YSynthetic/local/configs/body_fuse.local.yaml
```

若提供 `reference_mesh`，流程会把世界坐标 PLY/OBJ 以恒等变换导入冻结场景，并自动把 `reference_object_name` 加入 `human_objects`。若再提供 `reference_body_fuse_result`，程序会核对 result 中的 NPZ/mesh 路径、valid 状态和 topology hash。它不会为了“看起来正确”自动平移或缩放人物。

这里的 acknowledgement 仍不是形式选项：你必须确认导入人物的姿态、尺度和世界位置与场景确实共享坐标。代码可以核对同一次 body-fuse 的文件关系，但不能证明原始场景与该重建人体从一开始就在同一世界系；无纹理 PLY 的渲染域也仍需人工验收。

## 3. 打开 PowerShell 并运行

```powershell
Set-Location D:\camera\code
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'D:\camera\code\src'
$cameraPython = 'D:\westlake\YSynthetic\.venv\Scripts\python.exe'
& $cameraPython -B -m camera_planning run-pilot --config D:\camera\code\configs\your_scene.yaml
```

程序按顺序执行：

1. 用 Blender 导入源场景并保存冻结 `snapshot/scene_snapshot.blend`；
2. 默认导出所有未排除非结构 mesh 的 evaluated 世界 AABB、对象归属、单位和哈希；可选模式才导出三角网格；
3. 对目标家具盒面和确定性构图包络采样，用盒体线段相交判断碰撞/遮挡；三角网格模式则用 BVH；
4. 选择固定预算的相机集合；
5. 保存 `preview/camera_plan_preview.blend`，其中绿色相机为选中相机、灰色为其他候选；
6. 用同一冻结场景和 K/R/T 渲染含人物图与空场景图；
7. 若配置 reference，建立 RQ1 集合、受控 scene-consistency 和下游调用计划。

任何 Blender 阶段失败都会停止，并在 `logs/` 留下 stdout/stderr；`pilot_run.json` 标记失败阶段。修正后必须换一个新输出目录，不要覆盖旧失败记录。

## 4. 人工验收

运行结束不等于研究有效。至少检查：

- 打开 `preview/camera_plan_preview.blend`，相机是否真的在房间内、没有穿墙、朝向家具交互区域；
- 打开 `captures/human/*.png` 或 `rq1/renders/human/*.png`，人物是否完整、像素尺度足够、没有离谱遮挡；
- empty 与 human 是否只差指定人物对象；
- AABB 模式检查 `geometry/scene_boxes.json` 的 target、obstacle、ignored 清单；三角模式检查 `scene_export_manifest.json`；
- `plan/camera_plan_result.json` 的 `geometry_backend` 是否为 `triangle_mesh_bvh_v1`，候选拒绝原因和分项分数是否可解释；
- 每张 `render_manifest.json` 的 `calibration_max_error_px` 是否小于 0.02。

## 5. 显式启动重模型与汇总

正式 RQ1 配置会生成 `rq1/integration_commands.json`，但不会擅自花 GPU 时间。检查后运行：

```powershell
& $cameraPython -B -m camera_planning execute-integration `
  --commands D:\camera\code\runs\your_run\rq1\integration_commands.json `
  --acknowledge-heavy
```

中断后可加 `--resume`；只有执行日志中 returncode 为 0 的阶段才会跳过。每个阶段日志保存在 `rq1/logs/`。完成后汇总：

```powershell
& $cameraPython -B -m camera_planning summarize-rq1 `
  --experiment D:\camera\code\runs\your_run\rq1
```

结果位于 `rq1/summary/rq1_summary.json` 和 `.csv`。它会同时记录布局 proxy、body-fuse 状态、名义/有效视角数与 round-trip 误差；重复相机集合只复用结果，不会冒充独立实验样本。

## 6. 当前仍需人工或外部条件的部分

- 原始 `test_sofa` 目前没有现成可渲染 `.blend/.usd`，现有 `.hip` 在本机因 Houdini 无许可证不能导出；
- reference body-fuse result、NPZ 与 PLY 的文件对应关系可自动核对；它们与目标场景的世界坐标及渲染材质仍需人工审核；
- 真实 SAM 3D Body + pose-consensus + body-fuse 尚未针对新相机集合完整跑完；
- 没有真实 RQ1 误差表，也不能宣称当前贪心布局优于随机/均匀；
- 当前规划静态相机集合，不规划连续摄影机运动轨迹；
- 三角网格遮挡精确到导出的静态 mesh，但不含未知人体自遮挡、镜头外壳体积和动态物体。

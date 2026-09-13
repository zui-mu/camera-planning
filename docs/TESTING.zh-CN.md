# 从场景输入到相机照片：完整测试指南

## 先区分两种测试

`D:\camera\code\tests` 是 Python 自动化测试，检查坐标、遮挡、选择器、文件契约等；它不会替你打开真实房间，也不能证明真实场景里的相机合理。真正的场景闭环使用 `camera_planning run-pilot`。

## 一、在 Blender 中检查输入场景

1. 打开 Blender，File → Open 载入 `.blend`；USD/USDC 也可先通过 File → Import → Universal Scene Description 检查。程序本身支持 `.blend/.usd/.usdc/.usda/.usdz/.glb/.gltf/.fbx/.obj`。
2. 在右上角 Outliner 中选中要交互的家具。按 `N` 打开侧栏检查 Dimensions；场景应采用米制，家具尺寸不能离谱。
3. 推荐另外导出一个只含待交互家具的 USD/USDC，并保持它在完整房间中的世界变换。配置把完整房间写入 `source_scene`，把单独家具写入 `target_scene`。程序先取单独资产的主 mesh 世界 AABB，再在完整场景中做 0.1 mm 容差的唯一匹配，不需要猜对象名。
4. 单独家具文件应只有一个主要 mesh；Blender 默认 Cube 可放进 `exclude_objects`。若有多个真正的家具 mesh，可用 `target_asset_object` 显式指定，或先在导出副本里 Join。旧的 `target_object` 单场景输入仍兼容，但不再是推荐方式。
5. 相机允许区域有两种输入方式：
   - 推荐测试阶段在 Blender 新建 Cube，缩放到相机允许出现的室内范围，命名 `CameraAllowedRegion`，关闭渲染可见性；配置填写 `allowed_region_object`。
   - 或完全不建 Cube，直接在配置中填写世界坐标 `allowed_min_m` 与 `allowed_max_m`。
6. 不要覆盖原始文件。一键流程会另存冻结 snapshot，并在副本里添加预览相机。

## 二、复制配置

先测试受控样例：

`D:\camera\code\configs\engineering_aabb_demo.yaml`

真实客厅模板：

`D:\camera\code\configs\living_room_sofa.yaml`

至少确认这些字段：

```yaml
source_scene: E:/.../room.blend       # 整个房间，不是单独家具文件
target_scene: E:/.../target_sofa.usdc # 单独家具，保留房间世界坐标
output: D:/camera/code/runs/my_run_001 # 每次必须是新的目录
planning_geometry: aabb_boxes
box_structure_policy: auto_geometry    # 由允许区域与包围盒关系识别房间壳
box_ignore_name_tokens: []
budget: 8
floor_height_m: 0.0
tour:
  enabled: true
  close_loop: false
```

相机与构图常用参数：

- `shell_offsets_m`：相机壳层距离；不是到家具中心的固定圆半径，而是家具盒边缘外扩距离。
- `heights_above_floor_m`：候选相机高度。
- `focus_heights_above_floor_m`：镜头注视高度。
- `min_target_distance_m/max_target_distance_m`：相机到家具盒的距离限制。
- `full_body_height_m`：为未来完整人物预留的构图高度。
- `image_margin_ratio`：观察域必须落在扣除该比例后的安全画幅内。
- `budget`：最后选几台相机。
- `intrinsics`：输出图分辨率和 K；当前 Blender renderer 要求 `fx=fy`、主点位于图像中心。

若场景当前没有人物，不填写 `human_objects`。程序会拍摄原始空场景。若场景已有参考人物，把人物的全部对象名填入：

```yaml
human_objects: [RefHuman_Body, RefHuman_Clothes]
```

这样会分别保存含人物图和隐藏这些人物后的空场景图。

## 三、运行自动化测试

打开 PowerShell：

```powershell
Set-Location D:\camera\code
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'D:\camera\code\src'
$cameraPython = 'D:\westlake\YSynthetic\.venv\Scripts\python.exe'
& $cameraPython -B -m pytest -q
```

当前预期为 `38 passed, 3 skipped`。这一步只验证代码基础能力。

## 四、一条命令跑完整场景闭环

先跑约十秒的受控 demo：

```powershell
& $cameraPython -B -m camera_planning run-pilot `
  --config D:\camera\code\configs\engineering_aabb_demo.yaml
```

该输出目录已经存在时程序会拒绝覆盖。复制配置并将 `output` 改为 `pilot_aabb_end_to_end_v2` 等新名字后再跑。

真实客厅使用：

```powershell
& $cameraPython -B -m camera_planning run-pilot `
  --config D:\camera\code\configs\living_room_sofa.yaml
```

顺序如下：

1. Blender 只读导入源场景并保存冻结 `.blend`。
2. 从单独家具提取世界 AABB，并在完整房间里唯一匹配同一家具。
3. 按几何关系自动忽略包住允许区域的整体墙壳和大面积水平薄层；其他对象作为保守 AABB 障碍。
4. 生成候选相机并做允许区域、盒体碰撞、距离、画幅、遮挡和可见率过滤。
5. 用集合目标选出固定预算相机。
6. 将选中和未选相机都写进一个新的 Blender 预览文件。
7. 把选中相机的 K/R/T 重新写回冻结的原始高模场景并逐张渲染 PNG。
8. 按家具中心方位角排序，依次经过所有选中点位，输出不闭环的 Blender 动画、PNG 帧和 MP4；同时记录采样 AABB 碰撞结果。

## 五、去哪里看相机

打开：

`runs/<run_name>/preview/camera_plan_preview.blend`

在 Blender Outliner 中展开：

```text
CameraPlanning_Preview
  Selected_Cameras       最终相机，绿色
  Unselected_Candidates  未选候选，灰色
  Planning_Diagnostics   家具盒、家具表面点和构图包络点
```

可以先关闭 `Unselected_Candidates` 的视口显示，只看最终相机。选中 `Plan_cam_xxxx` 后，可在 Item/Transform 查看 Location；按小键盘 `0` 进入当前活动相机视角。需要切换活动相机时，选中目标相机后按 `Ctrl+小键盘0`。

不打开 Blender 也能读：

- `plan/camera_placements.json`：每台最终相机的世界坐标位置和观察方向；
- `plan/cameras.json`：给 YSynthetic 的严格 K/R/T 格式；
- `plan/candidate_cameras.json`：全部可行候选；
- `plan/camera_plan_result.json`：覆盖率、拒绝原因、限制和选择过程。

## 六、去哪里看照片

没有人物对象时：

```text
runs/<run_name>/captures/selected/cam_xxxx.png
```

配置了 `human_objects` 时：

```text
runs/<run_name>/captures/human/cam_xxxx.png
runs/<run_name>/captures/empty/cam_xxxx.png
```

每个目录都有 `render_manifest.json`，记录图片路径、哈希和 Blender/OpenCV 标定误差。误差应小于 0.02 px。

运镜结果位于：

```text
runs/<run_name>/tour/camera_tour.blend
runs/<run_name>/tour/camera_tour.mp4
runs/<run_name>/tour/frames/camera_tour_0001.png
runs/<run_name>/tour/camera_tour_manifest.json
```

运镜只是逐点观察用的可视化相机运动；`collision_free_under_aabb_proxy=true` 表示采样相机中心未进入扩张 AABB，不等价于真实机械导轨可执行证明。

## 七、真实场景必须人工验收的内容

1. 打开 `geometry/target_asset_bounds.json` 和 `geometry/scene_boxes.json`，确认单独家具与完整场景目标的 `target_match_error_m` 接近 0；再核对家具在 obstacles，房间壳在 ignored。
2. 若一个植物、L 形沙发或整组散布物的 AABB 过大，它可能保守地挡掉本来可行的相机；应把该对象加入 `exclude_objects`，或切换三角网格对照验证，而不是直接相信结果。
3. 检查允许区域没有越过墙，也没有覆盖天花板外部。
4. 检查每张图中家具完整、未来人物有足够画幅、视角不是全部集中在同一侧。
5. AABB 的 `furniture_surface_coverage=1` 只表示盒面采样覆盖，不等于真实曲面 100% 可见。

## 八、当前完成边界

已经实现并验证的是：完整场景＋单独家具输入 → 目标几何自动匹配 → 多相机规划 → Blender 可视化摆放 → 原始场景逐相机渲染 → PNG/KRT/诊断 → 不闭环运镜 MP4。

尚未成为一条全自动链路的是：把空场景图和交互 prompt 自动提交给视频生成服务、把生成的人物帧可靠匹配回 camera_id、再自动启动 SAM 3D Body 和 MHR 融合并形成最终评价。这些属于 YSynthetic 上下游集成；当前 camera 项目只生成它们所需的相机与背景图，不会自动调用外部视频模型。

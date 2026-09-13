# 指定家具的自适应视点域与生成后选帧

修改日期：2026-09-13。本版通过 51 项 Python 测试、4 项跳过；尚未运行新的 Blender 导入、规划或渲染。旧 test2 图片来自上一版算法，不能作为本版验证结果。

## 任务判断与边界

新设定合理：先在已知房间里围绕家具拍摄，让视频模型加入在世界坐标中保持同一姿态的人，然后再选择用于恢复这个静态姿态的多视角帧。因此不要求多个相机同时曝光，但必须验证人物静止以及背景对应的相机参数没有改变。相机移动时人体在图像中的位置、轮廓和遮挡应随投影变化；画面中固定不动不等于世界中静止。

第一阶段不知道人体体型和动作，不能保证未来完整人体入镜、关节都可见或没有穿模。构图留白是图像约束，可为生成留空间，但不能被称为人体可行域。现有 `furniture_free_space`、`occupancy_clearance_m`、`robust_space` 和 `poor_space_direction` 不参与新模式。

上次将自由点统一解释为人体中心不准确。旧预览用 `show_in_front=True` 把点画在家具前，且旧数据的茶几 AABB 内点数实际为零，不能只凭截图认定穿模。新预览改为正常深度遮挡，并移除人体空间点云。

## 文献支持与修正

- [MVP sensor planning, 1995](https://research.ibm.com/publications/the-mvp-sensor-planning-system-for-robotic-vision-tasks)：用已知场景、光学模型和任务的成像要求建立合法相机域。这支持“先构图约束，再求距离”的组织方式。
- [Optimal Camera Placement, 2012](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2012/EECS-2012-69.html)：讨论过早离散的误差与连续相机优化。当前实现借鉴“离散多起点再连续微调”，没有复现其梯度公式。

附件中建议的所有论文结论并非都已独立验证；此实现不声称是上述算法的完整复现或新理论。特别要修正：OBB 的 PCA 轴不代表家具语义前后；遮挡/碰撞非单调，不能全程只二分一次；下方留白不保证可见地板；未知人体时家具中心夹角只是几何代理。

## 1. 输入与新模式入口

`configs/test1_sofa_clean.yaml`：

```yaml
planning_mode: target_viewspace
source_scene: D:/camera/testinput/room/export_scene.usdc
target_scene: D:/camera/testinput/sofa/SofaFactory_1935615__spawn_asset_3130031_.usdc
output: D:/camera/testoutput/test1_target_viewspace
budget: 8
planning_geometry: aabb_boxes
```

`workflow.py` 读取配置，将 `planning_mode`、`target_view` 和目标 OBB 写进正式 request。`planner.plan()` 按 mode 分流。默认未声明 mode 的旧配置仍走 legacy；新配置不用写旧 observation/scoring/adaptive。

新输出使用独立目录；已有 test1 图片、视频和 .blend 保持原样，打开旧预览不会显示新算法。`run-pilot` 不覆盖非空目录。

## 2. 从模型提取家具几何

`blender_extract_target_bounds.py` 导入独立家具，在现有世界变换和米制下提取 AABB 用于与完整房间内的网格匹配，容差沿用 0.1 mm。当前独立资产仍要求目标对应一个主要 mesh；多零件资产不会自动合并，选取策略与候选排行写入 JSON。

新增 OBB：读取 evaluated mesh 顶点并变换到世界坐标，在水平 XY 平面做协方差特征分解，Z 轴保持竖直，再沿所得轴投影所有顶点求 min/max、中心和半尺寸。水平方向近似等方差时退回世界轴。它是由真实顶点测得的竖直 PCA 包围盒，非最小体积 OBB。几何方向不被标成“沙发正面”。

`geometry/target_asset_bounds.json` 保存 OBB；`scene_boxes.json` 的 target_asset 携带该信息；request.target_obb 使用它。旧缓存没有 OBB 时明确记录 AABB fallback。

当前配置的快速碰撞仍使用实心 AABB，房间包围壳仍沿用 auto_geometry 加显式 allowed 范围。凹家具、旋转家具间隙和内墙需要后续更精细建模；不会因为用了 OBB 就宣称房间碰撞变成精确 mesh。

## 3. 图像构图与固定注视点

`target_viewspace.ViewEvaluator` 建立家具八个角点。目标注视点 = 家具中心 + 世界竖直方向 × 家具高度 × aim_height_ratio，默认 0.12。它提供少量上方构图余量，不使用人体头部位置。

相机在注视点周围按 `x = aim + rho * direction` 表示；这与附件严格从家具中心 c 出发略有区别：以固定 aim 为射线起点，沿同一射线 R 不变，使构图距离可以解析求解。方向覆盖和基线评分仍相对家具中心计算。

把八角点投影进 K/R/T 后要求：全部在近裁剪面前；左右至少 10% 留白、顶部 20%、底部 10%；家具投影宽不超过 70%、高不超过 65%；较大的那一维至少占图像 18%。这些是初始工程值，尚未通过实验调优。bottom_margin 仅代表空出的画面面积。

## 4. 每个方向自动求距离区间

相机基向量固定时，角点相机坐标可写成 `(a_x, a_y, rho + a_z)`。`fit_interval()` 将画面边界的 pinhole 不等式转成 rho 的下界，取全部角点下界最大值。使用一个充分的构图矩形，因此结果保守；它不是所有旋转/注视点下的全局最短距离。

用 slab 射线盒相交求相机所在方向与 allowed_camera_region 的进出距离。相机安全余量使 allowed 范围收缩。对 AABB backend，再从区间中减去每个膨胀障碍物的相交区间，因此一条射线可保留多个分离片段。

每段测试近端、中间、远端，再用广度优先队列二分：先探测各个独立自由区间，再逐层加密，避免先在第一个区间的近端深挖。预算允许时至少细分两层，再按合法性切换、投影边界变化、可见性变化细分，受最大深度与家具尺度归一化步长限制。目标太小的点单独拒绝。没有把可见性当成单调函数，也不声称有限测试发现了所有极窄可行区间。

单方向最多 17 个距离探测，候选生成总计最多 4096 个实际位置评估（含局部微调）。位置缓存复用相同位置的结果。预算截断不被当作“剩余位置非法”；每条方向记录 `radial_budget_truncated`。当自由区间很多或预算很低时，连所有区间的首轮探测也可能无法完成，需要读取该标记。

该模式不使用旧 shell_offsets、固定 camera heights 或 ×1.15 后退。相机离地高度和离观察点的最大距离现在都是显式硬约束。

## 5. 水平角度优先的候选搜索

现行策略为 `azimuth_first_vertical_plane_v3`。旧 test2 图片来自上一个实现，不能用于判断本版输出。

家具观察点周围的位置写成 `(azimuth, elevation, radius)`。先均匀取 24 个水平角度；固定一个水平角度后，相机只能落在它与竖直轴构成的半平面中，再搜索 5 个初始仰角及各方向上的距离。每个竖直平面对“合法性发生切换、质量高、可见率变化大或仰角间隔大”的区间做两轮中点加密。

完成 24 个水平角度后，第一轮在相邻角度之间全部补一个中点，相当于检查粗采样遗漏的角度。更多轮只在相邻端点至少一侧可行或可行性变化的位置继续补点。它能近似发现可行水平角度段，但不是连续区间的形式证明。

方向给定后，由构图、房间边界和障碍盒先计算可搜索距离段；各段按广度优先测试近端、中点、远端并有限二分。每个方向最多 17 个距离点。只保留该方向中构图最好的点和距离最近的合法点，不再保留最远点。

候选点要同时满足：离地高度在 1.0–2.4 m、离观察点不超过 3.25 m、目标完整落在构图边界内、目标可见率至少 0.90、composition 至少 0.78。composition 由可见率、目标画面占比和近距离偏好组成，其中距离项权重 0.10。方向覆盖不能补偿这些硬门槛。

候选按最初 24 个水平角度分桶。第一轮每个有合法点的桶最多贡献一个，第二轮再各贡献一个；每桶最多 6 个、全局最多 128 个。桶内选择同时看与已保留角度的距离和单张质量。合法水平范围很窄时会如实保留较少方向，不会把同一区域复制成“全方位覆盖”。少于最终预算时直接报错。

局部微调从不同水平桶选择最多 8 个种子，在世界 x/y/z 正负方向做两轮有限 pattern search。每个新位置仍需重新通过全部硬门槛。方向和位置查询分别受 600 与 4096 的总预算限制；局部微调共享位置预算。

`viewspace_search.json` 记录各竖直平面的可行性、每次仰角和距离探测、拒绝原因、预算停止原因以及各水平桶裁剪前后数量。`target_view_evidence.json` 为每个候选记录水平角、仰角、距离、离地高度、可见率、填充率、composition 和所属水平桶。这些字段用于解释结果，不改变下游 K/R/T 格式。

## 6. 遮挡检查与两级几何

候选阶段用 geometry backend 对面朝相机的目标表面采样做射线遮挡检查，以可见样本比例过滤。目标仍参与自遮挡。这个比例是表面样本代理，不能写成真实可见像素比例。AABB backend 的样本在代理盒表面，不是真实沙发曲面。

渲染阶段新增 `blender_target_visibility.MeshViewValidator`：从目标 evaluated mesh 建真实 BVH；在目标投影包围矩形的规则图像网格中发射射线；先计算单独目标命中，再用完整场景 ray_cast 比较最前交点是否为目标。该比例估计目标轮廓区域中未被其他物体遮挡的面积，仍受有限射线分辨率和透明材质近似限制。

静态机位在拍摄前全部检查；运镜每个输出帧同样检查。失败记录到 mesh_view_validation.json / trajectory_view_validation.json 并停止采集。当前没有自动把失败角度回传规划器重新搜；该闭环是明确待完善项。实际 mesh 复核主要检查构图和遮挡，相机/路径体积碰撞仍依赖代理几何；不宣称全场景精确实体碰撞已完成。

## 7. 联合选择机位与开放路径

目标函数由六项组成：水平角度覆盖、以家具中心为锚的 sin² 夹角、平均单视图构图质量、最差单张构图质量、仰角差异，以及以家具尺度归一化后的路径长度惩罚。候选已经通过硬质量门槛；集合评分不会把不合格照片重新放回来。

水平覆盖使用固定 24 个水平角度桶，只评价邻近已找到合法候选的桶；每台相机以 35° 角核覆盖它们，集合取最大覆盖。仰角差异按配置的仰角范围归一化。`worst_composition_weight` 单独奖励集合中的最低分，避免平均质量掩盖一张差图。

选择从多个分散起点开始。每增加一台，在最多 12 个候选短名单和当前路线各插入位置间比较：需要的连接使用 FreeGrid、带相机安全余量的段检查和 A* 绕障；加入新的两段并扣掉被替换的旧段长度，再计算完整集合评分。无法连接的插入被拒绝。最多 300 次连接查询、60000 个缓存沿途构图检查，达到预算可报告无法完成 K 个机位。

路径搜索边还会每隔不超过 0.15 m 检查固定 aim 下的构图与遮挡。它避免仅端点好看、中间完全背离目标。失败可能是预算或网格分辨率不足，不能直接宣称物理不可达。

## 8. 运镜与实际相机参数

沿安全折线的每一段使用 `10u³−15u⁴+6u⁵` 时间插值，位置留在线段内，不用可能切进家具的自由样条。在每个折线顶点速度与加速度归零，再进入下一段；因此可能短暂停顿，不是恒速无停顿电影轨迹。

新模式每个采样姿态都重新 look-at 固定家具 aim。相邻段端点的位置和注视方向一致，五次时间曲线降低换向突变。逐帧写 keyframe，并在渲染前检查实际 mesh 的目标可见性。连续时间的遮挡仍没有形式证明。

图片 2560×1440；运镜由同一 rig 的分辨率乘 75% 得到 1920×1080、24 FPS。先前 100% 的解释错误，已改正。`camera_trajectory.json` 输出实际视频分辨率、同比缩放的 K、每帧 R/T/time_seconds。

当前渲染要求 Blender 世界坐标以米为单位，非 1.0 unit scale 会显式拒绝。参考世界房间为静态；动画家具场景不在本版假设内。

## 9. 人物生成后筛帧

新命令 `select-human-frames`，实现在 human_frames.py。它不运行检测器或视频模型。输入来自后续模块的实际人物 bbox、统一关键点列表及置信度、track_id、帧到相机的映射、图片路径。

必须先有外部相机/背景验证报告 status=passed，记录所使用轨迹的 SHA-256 以及已经检查的 frame→camera_frame 映射。代码检查这些契约，但不会替该模块完成背景匹配；手写 passed 不构成科学验证。

检测文件结构（以下字段示意，不是可以拿去替代检测结果的模拟数据）：

```text
schema_version: generated_human_detections_v1
scene_id: 与房间一致
world_static_subject: true
track_id: 同一人物的标识
keypoint_names: 所有帧相同顺序的关节名称列表
camera_validation_report: 外部已验证相机映射的 JSON 路径
frames:
  frame: 生成视频帧号
  camera_frame: camera_trajectory.json 内对应帧号
  image: 此帧真实图片路径
  width, height: 检测时的实际图像尺寸，须与轨迹一致
  track_id: 人物标识
  bbox_xyxy: [x_min, y_min, x_max, y_max]
  keypoints: 每关节 [u, v, confidence]，像素坐标
```

筛选顺序：拒绝未经验证的相机对应、换人、尺寸不匹配、bbox 接触图像边缘、有效关节不足等帧；按时间分段在每段取高置信度帧，将候选限制到最多 96 帧。计算已知 K/R/T 下的极线误差，做置信度加权 DLT 三角化，检查正深度、实际关节视线角度与重投影误差。默认置信度 0.5、至少 6 个共同关节、最大重投影误差 4 px、最小有效角度 3°。

从多个好帧对开始扩展相互兼容集合，每加一帧都对整组拟合一个世界静态关键点集；不能只靠每对通过。凑不够指定数量，输出 insufficient_consistent_views，不用重复帧补数量。通过后输出 human_frame_selection.json 与兼容的 cameras.json，并保存静态关键点拟合诊断。

稀疏关键点一致性不能充分证明人体形状稳定、脸部身份一致、接触合理或衣服几何一致。当前尚未读取 mask、未检查轮廓漂移，也未自动执行 SAM 3D Body/MHR。静态拟合出的点仅是筛帧诊断。

## 10. 输出与下次入口

新增 `plan/viewspace_search.json`（方向、径向区间、预算），`plan/target_view_evidence.json`（OBB 角点与每候选构图/可见性），`plan/selection_diagnostics.json`（路径代价与联合选择），`plan/post_generation_contract.json`（生成后接口）。仍输出 cameras.json、camera_path.json、预览 .blend、图片、视频。evidence.npz 仅留 8 个 kind=3 的构图角点供预览，不能再交给旧 select-cached 或 prepare-rq1；这两个入口会显式报不支持。

以后由用户自行启动（本次未执行）：

```powershell
Set-Location 'D:\camera\code'
$env:PYTHONPATH = 'D:\camera\code\src'
$cameraPython = 'D:\westlake\YSynthetic\.venv\Scripts\python.exe'
& $cameraPython -B -m camera_planning run-pilot --config 'D:\camera\code\configs\test1_sofa_clean.yaml'
```

生成视频、检测和背景匹配都完成后才可使用：

```powershell
& $cameraPython -B -m camera_planning select-human-frames --trajectory 'D:\camera\testoutput\test1_target_viewspace\tour\camera_trajectory.json' --detections 'D:\camera\generated\detections.json' --budget 8 --output 'D:\camera\testoutput\human_frames_v1'
```

上条命令中的 generated/detections.json 尚不存在，由未来检测模块提供，不能现在直接执行。本次没有修改 YSynthetic 源码或执行重建。

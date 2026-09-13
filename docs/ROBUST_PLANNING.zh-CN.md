# 0.4：人物状态未知时的自适应鲁棒视点规划

研究主线：场景与家具 → 局部自由空间观测域 → 候选相机 → 方向性空间可观测性 → 平均/下尾部鲁棒选择 → 开放避障路径 → Blender 导出。代码与结果均写在 `D:/camera/code`，不修改 YSynthetic。

## 已实现的算法

1. `observation.mode: furniture_free_space`：明确声明室内 observation.region，在体素中心做占用检查，从非底部家具表面邻近自由单元传播。六邻接边检查真实几何射线，防止薄墙恰好落在两个自由中心之间而被跨越。传播距离受 neighborhood_radius_m 限制。
2. 空间点表示几何观察位置，不是人体根节点、关节分布或人体可行体积。网格宽度、观察邻域尺度均是显式任务参数；窄缝可能保留，也可能因分辨率丢失。旧 full_body_height_m 等兼容字段在新域中不参与计算。
3. 以固定种子均匀抽取自由体素中心，携带代表体积的权重。超过预算的样本不会随细化轮次变化。家具表面点只用于构图/辅助筛选与旧基线，主评分只使用 kind=2 的空间点。
4. 初始候选使用 OBB 壳层、不同距离和高度；新模式瞄准家具中心，必要时沿原方向后退，检查家具包围盒角点在带边距画幅中的比例。角点检查只证明构图，不证明可见性。
5. E&E：在已选机位附近扰动位置并重新瞄准。TUS 风格补点：从低质量且空间分散的观测点出发，沿不同方向/距离提出新机位。二者均受轮数、候选上限、提案数和最小收益约束。是借鉴思想的原型，不是所引论文的完整复现。
6. 每台相机预计算 visibility、projection Jacobian information、direction、resolution；场景实体和目标本身都参与遮挡。新轮次只计算新增机位，旧矩阵行原样复用。选择器不调用 ray casting 或 Blender。
7. 主质量：I(x)=Σ visible_i J_iᵀJ_i/σ²；取最小特征值 a=λ_min(I)*observability_length_scale_m²，再 q=a/(1+a)，至少两视角才可非零。单视角、同向或反向共线视角都可能退化。默认 length scale=0.01m 是代理测量容差，不是人体尺寸，也不是 SAM3D 的真实精度。
8. U=(1-β)×体积加权平均质量＋β×最差 α 体积的平均质量。下尾部边界样本采用部分权重，不依赖每一区域的采样点数量。默认 α=.2、β=.5，须做敏感性实验，不是文献验证过的最优参数。
9. 主方法使用多起点 greedy marginal gain＋1-swap，不使用不适用的 CELF 上界。搜索有评分次数上限；保留完整 incumbent，细化扩池不能降低已有完整选集得分。初始合法机位少于预算时，允许先用不完整池指引补点；仍不足则报错，不输出不足数量的成功结果。2-swap 尚未实现。
10. `coverage`/`lazy_coverage` 使用同一个非负加权单视图覆盖目标。只有该基线使用 Lazy Greedy。`surface_coverage` 单独优化表面覆盖；`exact` 在小规模子集数量硬上限内枚举实际配置目标。
11. 可选 `milp_coverage` 用 SciPy/HiGHS 解固定预算的普通加权覆盖，记录 solver status、gap、是否证明最优。它不是特征值＋下尾部目标的 MILP，也不是连续空间全局最优 oracle。
12. 机位选定后，A* 求连接，再优化开放访问顺序；≤12 个机位对已计算的连接代价做精确顺序 DP，否则用多起点最近邻。LOS 简化只删经过相同安全检查的拐点。不是对机位集合的精确求解。
13. 路径使用射线＋带半采样间距附加余量的距离检查，在声明的静态几何下保守检验整个线段。运镜沿该折线用 quintic 时间函数移动，每个折点停缓，避免无约束样条穿墙；朝向四元数插值。目标全程可见性尚未成为硬约束。

## 不偷换的边界

- 观察域不是身体可行域，不能保证完整人或所有动作都在域内。
- 家具和其他实体的静态遮挡可测；未来人体的自遮挡未知。
- `candidate_pool_reference` 是所有离散候选同时参与的参考上界，突破八机位预算且忽略路径约束。不是物理理论上限。绝不自动删除零质量点来抬高分数。
- 附件中的 q[i,j] 标量不足以表达多视角信息量互补；本版缓存的是每相机每点的矩阵/方向，而不是简单把单机分数相加。
- 当前内存上限约束的是证据数组估算，不是整个进程 RSS；没有实现 GPU/Embree、可微优化、磁盘跨场景自动缓存或大规模稀疏信息矩阵。
- 路径目前后处理；有可能八个机位静态得分好却无法连接。此时输出 blocked 并拒绝按该路径渲染，不自动换成穿墙插值。
- 无新网络训练、无姿态预测、无真实 RQ1 成果。几何分数提高不能解释成人体恢复质量提高。

## 运行方式

使用本项目环境，不向 YSynthetic 安装包：

```powershell
Set-Location D:\camera\code
uv sync --extra dev --extra solver
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m camera_planning plan --request configs/robust_sofa_demo.yaml --output runs/my_robust_plan
```

这是合成 AABB 几何，不渲染真实房间。每次换一个不存在的 output，程序拒绝覆盖。

复用以前已经提取的真实房间几何，不重复导入/渲染：

```powershell
.\.venv\Scripts\python.exe -m camera_planning replan-geometry --geometry-request runs/testinput_sofa_8cams_v1/geometry/request.json --pilot-config configs/testinput_sofa_robust.yaml --output runs/my_room_replan
```

`replan-geometry` 保留原几何导出的 target、obstacles、坐标、地面和 allowed_camera_region；采用新配置的观察域、内参、候选、评分、细化与路径参数。它不重新验证源 USD 是否变了；来源 hash 写入 reused_geometry.json。修改几何/允许相机区域时应重新导出，而不是误用旧缓存。

从完整场景和单独家具重新导入、摆放、拍摄并输出运镜：

```powershell
.\.venv\Scripts\python.exe -m camera_planning run-pilot --config configs/testinput_sofa_robust.yaml
```

这条是重渲染，会生成较大文件；先检查 source_scene、target_scene、Blender 路径、室内区域和独立 output。正式配置已调整为原生 2560×1440 PNG；视频按 75% 渲染为 1920×1080、24 FPS，不闭环。相机 K 随像素尺寸同步缩放以保持视野，视频导出另记录对应实际尺寸的 K。分辨率更新只是下一次运行配置，不代表已有真实房间新图片；渲染器、采样和材质仍继承场景，需要在正式拍摄时检查。仍使用 AABB proxy，不应当作精确网格实验。

### 固定候选池做基线，不重新射线或渲染

```powershell
.\.venv\Scripts\python.exe -m camera_planning select-cached --plan-dir runs/my_room_replan --method lazy_coverage --output runs/my_room_coverage
.\.venv\Scripts\python.exe -m camera_planning select-cached --plan-dir runs/my_room_replan --method surface_coverage --output runs/my_room_surface
.\.venv\Scripts\python.exe -m camera_planning select-cached --plan-dir runs/my_room_replan --method milp_coverage --output runs/my_room_milp
```

`--budget` 可覆盖选取数量；缓存检查 request/pool/evidence 的 hash。新子集不自动继承原来的运镜路径。经典覆盖与鲁棒方法比较时使用同一候选池；候选生成方法本身的消融还须控制生成/射线预算。

## 输出与检查

- cameras.json：YSynthetic 原有严格相机格式，不加研究诊断字段。
- request.json、candidate_cameras.json、evidence.npz：冻结输入、完整候选池、带 weights/kinds 的证据。
- observation_domain.json：网格/种子/域规模、估算体积及参数语义。
- observability.npz：空间点的所选集质量和候选池参考质量。
- camera_plan_result.json：主/基线分数、每轮提案和收益、拒绝原因、求解次数、未验证事项。
- camera_path.json：开放访问顺序、避障折线，或者 blocked 原因。
- Blender 预览：选中/未选相机、空间质量分档点和开放路径；点默认只在视口显示，不混进正式渲染。
- camera_trajectory.json：视频每帧的世界到相机 R/T、时间、实际分辨率对应的 K；缩放视频时同步缩放 K。

## 下游与后续实验

主链仍是 SAM 3D Body → pose-consensus → body-fuse。首先用隐藏参考人体的同一冻结姿态做回环对照，再测试视频生成。不同时间的不同人体姿态不能当作同步静态多视图直接融合。

优先补齐：网格版真实房间验证、候选预算/网格分辨率敏感性、布局与路径联动、目标连续可见性约束、多个隐藏状态的 RQ1 和失败率分析。当前配置参数是可审计的工程起点，不是完成的论文结论。

## 方法依据与实现范围

- [ICRA 2022 overlapping coverage](https://arxiv.org/abs/2203.10479)：借鉴离散候选和重叠覆盖问题；本版普通覆盖 MILP 不是其全部实验复现。
- [ICCV 2017 trajectory coverage](https://graphics.stanford.edu/papers/aerial_scanning/)：借鉴覆盖与移动成本联系，不沿用其理论保证。
- [用户提供的 2026 自适应采样工作](https://www.sciencedirect.com/science/article/pii/S0305054826001590)：按 E&E/TUS 思想实现局部与薄弱区域提案，不宣称复现其求解器或达到其采样比例。

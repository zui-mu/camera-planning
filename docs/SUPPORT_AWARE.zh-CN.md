# 目标附属物体与自由交互区域

`target_viewspace` 在 AABB 场景代理下支持三类对象：

1. `target`：要交互、要保持可识别的目标家具；
2. `supported_objects`：几何上落在目标家具顶面的台灯、显示器等附属物体；
3. `obstacles`：墙、柜子、茶几等其他场景物体。

附属物体并没有被设成透明。它们对相机位置、运镜路径和未知人体区域的射线仍是实体；只有在评价“家具本体是否还可识别”时，单纯由附属物体造成的遮挡会按较小惩罚计入。

## 1. 支撑关系检测

对每个非目标物体，计算物体底面与目标 AABB 顶面的垂直间隙、物体水平 footprint 被目标水平 footprint 覆盖的比例，以及二者 footprint 面积之比。三项同时满足配置阈值时，才记作 `supported_by_target_top`。

结果写入 `geometry/support_graph.json`。这是可审计的几何代理，不是语义分类器；复杂、倾斜或凹形桌面后续应再增加网格向下射线确认。

## 2. 自由支撑区和未知人体代理

程序在目标 OBB 顶面建立确定性二维网格。每个网格中心是一条候选人体空间柱，并沿世界向上方向取若干高度样本。只要柱内任一样本进入任何家具经过安全距离膨胀后的 AABB，该锚点就记为占用。

剩余锚点上方的样本构成 `support_surface_free_volume_v1`。它只表达“这里存在可供未来交互使用的自由空间”，不预测人物身份、体型或具体动作。

结果写入 `geometry/interaction_region.json`。Blender 预览中显示：

- 绿色：自由支撑锚点；
- 红色：被附属物体或其他几何占用的锚点；
- 蓝色：未知人体交互体积样本；
- 紫色线框：自动识别的附属物体 AABB。

## 3. 单相机评分

启用 support-aware 模式后：

- 未知人体交互区域可见率：主项，权重 `0.45`；
- 目标家具可识别度：权重 `0.25`；
- 构图与投影大小：权重 `0.30`；
- 距离偏好仍由 `distance_preference_weight` 单独混合。

家具可识别度中，附属物体造成的遮挡只承担 `supported_target_occlusion_penalty`，其他障碍物和家具自身遮挡仍正常计入。

未知人体区域的射线不会忽略附属物体。台灯占据的位置已经从自由区域删除；台灯若继续遮挡其他自由人体样本，仍会降低该机位分数。

## 4. 相机集合评分

最终多机位集合先满足单相机硬门槛和相互分离门槛，再优化水平方向覆盖、视线基线、平均/最差单相机质量、高度多样性、自由交互样本单视角覆盖和双视角覆盖，以及软距离偏好。

因此不会因为一个视角能看见很多桌面，却看不清未来人体自由区域，就被误认为高质量机位。

## 5. 主要配置

所有参数位于 YAML 的 `target_view`：

```yaml
support_detection_enabled: true
support_max_vertical_gap_m: 0.035
support_max_vertical_penetration_m: 0.015
support_min_footprint_overlap_fraction: 0.50
support_max_footprint_area_ratio: 0.80
supported_target_occlusion_penalty: 0.15
interaction_anchor_count: 64
interaction_height_offsets_m: [0.15, 0.50, 0.90, 1.30]
interaction_clearance_m: 0.08
min_interaction_visibility_fraction: 0.55
```

如果没有检测到附属物体，程序自动回到原来的扩展 capture box 逻辑，既有沙发实验不会因桌面专用代理被重新解释。

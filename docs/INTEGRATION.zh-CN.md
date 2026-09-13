# 与现有 YSynthetic 的对接契约

核对基线：`D:\westlake\YSynthetic`，Git HEAD `44585b45603446ad5ef8a45e3691ea68f5fbb4b2`，2026-09-07。这个工作区原本有未提交修改，因此 HEAD 不是完整工作树版本；正式实验还需记录 diff/hash。这里没有修改该仓库。

## 1. 主链路以当前代码为准

核对文件：

- `src/ysynthetic/schemas.py`：Camera、CameraRig、Scene、SceneConsistencyResult 等。
- `src/ysynthetic/config.py`：场景后缀、相对路径、scene_id 和相机链接验证。
- `src/ysynthetic/cli.py`：body-fit-batch、pose-consensus、body-fuse 实际参数。
- `src/ysynthetic/modules/body_fuse/io.py`：融合参数文件字段。
- `src/ysynthetic/modules/body_fuse/coordinates.py`：相机约定、相机多样性权重。
- `src/ysynthetic/modules/pose_consensus/profile.py`：MHR 关节命名。
- `configs/models/body_fuse.yaml`：当前 `mhr_bundle_adjust`；至少 2 个视角、推荐 3 个；silhouette 默认关闭。

默认 body-fit CLI 后端是 `stub`。研究运行必须显式指定 `--backend sam3d_body`，否则“命令成功”可能只是占位链路。当前相机研究不使用 `multi-view-body` 的旧 MAMMA 分支。

## 2. 上游给我们什么

第一版输入是已经世界化的米制几何，见 `configs/demo_sofa.yaml`。真实集成需要提供场景网格及对象世界变换、目标家具 ID、地面/up 方向、允许相机区域、固定 K。

当前可用 Blender 自动导入 `.blend/.usd/.usdc/.usda/.usdz/.glb/.gltf/.fbx/.obj`，保存冻结快照并导出 evaluated 世界三角网格；但自动导入不能替人判断单位、语义、椅背/座面或可交互方向，真实实验仍必须核对。`.hip` 必须先由有许可证的 Houdini 导出。AABB 代理只保留为基线；真实入口不会把整间空房外包围盒当成实心障碍。

后续建议用 Blender evaluated mesh 或独立 USD 适配器输出统一世界三角网格；世界变换只应用一次。geometry backend 负责 ray casting 和占用查询，不能在核心规划代码中硬编码某一类场景路径。

## 3. 我们输出给上游渲染/下游恢复什么

```json
{
  "scene_id": "camera_demo_sofa",
  "cameras": [{
    "camera_id": "cam_0000",
    "width": 1024,
    "height": 1024,
    "K": [[900, 0, 512], [0, 900, 512], [0, 0, 1]],
    "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "T": [0, 0, 0],
    "coordinate_convention": "world_to_camera"
  }]
}
```

上面只是字段示例，不是推荐相机位置。图像名为 `cam_0000.png`，view_id=camera_id。现有 `body-fit-batch` 正是按 camera_id 找图片和输出目录，因此初版不用不同的 view_id 再映射一次。

CameraRig 禁止额外字段，故 up_axis、单位、分数、候选来源、实验 set_id 放在 sidecar，不写进 rig。`scene.yaml` 同时链接 `empty_views` 和 `inpaint` 两类图片。这里的 `inpaint` 是现有 artifact 名称，在 RQ1 中里面可以是受控渲染图，不代表调用过 inpainting。

场景文件必须使用 `.yaml` 或 `.yml`。实际 loader 并不把 `.json` 当作显式场景配置，即使其中内容可被 YAML 解析。这一点已通过实际加载检查修正。

## 4. 坐标与图像：不能依靠肉眼大致对齐

世界到 OpenCV 相机：`Xc = R Xw + T`；相机中心 `C = -Rᵀ T`。

相机坐标是 x 向右、y 向下、z 向前。世界可以 Y-up，也可以 Z-up，但必须在请求/manifest 中明确。K 的单位是像素；世界位移/网格使用米。

Blender camera-to-world 的旋转为 `Rᵀ diag(1,-1,-1)`，平移为 C。这是转换相机坐标轴，不是把整个世界再旋转一次。

本版 Blender 脚本只支持 fx=fy、零 skew、中心主点的 K。不支持时会报错，不会悄悄把 K 改成“差不多”。输出固定分辨率，关闭会变换图像的合成器和 sequencer；每台相机先校验离轴点的 Blender 投影与 OpenCV 投影。

任何后续裁剪/缩放都需要同步修改 K。不要只改 png 尺寸。参考 mesh、joints 和渲染场景不能分别居中或缩放；用于比较的 world 也必须完全一致。

## 5. 重建输出与目录的真实区别

| 入口 | 外部传入 output-dir=P 时的相关结果 |
| --- | --- |
| `body-fit-batch` | `P/<camera_id>/body_fit_result.json`，同目录的人体参数/mesh |
| `body-fit` 场景 workflow | 通常在 `P/body_fit/<view_id>/...`，不能与 batch 混用 |
| `pose-consensus` | `P/pose_consensus/...` |
| `body-fuse` | `P/body_fuse/fused_body_parameters.npz` 等 |

本框架把 pool 的 batch 输出直接放在 `experiment/body_fit_pool/<id>/...`，后续 `--body-fit-dir` 指向 `body_fit_pool`，不能再多拼一层 `body_fit`。

相机子集共用一个 scene_id，单独保存 set_id；否则既有 body-fit 和 scene_consistency 的 scene_id 检查会失败。每个子集独立执行 pose_consensus 和 body_fuse，因为换了集合，其共识、相机权重和优化也会改变。

所有 `--scene`、`--config` 和输出路径使用绝对路径。`--project-root D:\westlake\YSynthetic` 仅供现有模块定位默认配置/模型仓库；所有结果明确定向到 `D:\camera\code\runs`。

## 6. scene_consistency 不是可随意省略的输入

pose_consensus 会读 `scene_consistency_pool/<id>/scene_consistency_result.json`。status 不可用、r_scene 缺失或太低都可能排除该视角。

RQ1 有两个合法但不同的协议：

**A. 隔离相机布局影响。** 同一冻结 .blend 快照分别显示/隐藏 reference mesh 渲染，明确采用 `r_scene=1` 常量控制，避免场景评分算法增加混杂。运行 `prepare-controls --acknowledge-assumption` 前必须校验两组 render manifest 的快照、KRT、帧、图像 hash，以及只隐藏指定人体对象。导出的 `backend=precomputed` 不表示分数被测量；metadata/warnings 明确注明是实验假设。它仍不能自动证明 .blend 中的人体就是 reference NPZ，需要人工/后续自动顶点验证。

**B. 完整场景质量链路。** 真实运行现有 scene_consistency，保存其模型配置、mask 来源、对象引用和分数。在 RQ3 生成视频实验中必须走真实质量检测，不能用协议 A 的控制文件凑齐输入。

不要只写一个所有 view 都成功的 stub JSON，再把结果当作验证过的 scene consistency。本版提供控制入口正是为了把“做了假设”和“实际测到了什么”分清楚。

## 7. MHR artifact 与指标

当前 fused NPZ 格式是 `mhr_fused_body_parameters_v1`，profile `mhr_127_v1`。

| 字段 | 当前维度/含义 |
| --- | --- |
| shape_params / scale_params | 45 / 28 |
| pose_continuous / hand_pose_params / expr_params | 260 / 108 / 72 |
| mhr_model_params | 204；不要与 continuous pose 当成同一个数组 |
| root_rotation_world / root_translation_world | 3×3 / 3 |
| local_rotations / joints_world | 127×3×3 / 127×3 |
| length_unit | meter |
| mhr_asset_hash / checkpoint_hash / topology_hash / camera_rig_hash | 版本和来源身份 |

MHR 第 0 个 joint 是 `body_world`，第 1 个才是 `root`。本版 root-aligned 指标使用 index=1，默认平均 indices 1..126 并注明包括手/辅助关节；论文应再注册 body-only 关节列表，不能把“所有 127 点平均”当作标准 benchmark body MPJPE。

评估要求 reference 和 prediction 的 MHR、拓扑、checkpoint 等兼容，使用同一长度单位。相机集合本来就在改变，所以不要求 reference/prediction 的 camera_rig_hash 相等。比较顶点时还要检查实际 faces/order，不是仅凭顶点数一样。

当前 `mhr_bundle_adjust` 的相机多样性权重由相机光轴分布构造，存在于下游。RQ1 得到的是“相机布局对整套后端的总效应”，不能全解释为三角化条件变好。若进一步拆因果，应在未来获得修改下游授权后单独做相机权重冻结消融，而不是这次悄悄改融合代码。

## 8. 缓存、生成视频和并入策略

pool 图像和单视角人体提案理论上可复用，但至少要比对场景快照、reference、KRT/图像规格、实际图片、mask/bbox/prompt、checkpoint、有效配置、代码版本和随机种子。当前只提供元数据和检查部分规划文件的 hash，没有实现完整安全的推理缓存系统；不得因为 camera_id 相同就盲用旧结果。

生成视频可能漂移相机、改变背景、改变人体姿态。匹配到同一相机视角不代表是同一时刻或同一姿态；pose_consensus 是筛选/降权，不是时序同步器。未来集成需要将匹配可用率和接受视角数反馈到实验，不应把帧号直接当作相机号。

并入步骤：先固定 artifact 契约并通过真实场景 smoke test；再建立上游几何适配器；然后薄包装成 YSynthetic 的独立模块；最后才讨论是否添加 CLI。这个仓库当前不修改任何下游源码。

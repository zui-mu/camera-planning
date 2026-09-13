# 文献依据与初版取舍

核对日期：2026-09-13。本次读了你的 `D:\camera\paper\构思.md` 和文献列表，并对以下论文/作者页面做定向核查。这不是“已读完所有全文”的声明，也不是新颖性检索证明。程序是根据研究需求编写的原型，没有把相关论文代码复制后冒充完整复现。

## 最应优先理解的几组工作

| 工作、一手入口 | 为什么相关 | 本项目采用什么 / 不能直接套什么 | 优先级 |
| --- | --- | --- | --- |
| Chen & Davis, *Camera Placement Considering Occlusion for Robust Motion Capture*, Stanford technical report 2000，[作者页面](https://graphics.stanford.edu/papers/OcclusionMetric/) | 直接讨论相机布局、成像分辨率与遮挡的关系 | 用作问题背景；当前尚未实现其概率人体自遮挡模型 | 精读，优先于泛 NeRF 论文 |
| Dai & Baumgartner, *Optimal Camera Configuration for Large-Scale Motion Capture Systems*, BMVC 2023，[会议页面](https://proceedings.bmvc2023.org/448/) | 固定相机配置、几何/光学约束与遮挡建模 | 学习约束和基线设计；标记点动捕目标不同于未知人体单图网络融合 | 精读几何建模，按需复现约束基线 |
| Malhotra et al., *Optimizing Camera Placements for Overlapped Coverage with 3D Camera Projections*, ICRA 2022，[论文](https://arxiv.org/abs/2203.10479) | 用针孔相机、三维候选姿态和射线遮挡构造有限候选池，再做固定预算集合选择 | 采用“光学/位置约束先过滤、离散选择在后”的分层思想；其屋顶规则网格和商店体素覆盖目标不直接套用 | 精读相机视图计算与候选约束；作为离散覆盖基线 |
| *ActiveMoCap: Optimized Viewpoint Selection for Active Human Motion Capture*, CVPR 2020，[论文](https://arxiv.org/abs/1912.08568) | 相机选择与人体恢复误差最直接的相关工作之一 | 学习用恢复不确定性评价相机；其已观察人体的主动时序设定不能直接当作本项目无人体输入算法 | 必须精读，复现思想而非立即搬整个系统 |
| *Onboard View Planning of a Flying Camera for High Fidelity 3D Reconstruction of a Moving Actor*, 2023，[论文](https://arxiv.org/abs/2308.00134) | 简单几何和成像质量代理用于人物视角规划 | 启发像素密度项；当前 fx·fy/z² 不含面法线，也没有该工作的已知运动演员信息 | 精读评分部分，小组件复现 |
| *FisherRF*, ECCV 2024，[作者项目](https://jiangwenpl.github.io/FisherRF/) | 信息量与视角价值的联系 | 帮助辨清后端信息量与几何量；本版是三维点投影 Fisher，不是 FisherRF，也没有计算 SAM 网络参数不确定性 | 精读 formulation，暂不复现整个 RF 系统 |
| *Coverage Optimization for Camera View Selection* / COVER, CVPR 2026，[作者项目](https://chengine.github.io/nbv_gym/) | 新近、轻量、可解释的覆盖型选择指标 | 启发“补充欠观测区域”的集合收益；原工作有 RF/高斯场景与增量训练，不等同于本项目家具盒面/构图包络覆盖 | 近期重点精读；可做思想对应基线，不写成完整复现 |
| *Hestia: Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction*, WACV 2026，[官方代码](https://github.com/johnnylu305/Hestia-NBV) | 新近的体素面感知、层次化 NBV | 后续候选粗细分层与覆盖表示参考；原代码涉及 IsaacLab/IsaacSim 和 RL，当前不需要以它作为启动依赖 | 第二批；先读，暂不整套复现 |
| Krause et al., *Near-Optimal Sensor Placements in Gaussian Processes*, JMLR 2008，[原文入口](https://jmlr.org/papers/v9/krause08a.html) | 传感器集合选择与次模优化基础 | 帮助分析纯覆盖/logdet 子问题；本文组合目标的最弱区域和配对项不能直接继承理论保证 | 读理论假设与适用边界 |
| *SAM 3D Body: Robust Full-Body Human Mesh Recovery*, 2026，[论文](https://arxiv.org/abs/2602.15989)；[MHR 官方代码](https://github.com/facebookresearch/MHR) | 决定实际恢复对象和后端行为 | 固定单图估计、MHR 表示和提示输入；本地 pose_consensus/body_fuse 不是官方论文给定的多视角黑箱 | 必须读且优先核对当前代码 |

Hestia 的官方说明列出了 IsaacSim/IsaacLab 环境；因此本版采用普通 NumPy 核心＋可选 Blender，不要求你为了验证 RQ1 先搭建其训练栈。[环境依据](https://github.com/johnnylu305/Hestia-NBV)

## 初版算法与论文的关系

本版不是把任意一篇论文改一个后端就形成“我们的贡献”。相机候选、射线可见性、覆盖、投影雅可比和贪心搜索本身均是成熟工具。现阶段它们的价值是构成可复现实验基线。

2026-09-13 的构图壳层实现依据标准针孔投影 `u=fx Xc/Zc+cx, v=fy Yc/Zc+cy`，把 capture-box 角点的画面边界约束转成方向相关近界，并用一维有界求解得到最小投影尺寸对应的远界。文献只支持“先形成满足光学/几何约束的有限视点区域，再做遮挡与集合选择”这一设计原则；方向相关壳层公式是本仓库针对固定 look-at capture box 的直接推导，不应写成复现了某篇论文的专有公式。

值得进一步研究的差异包括：规划发生在人物出现前，不假设站/坐/躺位置概率；相机同时覆盖家具表面和确定性完整构图包络；实际效果由 SAM 单图提案与项目融合共同决定；相机还会经过生成与匹配环节，几何最佳不必然是最终最佳。

这些差异是目前的研究假设，不是已被证明的新颖贡献。尤其不能宣称“第一次研究 camera placement 对人体重建的影响”。

## 推荐复现顺序与停止条件

1. 先复现本仓库的简单基线和坐标/遮挡测试，确认与 YSynthetic 契约一致。
2. 将一个真实场景和冻结 reference 接进 RQ1，建立可信的回环差异分布。
3. 逐项消融 coverage、parallax、resolution、information 和区域均衡；与随机/均匀基线比较。若复杂评分不胜简单覆盖，不为追新而保留。
4. 选择一篇直接的人体视角规划工作，将其可适配评分放进共同候选池；明确缺少其人体输入时的改造，不把改造版称为严格复现。
5. COVER/FisherRF/Hestia 的完整复现，仅当研究要比较其原本 RF/主动获取任务或能说明公平适配条件时再做。否则优先花算力跑当前下游，而不是训练无关的大系统。

后续学习评分器的训练标签应来自训练场景的真实回环实验，验证集选超参数；测试场景/人物不参与 ROI 和权重选择。评估内容包含排序相关性、选点收益、失败率和推理开销。单凭一张漂亮相机分布图不能支持论文结论。

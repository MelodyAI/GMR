# GMR 动作后处理工具说明

这个工具用于处理 GMR 生成的标准 `robot_motion.pkl`，目标是让预览视频里的 ELF3 动作更稳定、更顺眼。

它不是训练模型，也不是重新做 retarget。它是在 GMR 已经生成机器人动作之后，对数据做一次后处理：

```text
robot_motion.pkl
  -> motion_postprocess.py
  -> motion_foot.pkl / preview_foot.mp4 / quality_foot.json
```

## 当前推荐用法

推荐使用 `soft + v2_foot`，这是目前的综合版本：

```bash
PYTHONNOUSERSITE=1 conda run -n gvhmr python tools/motion_postprocess.py optimize \
  --input runtime/jobs/吉利-精武门3_163f1163/robot_motion.pkl \
  --robot elf3 \
  --profile soft \
  --pipeline v2_foot \
  --render
```

默认会在输入文件同目录生成：

```text
motion_foot.pkl
preview_foot.mp4
quality_foot.json
```

如果不想生成视频，只想生成优化后的 pkl 和质量报告，可以去掉 `--render`：

```bash
PYTHONNOUSERSITE=1 conda run -n gvhmr python tools/motion_postprocess.py optimize \
  --input runtime/jobs/xxx/robot_motion.pkl \
  --robot elf3 \
  --profile soft \
  --pipeline v2_foot
```

## 只做质量诊断

如果只想看一个动作抖不抖、脚滑不滑，不改数据，可以用 `quality`：

```bash
PYTHONNOUSERSITE=1 conda run -n gvhmr python tools/motion_postprocess.py quality \
  --input runtime/jobs/xxx/robot_motion.pkl \
  --robot elf3
```

默认输出：

```text
motion_quality.json
```

质量报告里重点看这些字段：

```text
dof_velocity / dof_acceleration / dof_jerk
root_velocity / root_acceleration / root_jerk
contact.estimated_foot_sliding_speed
contact.estimated_ground_penetration_depth
spike_count_total
```

简单理解：

```text
dof_acceleration / dof_jerk 越大，关节越容易突然抽动
root_acceleration / root_jerk 越大，身体整体位置越容易抖
estimated_foot_sliding_speed 越大，脚底贴地时越容易滑
estimated_ground_penetration_depth 大于 0，说明可能有脚穿地
```

## 做了哪些优化

### 1. 关节限速和平滑

工具会读取 `dof_pos`，也就是机器人每个关节的角度序列。

它会根据关节类型设置不同阈值：

```text
肩、肘、腕：重点压制上肢抽动
髋、膝、踝：做基础平滑，避免下肢突然跳变
腰部：更保守，避免身体姿态过度晃动
```

处理指标主要是：

```text
速度 velocity
加速度 acceleration
jerk，也就是加速度变化率
```

肉眼看到的“胳膊突然抽一下”，通常对应 `acceleration` 或 `jerk` 的尖峰。

### 2. 四元数符号修正

`root_rot` 是根节点旋转，使用四元数。

四元数有一个特点：`q` 和 `-q` 表示同一个旋转，但数值上会突然跳变。工具会先统一四元数符号方向，再做平滑，避免根节点旋转出现假的跳变。

### 3. 关节范围裁剪

工具会从 MuJoCo XML 里读取 ELF3 的关节范围，然后把优化后的 `dof_pos` clip 到合法范围内。

这样可以避免后处理把关节角度推到模型限制外。

### 4. 脚底高度保护

普通平滑可能会把 root 稍微压低，导致脚更容易穿地。

工具会估计优化前后的最低脚底高度，如果优化后更低，会整体抬高 root Z，避免凭空引入更严重的脚穿地。

### 5. Foot-lock 脚滑补偿

`v2_foot` 会在 `v2` 的基础上再做一层轻量 foot-lock。

它会读取 ELF3 MuJoCo XML 里的真实 foot collision geom，例如：

```text
l_foot*_collision
r_foot*_collision
```

然后通过 MuJoCo 正运动学计算每一帧脚底 collision geom 的世界坐标。

当检测到某只脚处于支撑段时，它会估计这只脚在地面上的 XY 漂移，再反向微调 `root_pos[:, :2]`，也就是机器人的整体地面位置。

注意：

```text
foot-lock 不直接改脚踝、膝盖、髋关节角度
foot-lock 不改上身关节
foot-lock 只额外修 root XY
```

所以当前推荐版本可以理解成：

```text
motion_foot.pkl = v2 上身/关节平滑 + foot-lock 脚滑补偿
```

## Pipeline 怎么选

### `v2_foot`

当前推荐。

```bash
--profile soft --pipeline v2_foot
```

它包含：

```text
v2 关节平滑
肩、肘、腕额外去抖
foot collision geom 脚滑检测
root XY 轻量脚滑补偿
```

适合生成最终预览文件：

```text
motion_foot.pkl
preview_foot.mp4
quality_foot.json
```

### `v2`

只做关节平滑，不做 foot-lock。

```bash
--profile soft --pipeline v2
```

默认输出：

```text
motion_soft.pkl
preview_soft.mp4
quality_soft.json
```

如果 foot-lock 导致整体位置轻微晃动，可以用这个版本做对比。

### `legacy`

旧版全身平滑逻辑，主要用于回归对比。

一般不推荐作为最终结果。

## Profile 怎么选

### `soft`

当前推荐。

它比 `preview` 更柔和，对舞蹈动作和手臂抽动更友好。

### `preview`

更保留原动作，平滑力度较轻。

适合想尽量少改原始数据时使用。

### `strict`

更强限速和限加速度。

适合检查异常动作，但可能让动作变软、节奏变钝。

## 指定输出文件名

默认短文件名已经够用。如果你想自己指定路径，可以这样：

```bash
PYTHONNOUSERSITE=1 conda run -n gvhmr python tools/motion_postprocess.py optimize \
  --input runtime/jobs/xxx/robot_motion.pkl \
  --robot elf3 \
  --profile soft \
  --pipeline v2_foot \
  --output runtime/jobs/xxx/my_motion.pkl \
  --quality-json runtime/jobs/xxx/my_quality.json \
  --video-output runtime/jobs/xxx/my_preview.mp4 \
  --render
```

## 兼容旧命令

旧脚本 `tools/optimize_robot_motion.py` 还保留着，但它只是 wrapper。

新功能建议直接使用：

```bash
tools/motion_postprocess.py
```

## 输出文件含义

### `motion_foot.pkl`

优化后的机器人动作数据。

字段保持 GMR 标准结构，仍然包含：

```text
fps
root_pos
root_rot
dof_pos
local_body_pos
link_body_list
```

### `preview_foot.mp4`

优化后动作的 MuJoCo 预览视频。

注意：这个视频是预览用，不代表真机安全。

### `quality_foot.json`

质量报告，记录优化前后指标、脚滑指标、foot-lock 修正量等信息。

其中 `foot_lock` 字段会记录：

```text
corrected_frames
contact_segments_used
mean_root_xy_correction_m
max_root_xy_correction_m
p95_root_xy_correction_m
```

这些字段可以用来判断 foot-lock 是否太激进。

## 当前限制

这个工具目前是预览级后处理，不保证可以直接上真机。

它能缓解：

```text
手臂抽动
关节突变
轻微 root 抖动
轻微脚滑
轻微脚穿地
```

它不能彻底解决：

```text
IK 目标绑定错误
机器人关节轴定义不匹配
动作本身严重缺帧或跳变
脚底接触物理不合理
真机动力学和稳定性问题
```

如果要上真机，还需要单独做控制侧滤波、动力学约束、速度/力矩限制和安全检查。

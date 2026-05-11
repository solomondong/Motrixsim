# Controller Demo

## Overview

当前项目使用统一控制器 `shelves-demo.controller.RobotController`。

它支持 3 种模式：

- 底盘控制模式 `base`
- 双臂控制模式 `dual_arm`
- 单臂控制模式 `single_arm`

对应的演示脚本是：`scripts/controller_demo.py`

## Demo Modes

### 1. Base Demo

底盘模式会随机采样世界坐标目标 `(x, y, heading)`，然后执行：

1. 先转向路径方向
2. 再向目标点移动
3. 最后转到目标朝向

运行命令：

```bash
python scripts/controller_demo.py --mode base
```

### 2. Dual-Arm Demo

双臂模式会基于当前左右末端位姿，在给定范围内随机采样新的左右目标位姿，然后调用统一控制器同时规划两只手。

运行命令：

```bash
python scripts/controller_demo.py --mode dual_arm
```

也可以调整采样范围：

```bash
python scripts/controller_demo.py --mode dual_arm --range-x 0.10 --range-y 0.08 --range-z 0.10
```

### 3. Single-Arm Demo

单臂模式每次只规划一只手，另一只手保持当前状态。

支持三种臂选择方式：

- `--arm left`
- `--arm right`
- `--arm alternate`

默认是左右交替。

运行命令：

```bash
python scripts/controller_demo.py --mode single_arm
```

固定左臂：

```bash
python scripts/controller_demo.py --mode single_arm --arm left
```

固定右臂：

```bash
python scripts/controller_demo.py --mode single_arm --arm right
```

## Script Parameters

`scripts/controller_demo.py` 支持这些参数：

- `--mode {base,dual_arm,single_arm}`
- `--arm {left,right,alternate}`
- `--model MODEL`
- `--max-loop MAX_LOOP`
- `--range-x RANGE_X`
- `--range-y RANGE_Y`
- `--range-z RANGE_Z`

示例：

```bash
python scripts/controller_demo.py --mode single_arm --arm alternate --max-loop 10 --range-x 0.12 --range-y 0.10 --range-z 0.08
```

## RobotController API

统一控制器入口：

```python
from shelves_demo.controller import RobotController
```

核心接口：

- `RobotController(model)`
- `initialize(data)`
- `status()`
- `update(data)`
- `stop(data, error=None)`
- `move_base_world(data, x, y, heading)`
- `move_dual_arm(data, left_target, right_target, constrain_orientation=False)`
- `move_single_arm(data, arm, target_pose, constrain_orientation=False)`
- `get_base_pose(data)`
- `get_end_effector_pose(data, arm)`
- `open_gripper(data, arm)`
- `close_gripper(data, arm)`
- `set_gripper(data, arm, opening)`

## Coordinate Convention

当前控制器内部采用如下约定：

- 世界坐标系：`+Z` 向上
- 机器人工作前向：局部 `-Y`
- MuJoCo 刚体 `body yaw` 是围绕局部 `+X` 参考出来的平面角
- 对外暴露的 `heading` 表示“机器人正面朝向”的世界平面角

因此：

- `move_base_world(data, x, y, heading)` 里的 `heading` 不是简单的 body 原始 yaw
- 这层换算已经封装在 `RobotController` 内部

如果只想调用接口，不需要自己再做 `pi / 2` 偏移。

## Pose Format

机械臂目标位姿统一使用长度为 7 的数组：

```python
[x, y, z, qx, qy, qz, qw]
```

其中：

- `x, y, z` 是世界坐标位置
- `qx, qy, qz, qw` 是四元数

## Minimal Usage Example

下面是一个最小使用示例：

```python
from motrixsim import SceneData, load_model, step
from shelves_demo.controller import RobotController, RobotPhase

model = load_model("xmls/clean_world.xml")
data = SceneData(model)
controller = RobotController(model)

controller.initialize(data)

for _ in range(400):
    step(model, data)

# 1. Base move
controller.move_base_world(data, x=1.0, y=0.5, heading=0.0)
while True:
    status = controller.update(data)
    step(model, data)
    if status.phase == RobotPhase.DONE:
        break
    if status.phase == RobotPhase.ERROR:
        raise RuntimeError(status.error)

# 2. Single-arm move
left_pose = controller.get_end_effector_pose(data, "left")
left_target = left_pose.copy()
left_target[0] += 0.05
left_target[2] += 0.05

controller.move_single_arm(data, arm="left", target_pose=left_target, constrain_orientation=False)
while True:
    status = controller.update(data)
    step(model, data)
    if status.phase == RobotPhase.DONE:
        break
    if status.phase == RobotPhase.ERROR:
        raise RuntimeError(status.error)
```

## Status Meaning

`controller.status()` 和 `controller.update()` 返回 `RobotStatus`。

主要字段：

- `mode`: 当前模式，可能是 `base` / `dual_arm` / `single_arm`
- `phase`: 当前阶段，可能是 `idle` / `running` / `done` / `error`
- `active`: 当前是否还有正在执行的任务
- `detail`: 更细粒度状态，例如底盘的内部阶段
- `error`: 失败原因
- `plan`: 当前活动规划

## Notes

- 调用运动接口前必须先执行 `initialize(data)`
- 每次物理循环都要先 `controller.update(data)`，再 `step(model, data)`
- 如果切换到底盘模式，控制器会保持上半身当前状态
- 如果切换到机械臂模式，控制器会先把底盘轮子命令清零
- 单臂模式下，未活动那只手不会主动跟随新目标

## 双臂 Home Position

| 关节 | 左臂 (rad) | 右臂 (rad) | 说明 |
|------|------------|------------|------|
| J1 | 0.0 | 0.0 | 基座旋转 |
| J2 | -1.25 | 1.25 | 肩部俯仰 |
| J3 | 0.0 | 0.0 | 肩部旋转 |
| J4 | -2.35 | 2.35 | 肘部弯曲 |
| J5 | -1.57 | 1.57 | 前臂旋转 |
| J6 | -1.57 | 1.57 | 腕部俯仰 |
| J7 | -1.93 | 1.93 | 腕部旋转 |

```python
left_home = np.array([0.0, -1.25, 0.0, -2.35, -1.57, -1.57, -1.93])
right_home = np.array([0.0, 1.25, 0.0, 2.35, 1.57, 1.57, 1.93])
```

## 升降台 Home Position

| 参数 | 值 |
|------|-----|
| platform_home | 0.5 m |

## Keyframe 定义

Keyframe 已定义在 `xmls/Assets/robot/realman_02_description.xml` 中，名称为 `home`。

```xml
<keyframe>
  <key name="home" qpos="0 0.1 0.1 1 0 0 0  0 0 0.5  0 0  0 1.25 0 2.35 1.57 1.57 0.36  0 0 0 0 0 0 0  0 -1.25 0 -2.35 -1.57 -1.57 -0.36  0 0 0 0 0 0 0"/>
</keyframe>
```


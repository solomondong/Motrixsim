# Motrixsim1

基于 [MotrixLab](https://motrixlab.readthedocs.io/) 搭建的 URDF 仿真训练环境，用于机器人控制算法开发与强化学习训练。

## 环境信息

- **操作系统**: Ubuntu 22.04
- **Python**: 3.11
- **Conda 环境**: `motrix-lerobot`
- **代码路径**: `/opt/motrix-lerobot`
- **底层框架**: [MotrixLab](https://motrixlab.readthedocs.io/) / MotrixSim

## 硬件配置（真机）

| 部件 | 型号 / 说明 |
|------|------------|
| 机械臂 | 自研 7-DOF 人形机械臂（仿真模型代号 `hrb3q`） |
| 末端夹爪 | 大寰（Dahuan）电动夹爪 |
| 手腕相机 | 奥比中光 Orbbec Gemini 335（RGBD） |

> 仿真环境需与真机硬件参数保持一致：URDF 中的关节限位、夹爪行程、相机内外参均以真机为准。修改 URDF 或相机标定前请先与硬件团队对齐。

## 环境激活

每次进入新 shell 都需要先激活环境：

```bash
source /opt/miniconda3/bin/activate
conda activate motrix-lerobot
```

或使用 `conda run` 单次执行：

```bash
conda run -n motrix-lerobot python <script.py>
```

需要实时查看输出时加 `--live-stream`：

```bash
conda run --live-stream -n motrix-lerobot python <script.py>
```

## 快速验证

运行以下脚本检查环境是否就绪（应能正常显示 hrb3q 机器人的 DOF 信息）：

```bash
cd /opt/motrix-lerobot
source /opt/miniconda3/bin/activate
conda activate motrix-lerobot
python scripts/display_hrb3q_dof.py
```

## 项目结构

```
/opt/motrix-lerobot/
├── HRB3Q-LC/          # 真机 URDF 原始包（含 meshes / config）
├── xmls/              # MuJoCo / MJCF 仿真模型（实际仿真用）
│   ├── hrb3q_demo.xml          # 基础展示模型
│   ├── hrb3q_dof_demo.xml      # DOF 调试模型
│   ├── hrb3q_gripper_test.xml  # 夹爪开闭测试模型（最早集成夹爪）
│   └── hrb3q_grasp_demo.xml    # 抓取 demo（现已集成夹爪 + 货架 + 桌子 + 瓶子）
├── scripts/           # 工具脚本（含诊断 / 测试 / demo）
├── shelves_demo/      # 货架抓取策略相关代码
├── docs/              # 设计文档
└── CLAUDE.md / AGENTS.md
```

> **重要**：`HRB3Q-LC/urdf/HRB3Q-LC.urdf` 当前用 `package://` URI 引用 meshes，motrixsim 加载会失败；
> 直接使用 `xmls/` 下的 MJCF 模型即可，URDF 仅作为真机配置参考。

## 仿真模型与夹爪接口

### 夹爪硬件抽象（仿真侧）

仿真用的是 DH 平行夹爪近似建模：每边各一对滑动手指，通过 MuJoCo `<equality polycoef>` 让两根手指同步移动，再用单个 `position` actuator 控制。

| Actuator | 关节 | ctrlrange | 单位 | 含义 |
|----------|------|-----------|------|------|
| `actuator_l_gripper` | `Joint_l_gripper_l`（mimic `Joint_l_gripper_r`）| `[0, 0.04]` | 米 | 左夹爪单边行程，0=闭合，0.04=完全张开（总开口 80mm） |
| `actuator_r_gripper` | `Joint_r_gripper_l`（mimic `Joint_r_gripper_r`）| `[0, 0.04]` | 米 | 右夹爪同上 |

控制示例：

```python
from motrixsim import SceneData, load_model

model = load_model("xmls/hrb3q_grasp_demo.xml")
data = SceneData(model)

l = model.get_actuator("actuator_l_gripper")
r = model.get_actuator("actuator_r_gripper")

l.set_ctrl(data, 0.04)
l.set_ctrl(data, 0.00)
l.set_ctrl(data, 0.02)
```

### 哪些 XML 含夹爪

| 模型 | 包含夹爪？ | 用途 |
|------|----------|------|
| `xmls/hrb3q_gripper_test.xml` | ✅ | 夹爪开闭与 mimic 行为测试 |
| `xmls/hrb3q_grasp_demo.xml` | ✅ | 完整抓取场景（货架 + 桌子 + 瓶子 + 双臂夹爪） |
| `xmls/hrb3q_demo.xml` / `xmls/hrb3q_dof_demo.xml` | ❌ | 仅手臂展示，不含夹爪 |
| `HRB3Q-LC/urdf/HRB3Q-LC.urdf` | ❌ | 真机 URDF 暂未建模夹爪 |

### 夹爪可控性诊断

任何时候怀疑"夹爪控制不上"，先跑诊断脚本：

```bash
python scripts/_diag_gripper.py --model <xml_path>
```

脚本会：列出全部 actuator/joint → 探测夹爪命名 → 跑 `0 → 0.04 → 0 → 0.04` 闭环响应测试。
判读：

- `actuator_l_gripper -> None` 即模型里没夹爪，**不是仿真器问题，是 XML 不对**
- 两根手指都跟随 ctrl 即 mimic equality 正常
- 仅 `finger_l` 动而 `finger_r` 不动，说明 mimic 失效，需要改双 actuator 方案

## 工作流约定

- 修改代码前先确认当前 conda 环境为 `motrix-lerobot`
- 所有 Python 命令必须在激活环境后执行，避免使用系统 Python
- 涉及 GUI / 渲染的脚本如需远程运行，请设置 `DISPLAY` 变量
- 改动 URDF / MJCF 资源后，务必跑 `scripts/display_hrb3q_dof.py` 验证基本可加载
- 改动夹爪 / 末端结构后，跑 `scripts/_diag_gripper.py --model <xml>` 验证 actuator 命名与 mimic 行为
- 仿真侧的关节顺序、夹爪开合方向、相机坐标系必须与真机 SDK 输出对齐
- 涉及 RGBD 数据的代码请同时考虑仿真渲染（虚拟相机）与真机 Gemini 335 的接口差异

## 参考资源

- MotrixLab 文档：<https://motrixlab.readthedocs.io/>
- MotrixLab GitHub：<https://github.com/Motphys/MotrixLab>

## 注意事项

- 代码在 Windows 工作区编辑、在 Ubuntu 上运行，路径处理需用 POSIX 风格
- 安装新依赖时请同步更新 `requirements.txt` / `environment.yml`
- 不要直接在 `main` 分支提交，使用功能分支 + PR 流程

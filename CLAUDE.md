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
├── scripts/        # 工具脚本（DOF 查看、仿真启动、可视化等）
├── assets/         # URDF / MJCF / 网格资源
├── ...             # 训练、仿真、策略相关代码
```

## 工作流约定

- 修改代码前先确认当前 conda 环境为 `motrix-lerobot`
- 所有 Python 命令必须在激活环境后执行，避免使用系统 Python
- 涉及 GUI / 渲染的脚本如需远程运行，请设置 `DISPLAY` 变量
- 改动 URDF / 资源文件后，务必跑一遍 `scripts/display_hrb3q_dof.py` 验证
- 仿真侧的关节顺序、夹爪开合方向、相机坐标系必须与真机 SDK 输出对齐
- 涉及 RGBD 数据的代码请同时考虑仿真渲染（虚拟相机）与真机 Gemini 335 的接口差异

## 参考资源

- MotrixLab 文档：<https://motrixlab.readthedocs.io/>
- MotrixLab GitHub：<https://github.com/Motphys/MotrixLab>

## 注意事项

- 代码在 Windows 工作区编辑、在 Ubuntu 上运行，路径处理需用 POSIX 风格
- 安装新依赖时请同步更新 `requirements.txt` / `environment.yml`
- 不要直接在 `main` 分支提交，使用功能分支 + PR 流程

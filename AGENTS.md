# Shelf Handling Project

## Environment

使用 conda 环境 **MotrixLab** 运行 Python 脚本，需要设置 DISPLAY 环境变量：

```bash
DISPLAY=:1 conda run -n MotrixLab python script.py
```

对于需要实时查看输出的脚本，使用 `--live-stream` 选项：

```bash
DISPLAY=:1 conda run --live-stream -n MotrixLab python script.py
```

## 参考资源

可以查看 `~/Desktop/motrixsim-docs` 的 sample code 获取 API 使用示例。

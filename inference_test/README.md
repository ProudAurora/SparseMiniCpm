# Inference Test

对 `../../models/CPM8B`（即仓库上一级目录 `models/CPM8B`，MiniCPM4-8B）进行推理测试的脚本集合。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 命令行运行

```bash
python run_inference.py --prompt "你好，请自我介绍一下。"
```

默认会自动定位到 `<仓库上级目录>/models/CPM8B`，也可通过 `--model-dir` 指定其他路径。

## 可视化推理窗口（Gradio）

```bash
python app.py
```

启动后浏览器访问 `http://<服务器IP>:7860`（AutoDL 用户可通过自定义服务/端口映射访问）。支持流式输出，并可在界面中调节 `max_new_tokens`、`temperature`、`top_p`。

常用参数：

```bash
python app.py --port 7860 --model-dir /path/to/model --share
```

- `--share`：生成一个临时公网访问链接（需要能连外网）
- `--model-dir`：指定其他模型路径，默认同 `run_inference.py`

若当前 shell 设置了 `HTTP_PROXY`/`HTTPS_PROXY`（如 `source ../../proxy_on.sh`），`app.py` 已自动将 `localhost`/`127.0.0.1` 加入 `NO_PROXY`，避免 Gradio 启动自检请求被代理拦截导致报错。

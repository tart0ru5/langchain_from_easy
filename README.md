# 实验成果

LangChain Agent 实验项目。

## 内容

- `testagent.py` —— 简单的agent，实现了一个计算器和一个查询时间的方法。并且能往项目里写入对话历史记录文件。
- `app_gradio.py` —— 配合上边agent的简单的gradio webui实现。

## 运行


可选环境变量：

| 变量 | 说明 |
| --- | --- |
| `GRADIO_SERVER_NAME` | 监听地址，默认 `127.0.0.1`（设为 `0.0.0.0` 可局域网访问） |
| `GRADIO_SERVER_PORT` | 端口，默认 `7860` |
| `GRADIO_SHARE` | 设为 `1` 时申请临时公网链接 |

## 依赖

- Python 3.10+
- `langchain` / `langchain-core` 等
- `gradio`

## 许可证

本项目采用 GNU General Public License v3.0，详见 [LICENSE](LICENSE)。


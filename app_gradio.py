"""给 testagent.py 的 Agent 套一个 Gradio 网页界面。

运行：
    D:\\AI\\langchain\\Scripts\\python.exe D:\\AI\\langchain\\app_gradio.py
然后浏览器打开 http://127.0.0.1:7860

可选环境变量：
    GRADIO_SERVER_NAME   监听地址，默认 127.0.0.1（想让局域网访问就设 0.0.0.0）
    GRADIO_SERVER_PORT   端口，默认 7860
    GRADIO_SHARE         设为 1 时请求一个临时公网链接

说明：
- Agent、工具（calculator / current_time）、模型都直接复用 testagent.py，本文件不重新定义。
- 对话记录同样写入 agent_history/ 下的 JSON 文件；一个进程（一次启动）用一个文件。
- 每轮都把完整历史重新传给 Agent，所以多个标签页同时用不会互相串。
"""

from __future__ import annotations

import json
import os

import gradio as gr
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from testagent import (
    DEFAULT_MODEL,
    ConversationLog,
    _content_to_text,
    build_agent,
)

AGENT = build_agent()
LOG = ConversationLog(model=DEFAULT_MODEL)


def _short(value: object, limit: int = 160) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    value = value.replace("\n", " ").strip()
    return value if len(value) <= limit else value[:limit] + "…"


def _format_trace(calls: dict, results: dict) -> str:
    """把本轮的工具调用拼成几行文字，附在回复末尾。"""
    if not calls:
        return ""
    lines = ["", "—" * 12]
    for call_id, call in calls.items():
        args = ", ".join(f"{k}={v!r}" for k, v in (call.get("args") or {}).items())
        lines.append(f"调用 {call.get('name')}({args})")
        if call_id in results:
            lines.append(f"返回 {_short(results[call_id])}")
        else:
            lines.append("返回 （未执行/无结果）")
    return "\n".join(lines)


def respond(message: str, history: list):
    """ChatInterface 的对话函数：生成器，边生成边刷新。"""
    text_out = ""
    calls: dict[str, dict] = {}
    results: dict[str, str] = {}
    try:
        LOG.add_user(message)
        msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in (history or [])
            if isinstance(m, dict) and m.get("content")
        ]
        msgs.append({"role": "user", "content": message})

        for mode, payload in AGENT.stream(
            {"messages": msgs}, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                chunk, _meta = payload
                # 只显示 AI 正文增量，跳过工具调用参数流
                if isinstance(chunk, AIMessageChunk) and not chunk.tool_call_chunks:
                    piece = _content_to_text(chunk.content)
                    if piece:
                        text_out += piece
                        yield text_out
            elif mode == "updates":
                for update in payload.values():
                    if not isinstance(update, dict):
                        continue
                    for msg in update.get("messages", []):
                        LOG.add_message(msg)
                        if isinstance(msg, ToolMessage):
                            results[msg.tool_call_id] = _content_to_text(msg.content)
                        elif isinstance(msg, AIMessage):
                            for call in msg.tool_calls:
                                calls[call.get("id")] = call

        trace = _format_trace(calls, results)
        if not text_out:
            text_out = "（这次没有产生文字回复）"
        yield text_out + trace
    except Exception as exc:  # noqa: BLE001
        yield (text_out + f"\n\n**出错了**：{type(exc).__name__}: {exc}").strip()


def main() -> None:
    demo = gr.ChatInterface(
        respond,
        title="本地 LangChain Agent",
        description=(
            f"模型：{DEFAULT_MODEL}　工具：calculator / current_time"
            f"　记录：{LOG.path}"
        ),
        examples=["(2+3)*4 等于几", "现在几点", "12345*6789 再除以 7"],
    )
    demo.queue()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE") == "1",
    )


if __name__ == "__main__":
    main()

"""一个带对话记录的 LangChain Agent 示例（OpenRouter + LangGraph）。

运行方式：
    D:\\langchain\\Scripts\\python.exe D:\\langchain\\testagent.py                 # 交互模式
    D:\\langchain\\Scripts\\python.exe D:\\langchain\\testagent.py "1+1等于几"      # 单次提问

对话记录：
    每次启动对话时，在 D:\\langchain\\agent_history\\ 下用开始时间创建一个 .json 文件，
    之后这一轮对话的所有消息都实时写进该文件。
    消息的 role 有四种：
        user        用户输入
        assistant   模型输出的文字
        tool_call   模型请求调用的工具（name / arguments / tool_call_id）
        tool_result 工具执行结果（name / content / status / tool_call_id）

模型路由：
    默认走 deepseek/deepseek-v4.1-flash-1017，并通过 OpenRouter 的 provider
    路由参数锁定到 DeepSeek 官方端点（见 OPENROUTER_PROVIDER）。

API Key 来源（按优先级）：
    1. 环境变量 OPENROUTER_API_KEY
    2. 同目录下的 .openrouter_key 文件（内容就是 sk-or-... 一行）
"""

from __future__ import annotations

import ast
import json
import operator
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain.tools import tool
from langchain_openrouter import ChatOpenRouter

DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash-1017")
HISTORY_DIR = Path(__file__).with_name("agent_history")

# OpenRouter 的 provider 路由参数（请求体里的顶层 provider 字段）。
# only + allow_fallbacks=false：只允许 DeepSeek 官方端点，推理不进别家的转载站；
# order 把它排在第一位（只是双保险，配合 only 后候选就剩它一个）。
# 字段说明见 https://openrouter.ai/docs/features/provider-routing
OPENROUTER_PROVIDER = {
    "order": ["deepseek"],
    "only": ["deepseek"],
    "allow_fallbacks": False,
}

SYSTEM_PROMPT = (
    "你是一个简洁的中文助手。"
    "遇到算术必须调用 calculator 工具，不要自己心算；"
    "遇到问时间必须调用 current_time 工具。"
    "回答尽量简短，直接给结论。"
)


# ---------------------------------------------------------------- 工具

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("表达式只支持数字、括号和 + - * / // % ** 运算")


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式，例如 "(2+3)*4" 或 "2**10"。支持 + - * / // % ** 和括号。"""
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
    except Exception as exc:  # noqa: BLE001
        return f"计算失败：{exc}"
    return str(result)


@tool
def current_time(zone: str = "local") -> str:
    """查询当前时间。zone 传 "local" 表示本机时间，"utc" 表示 UTC 时间。"""
    if zone.lower() == "utc":
        now = datetime.now(timezone.utc)
        return now.strftime("UTC %Y-%m-%d %H:%M:%S")
    return datetime.now().strftime("本地时间 %Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 对话记录

def _content_to_text(content: Any) -> str:
    """把消息的 content 统一成可读字符串（兼容 str 和 content blocks 两种形式）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


class ConversationLog:
    """把一次对话以 JSON 形式实时落盘。

    文件形如 agent_history/2026-11-09_18-32-00.json：
        {
          "session_id": "2026-11-09_18-32-00",
          "started_at": "2026-11-09T18:32:00.123456+00:00",
          "model": "openai/gpt-4o-mini",
          "messages": [
            {"role": "user",    "time": "...", "content": "..."},
            {"role": "tool_call",   "time": "...", "tool_call_id": "call_1",
             "name": "calculator", "arguments": {"expression": "(2+3)*4"}},
            {"role": "tool_result", "time": "...", "tool_call_id": "call_1",
             "name": "calculator", "status": "success", "content": "20"},
            {"role": "assistant", "time": "...", "content": "..."}
          ]
        }
    """

    def __init__(self, model: str, directory: Path = HISTORY_DIR) -> None:
        self.model = model
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now(timezone.utc)
        self.path = self._unique_path(self.started_at)
        self.records: list[dict[str, Any]] = []
        self._seen_ids: set[str] = set()
        self._flush()  # 开始对话时就把文件建好

    def _unique_path(self, moment: datetime) -> Path:
        stem = moment.astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        path = self.directory / f"{stem}.json"
        index = 1
        while path.exists():
            index += 1
            path = self.directory / f"{stem}_{index}.json"
        return path

    def _flush(self) -> None:
        data = {
            "session_id": self.path.stem,
            "started_at": self.started_at.isoformat(),
            "model": self.model,
            "messages": self.records,
        }
        # 先写临时文件再替换，保证任何时刻磁盘上的都是合法 JSON
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self.path)

    def add(self, role: str, content: str | None = None, **extra: Any) -> None:
        record: dict[str, Any] = {
            "role": role,
            "time": datetime.now(timezone.utc).isoformat(),
        }
        if content is not None:
            record["content"] = content
        record.update(extra)
        self.records.append(record)
        self._flush()

    def add_user(self, text: str) -> None:
        self.add("user", text)

    def add_message(self, message: BaseMessage) -> None:
        """按消息类型拆成 user / assistant / tool_call / tool_result 记录。"""
        message_id = getattr(message, "id", None)
        if message_id:
            if message_id in self._seen_ids:
                return
            self._seen_ids.add(message_id)

        if isinstance(message, ToolMessage):
            content = message.content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            self.add(
                "tool_result",
                content,
                tool_call_id=message.tool_call_id,
                name=message.name,
                status=getattr(message, "status", None),
            )
            return

        if isinstance(message, AIMessage):
            text = _content_to_text(message.content)
            if text:
                self.add("assistant", text)
            for call in message.tool_calls:
                self.add(
                    "tool_call",
                    tool_call_id=call.get("id"),
                    name=call.get("name"),
                    arguments=call.get("args"),
                )
            return

        # 其它类型（如 system）原样记录，便于排查
        self.add(message.type, _content_to_text(message.content))


# ---------------------------------------------------------------- 组装

def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    key_file = Path(__file__).with_name(".openrouter_key")
    if key_file.is_file():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    sys.exit(
        "未找到 OpenRouter API Key。\n"
        "请设置环境变量 OPENROUTER_API_KEY，"
        "或在 D:\\langchain\\.openrouter_key 里写入你的 key（一行）。"
    )


def build_agent():
    model = ChatOpenRouter(
        model_name=DEFAULT_MODEL,
        openrouter_api_key=load_api_key(),
        temperature=0,
        app_title="langchain-simple-agent",
        # 请求体里带上 provider 字段，锁定 DeepSeek 官方端点
        openrouter_provider=OPENROUTER_PROVIDER,
    )
    return create_agent(
        model,
        tools=[calculator, current_time],
        system_prompt=SYSTEM_PROMPT,
    )


def run_turn(agent, messages: list, log: ConversationLog) -> str:
    """跑一轮对话：实时打印模型文本，并把这一步产生的所有消息写入记录文件。"""
    printed: list[str] = []
    for mode, payload in agent.stream(
        {"messages": list(messages)}, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            chunk, _metadata = payload
            # 只打印 AI 的文本增量，跳过工具结果和工具调用参数流
            if isinstance(chunk, AIMessageChunk) and not chunk.tool_call_chunks:
                text = _content_to_text(chunk.content)
                if text:
                    print(text, end="", flush=True)
                    printed.append(text)
        elif mode == "updates":
            for update in payload.values():
                if not isinstance(update, dict):
                    continue
                for message in update.get("messages", []):
                    log.add_message(message)
    print()
    return "".join(printed)


def main() -> None:
    agent = build_agent()
    log = ConversationLog(model=DEFAULT_MODEL)
    history: list = []

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"记录：{log.path}")
        log.add_user(question)
        history.append({"role": "user", "content": question})
        run_turn(agent, history, log)
        return

    print(f"模型：{DEFAULT_MODEL}    输入 exit 退出")
    print(f"记录：{log.path}")
    while True:
        try:
            question = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        log.add_user(question)
        history.append({"role": "user", "content": question})
        print("AI > ", end="", flush=True)
        answer = run_turn(agent, history, log)
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

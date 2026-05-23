"""
Agent service: ToolRegistry + SemanticNavigator + AgentLLM

ToolRegistry  — register and execute tools (get_time, calculator, web_search)
SemanticNavigator — map semantic motion labels to graph paths
AgentLLM      — DeepSeek wrapper with tool-calling loop, yields SSE-ready events
"""

import json
import math
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Generator

_TZ_CST = timezone(timedelta(hours=8))  # Asia/Shanghai / Asia/Chongqing

import networkx as nx
import numpy as np
from openai import OpenAI

import models
from utils import calculate_loss


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_get_time(_args: dict) -> str:
    now = datetime.now(_TZ_CST)
    return now.strftime("现在是北京时间 %Y年%m月%d日 %H:%M:%S（UTC+8）")


def _tool_calculator(args: dict) -> str:
    expr = str(args.get("expression", "")).strip()
    # Whitelist: only allow numbers, basic ops, spaces, parentheses, decimal point
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return "表达式包含不支持的字符"
    try:
        result = eval(expr, {"__builtins__": {}})  # noqa: S307
        return f"{expr} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def _tool_web_search(args: dict) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "请提供搜索关键词"
    try:
        encoded = urllib.parse.quote_plus(query)
        # DuckDuckGo Lite — returns actual search result snippets for any language
        url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Extract result snippets: text between <td class="result-snippet"> … </td>
        snippets = re.findall(
            r'class=["\']result-snippet["\'][^>]*>(.*?)</td>',
            html, re.DOTALL | re.IGNORECASE
        )
        # Strip inner HTML tags
        clean = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:5]]
        clean = [s for s in clean if s]

        if clean:
            return "\n".join(f"• {s}" for s in clean)

        # Fallback: extract any <a> link titles
        titles = re.findall(r'class=["\']result-link["\'][^>]*>(.*?)</a>', html, re.DOTALL)
        titles = [re.sub(r"<[^>]+>", "", t).strip() for t in titles[:5] if t.strip()]
        if titles:
            return "搜索结果标题：\n" + "\n".join(f"• {t}" for t in titles)

        return "未找到搜索结果（可能被反爬虫拦截，请稍后重试）"
    except Exception as e:
        return f"搜索失败: {e}"


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_time",
        "description": "获取当前日期和时间",
        "parameters": {},
    },
    {
        "name": "calculator",
        "description": "计算数学表达式，仅支持 +、-、*、/、() 运算",
        "parameters": {
            "expression": "字符串，要计算的数学表达式，例如 '(3 + 5) * 2'"
        },
    },
    {
        "name": "web_search",
        "description": "用 DuckDuckGo 搜索互联网，返回摘要",
        "parameters": {
            "query": "字符串，搜索关键词"
        },
    },
]

_TOOL_FN_MAP = {
    "get_time": _tool_get_time,
    "calculator": _tool_calculator,
    "web_search": _tool_web_search,
}


class ToolRegistry:
    def schema_text(self) -> str:
        lines = []
        for t in TOOL_DEFINITIONS:
            params = ", ".join(f"{k}: {v}" for k, v in t["parameters"].items()) or "无参数"
            lines.append(f'- {t["name"]}: {t["description"]}（参数: {params}）')
        return "\n".join(lines)

    def execute(self, name: str, args: dict) -> str:
        fn = _TOOL_FN_MAP.get(name)
        if fn is None:
            return f"未知工具: {name}"
        try:
            return fn(args)
        except Exception as e:
            return f"工具执行错误: {e}"

    def names(self) -> list[str]:
        return list(_TOOL_FN_MAP.keys())


# ---------------------------------------------------------------------------
# SemanticNavigator
# ---------------------------------------------------------------------------

class SemanticNavigator:
    def __init__(self, labels_path: str = "nuero/labels.json"):
        path = Path(labels_path)
        if not path.exists():
            raise FileNotFoundError(f"labels.json not found: {path}. Run auto_label.py first.")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._folder_to_label: dict[str, str] = data["folder_to_label"]
        self._label_to_folders: dict[str, list[str]] = data["label_to_folders"]

    @property
    def available_motions(self) -> list[str]:
        return sorted(self._label_to_folders.keys())

    def _nodes_for_label(self, label: str) -> list[str]:
        """Return all graph nodes (image paths) belonging to a semantic label."""
        folders = self._label_to_folders.get(label, [])
        if not folders:
            return []
        nodes = []
        for node in models.G.nodes:
            # node format: nuero/images/video_XXXXX/XXXXXX.png
            parts = Path(node).parts
            if len(parts) >= 3:
                folder_name = parts[-2]  # e.g. video_00014
                if folder_name in folders:
                    nodes.append(node)
        return nodes

    def navigate_to(
        self,
        label: str,
        current_node: str,
        current_t0: list[list[float]],
    ) -> list[str]:
        """
        Find the frame in the target label cluster that represents the PEAK of
        the motion (most different from current position), then return the
        shortest path from current_node to it.

        We pick the frame with MAX displacement rather than MIN, so the
        character visibly moves instead of staying at the nearest (current) frame.

        Returns a list of node paths (not including current_node itself).
        """
        if models.G is None:
            return []

        # Exclude current_node so we always move somewhere
        candidates = [c for c in self._nodes_for_label(label) if c != current_node]
        if not candidates:
            candidates = [c for c in self._nodes_for_label("neutral") if c != current_node]
        if not candidates:
            return []

        n = len(current_t0)
        if current_t0 and n > 0:
            t0_np = np.array(current_t0, dtype=np.float32)
            # Pick the frame MOST different from current (= peak of motion)
            best_node = None
            best_loss = -1.0
            for node in candidates:
                node_pos = models.NODE_POSITIONS.get(node, [])
                if not node_pos or len(node_pos) != n:
                    continue
                loss = float(np.mean((t0_np - np.array(node_pos, dtype=np.float32)) ** 2))
                if loss > best_loss:
                    best_loss = loss
                    best_node = node
            if best_node is None:
                # Fallback: no matching point count — pick temporal midpoint
                best_node = sorted(candidates)[len(candidates) // 2]
        else:
            # No tracking data: pick temporal midpoint of candidate list
            best_node = sorted(candidates)[len(candidates) // 2]

        if best_node not in models.G or current_node not in models.G:
            return []

        try:
            path = nx.shortest_path(models.G, current_node, best_node)
            return path[1:]  # exclude current_node
        except nx.NetworkXNoPath:
            return []


# ---------------------------------------------------------------------------
# AgentLLM
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """你是一个名为 nuero-sama 的女性虚拟主播，有真实的动作数据驱动你的头部运动。

## 你的动作词表（来自真实角色视频数据）
{motions_list}
- neutral: 保持静止

## 你可以调用的工具
{tools_schema}

## 严格的回复格式（必须完全按此输出）

正常回复时，先输出控制块，再输出回答文本：

<ctrl>{{"motion":"动作标签","tool":null,"tool_args":{{}}}}</ctrl>
<answer>
你的回答正文，可以多行
</answer>

需要调用工具时：
<ctrl>{{"motion":"neutral","tool":"工具名","tool_args":{{"参数名":"参数值"}}}}</ctrl>
<answer></answer>

工具结果会以 [TOOL_RESULT: ...] 形式给出，之后你再按上面正常格式回复。

## 动作标签映射（必须严格按此选择，不能自创）
| 用户意图 | 应选标签 |
|---------|---------|
| 点头、肯定、同意、打招呼 | nod |
| 摇头（左右摇）、否定、拒绝 | shake |
| 向左看、向左转头、向左歪头 | look_left |
| 向右看、向右转头、向右歪头 | look_right |
| 无特定动作、保持静止 | neutral |

⚠️ 注意：「向左摇头」= look_left（不是 shake）；「向右摇头」= look_right（不是 shake）
"""


class AgentLLM:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ):
        self._tools = tool_registry
        self._model = model
        self._client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=base_url,
        )

        try:
            motions_list = "\n".join(
                f"- {m}" for m in SemanticNavigator("nuero/labels.json").available_motions
            )
        except FileNotFoundError:
            motions_list = "- neutral: 保持静止"
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            motions_list=motions_list,
            tools_schema=tool_registry.schema_text(),
        )
        self._history: list[dict] = []

    def reset(self):
        self._history.clear()

    def _stream_response(self, messages: list[dict]) -> Generator[dict, None, None]:
        """
        Stream one LLM call, parsing <ctrl>...</ctrl> and <answer>...</answer> tags.
        Yields:
          {"type": "ctrl",       "data": dict}            — parsed control block
          {"type": "text_delta", "delta": str}            — streaming answer chars
          {"type": "_raw",       "content": str}          — full raw response for history
        """
        # 每次调用动态注入当前时间，避免模型用训练数据里的旧日期
        now_str = datetime.now(_TZ_CST).strftime("%Y年%m月%d日 %H:%M:%S")
        sys_with_time = f"【当前北京时间：{now_str}】\n\n" + self._system_prompt
        full_messages = [{"role": "system", "content": sys_with_time}] + messages
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            stream=True,
        )

        full_text = ""
        buf = ""
        state = "SCAN"   # SCAN | IN_CTRL | IN_ANSWER
        ctrl_buf = ""
        ctrl_done = False

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            full_text += delta
            buf += delta

            while True:
                if state == "SCAN":
                    if "<ctrl>" in buf:
                        buf = buf[buf.index("<ctrl>") + len("<ctrl>"):]
                        state = "IN_CTRL"
                        ctrl_buf = ""
                    elif "<answer>" in buf:
                        buf = buf[buf.index("<answer>") + len("<answer>"):]
                        state = "IN_ANSWER"
                    else:
                        # keep last few chars for partial tag match
                        if len(buf) > 10:
                            buf = buf[-10:]
                        break

                elif state == "IN_CTRL":
                    if "</ctrl>" in buf:
                        idx = buf.index("</ctrl>")
                        ctrl_buf += buf[:idx]
                        buf = buf[idx + len("</ctrl>"):]
                        state = "SCAN"
                        try:
                            ctrl_data = json.loads(ctrl_buf.strip())
                        except json.JSONDecodeError:
                            ctrl_data = {"motion": "neutral", "tool": None,
                                         "tool_args": {}, "describe": ""}
                        if not ctrl_done:
                            ctrl_done = True
                            yield {"type": "ctrl", "data": ctrl_data}
                    else:
                        ctrl_buf += buf
                        buf = ""
                        break

                elif state == "IN_ANSWER":
                    if "</answer>" in buf:
                        idx = buf.index("</answer>")
                        text_chunk = buf[:idx]
                        if text_chunk:
                            yield {"type": "text_delta", "delta": text_chunk}
                        buf = buf[idx + len("</answer>"):]
                        state = "SCAN"
                    else:
                        # Stream all but last few chars (partial tag guard)
                        safe = len(buf) - 10
                        if safe > 0:
                            yield {"type": "text_delta", "delta": buf[:safe]}
                            buf = buf[safe:]
                        break

        # Fallback: if LLM didn't use tags, recover text and motion
        if not ctrl_done:
            fallback_ctrl = {"motion": "neutral", "tool": None, "tool_args": {}, "describe": ""}
            fallback_answer = ""

            # 1. Try strict JSON parse
            try:
                obj = json.loads(full_text.strip())
                fallback_ctrl = {
                    "motion":    obj.get("motion", "neutral"),
                    "tool":      obj.get("tool", None),
                    "tool_args": obj.get("tool_args", {}),
                    "describe":  obj.get("describe", ""),
                }
                fallback_answer = obj.get("answer", "")
            except (json.JSONDecodeError, AttributeError):
                pass

            # 2. Try finding JSON block anywhere in the response (handles leading text)
            if not fallback_answer:
                m = re.search(r'\{[^{}]*"motion"[^{}]*\}', full_text, re.DOTALL)
                if m:
                    try:
                        obj = json.loads(m.group())
                        fallback_ctrl = {
                            "motion":    obj.get("motion", "neutral"),
                            "tool":      obj.get("tool", None),
                            "tool_args": obj.get("tool_args", {}),
                            "describe":  obj.get("describe", ""),
                        }
                        fallback_answer = obj.get("answer", "")
                    except (json.JSONDecodeError, AttributeError):
                        pass

            # 3. Last resort: strip all tags and ctrl JSON, emit remaining text
            if not fallback_answer:
                stripped = re.sub(r'<[^>]+>', '', full_text).strip()
                # 去掉 ctrl 块 JSON（含 "motion" 键的对象），支持一层嵌套花括号
                stripped = re.sub(
                    r'\{(?:[^{}]|\{[^{}]*\})*"motion"(?:[^{}]|\{[^{}]*\})*\}?',
                    '', stripped, flags=re.DOTALL
                ).strip()
                if stripped:
                    fallback_answer = stripped

            if fallback_answer:
                yield {"type": "text_delta", "delta": fallback_answer}
            yield {"type": "ctrl", "data": fallback_ctrl}

        yield {"type": "_raw", "content": full_text}

    def chat(self, user_input: str) -> Generator[dict, None, None]:
        """
        Yields SSE-ready event dicts:
          {"type": "tool_call",   "name": str, "args": dict}
          {"type": "tool_result", "content": str}
          {"type": "text_delta",  "delta": str}           ← streaming answer text
          {"type": "answer",      "text": str, "motion": str, "describe": str}
          {"type": "motion",      "label": str}
          {"type": "done"}
        """
        self._history.append({"role": "user", "content": user_input})

        for _ in range(3):
            ctrl_data = None
            answer_text = ""
            raw_content = ""

            for event in self._stream_response(self._history):
                if event["type"] == "ctrl":
                    ctrl_data = event["data"]
                    tool_name = (ctrl_data.get("tool") or "").strip() or None
                    if tool_name and tool_name in self._tools.names():
                        yield {"type": "tool_call", "name": tool_name,
                               "args": ctrl_data.get("tool_args") or {}}
                elif event["type"] == "text_delta":
                    answer_text += event["delta"]
                    yield event   # forward to SSE stream in real time
                elif event["type"] == "_raw":
                    raw_content = event["content"]

            self._history.append({"role": "assistant", "content": raw_content})

            if ctrl_data is None:
                ctrl_data = {"motion": "neutral", "tool": None, "describe": ""}

            tool_name = (ctrl_data.get("tool") or "").strip() or None
            if tool_name and tool_name in self._tools.names():
                result = self._tools.execute(tool_name, ctrl_data.get("tool_args") or {})
                yield {"type": "tool_result", "content": result}
                self._history.append({
                    "role": "user",
                    "content": f"[TOOL_RESULT: {result}]\n请根据以上结果给出最终回答。",
                })
                # continue loop for final answer
            else:
                motion = ctrl_data.get("motion", "neutral")
                describe = ctrl_data.get("describe", "")
                yield {"type": "answer", "text": answer_text, "motion": motion, "describe": describe}
                yield {"type": "motion", "label": motion}
                yield {"type": "done"}
                return

        yield {"type": "answer", "text": "（处理超时）", "motion": "neutral", "describe": ""}
        yield {"type": "done"}

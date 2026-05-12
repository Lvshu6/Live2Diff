"""
桌宠版 Live2Diff —— app_qt5_desk.py
从 app_qt5.py 改造，专注于桌面宠物体验：
  - 透明无边框悬浮窗
  - 左键拖：移动宠物位置
  - 全局鼠标追踪：桌宠实时看向鼠标方向
  - AnimationController 驱动空闲动画 + 语义动作
  - 监控活跃窗口 → 预取 LLM → 延迟显示气泡
  - InputBar：紧凑浮动条，人格/对话选择，Enter 发送，回复以气泡显示
  - 右键菜单：聊天 / 退出

启动（从 app/Live2D/ 目录）:
  DEEPSEEK_API_KEY=<key> python app_qt5_desk.py
"""

import json
import math
import os
import re
import sys
import time
import threading
import random
import subprocess
import platform
import webbrowser
import urllib.request
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import networkx as nx

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QCursor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QMenu, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox,
    QFrame, QInputDialog, QDialog, QTextEdit,
)

try:
    from openai import OpenAI
    _openai_ok = True
except ImportError:
    _openai_ok = False

try:
    from agent_service import AgentLLM, ToolRegistry, SemanticNavigator
    _agent_ok = True
except ImportError:
    _agent_ok = False

try:
    from animation import AnimationController
    _anim_ok = True
except ImportError:
    _anim_ok = False

import models

# ── 配置 ─────────────────────────────────────────────────────────────
BASE_PATH = "nuero"
models.BASE_DIR = BASE_PATH
models.GRAPH_PATH = os.path.join(BASE_PATH, "graph.txt")
models.IMAGES_DIR = os.path.join(BASE_PATH, "images")
models.TRACKS_DIR = os.path.join(BASE_PATH, "track")
models.load_graph()

IMAGE_BASE_REL_DIR = os.path.join(BASE_PATH, "images", "video_00014")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
PET_W, PET_H = 300, 410
IMG_W = 480
FPS = 30
ANIM_TICK_MS = 80
IDLE_INTERVAL_MS = 2500       # fallback when AnimationController unavailable
BUBBLE_DURATION_MS = 8000
LLM_DELAY_S = 3              # 窗口切换后多少秒显示气泡
WATCHER_INTERVAL_S = 3
WATCHER_COOLDOWN_S = 10      # 同一窗口最小触发间隔（秒）

# 凝视：鼠标距宠物中心小于此阈值（屏幕像素）时视为"看中心"
GAZE_CENTER_THRESHOLD = 40
GAZE_POLL_MS = 80             # 凝视轮询间隔（ms）
GAZE_ZONES = 8                # 将 360° 分成几个扇区（越多越细腻，越少越稳定）
GAZE_ZONE_RADIUS = 120        # 扇区方向计算用的虚拟半径（屏幕像素）

HISTORY_FILE = os.path.join(BASE_PATH, "chat_history.json")
PERSONAS_FILE = os.path.join(BASE_PATH, "personas.json")


# ── 数据辅助 ──────────────────────────────────────────────────────────
def load_personas() -> list:
    default = [{"name": "默认", "description": "你是 nuero-sama，一个可爱的二次元虚拟主播。"}]
    if not os.path.exists(PERSONAS_FILE):
        return default
    try:
        with open(PERSONAS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass
    return default


def save_personas(personas: list):
    try:
        os.makedirs(BASE_PATH, exist_ok=True)
        with open(PERSONAS_FILE, "w", encoding="utf-8") as f:
            json.dump(personas, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[人格] 保存失败: {e}")


def load_history() -> dict:
    """格式: {persona_name: {conv_name: [{"role", "content", "timestamp"}]}}
    兼容旧格式 {persona_name: [messages]}，自动迁移为新格式。
    """
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        # 迁移旧格式：值为 list 的条目升级为 {"对话1": list}
        migrated = False
        for k, v in list(data.items()):
            if isinstance(v, list):
                data[k] = {"对话1": v}
                migrated = True
        if migrated:
            save_history(data)
        return data
    except Exception:
        return {}


def save_history(history_data: dict):
    try:
        os.makedirs(BASE_PATH, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[历史] 保存失败: {e}")


# ── 检查是否已有 track 数据 ───────────────────────────────────────────
def check_tracks_exist() -> bool:
    tracks_dir = os.path.join(BASE_PATH, "track")
    if not os.path.exists(tracks_dir):
        return False
    for _, _, files in os.walk(tracks_dir):
        if any(f.endswith(".json") for f in files):
            return True
    return False


# ── track 缓存预加载 ──────────────────────────────────────────────────
def _preload_track_caches():
    if models.G is None:
        return
    loaded = 0
    for node in models.G.nodes:
        if models.NODE_POSITIONS.get(node):
            loaded += 1
            continue
        try:
            pos = models.get_node_position(node)
            if pos:
                models.NODE_POSITIONS[node] = pos
                loaded += 1
        except Exception:
            pass
    print(f"track 缓存预加载：{loaded} 个节点")

_preload_track_caches()


# ── 工具 ─────────────────────────────────────────────────────────────
def calculate_loss(points, node_positions):
    if not points or not node_positions or len(points) != len(node_positions):
        return float("inf")
    p = np.array(points, dtype=np.float32)
    n = np.array(node_positions, dtype=np.float32)
    return float(np.mean((p - n) ** 2))


# ── Agent 工作线程 ────────────────────────────────────────────────────
class AgentWorker(QThread):
    text_delta = pyqtSignal(str)
    tool_call_sig = pyqtSignal(str, str)    # name, args_json
    tool_result_sig = pyqtSignal(str)
    answer_done = pyqtSignal(str, str, str) # text, motion, describe
    motion_requested = pyqtSignal(str)
    error_sig = pyqtSignal(str)

    def __init__(self, agent, user_input: str, parent=None):
        super().__init__(parent)
        self._agent = agent
        self._input = user_input

    def run(self):
        try:
            _in_tool = False  # 工具调用轮次中屏蔽中间 delta
            for event in self._agent.chat(self._input):
                t = event["type"]
                if t == "text_delta":
                    if not _in_tool:
                        self.text_delta.emit(event["delta"])
                elif t == "tool_call":
                    _in_tool = True
                    self.tool_call_sig.emit(
                        event["name"],
                        json.dumps(event.get("args", {}), ensure_ascii=False),
                    )
                elif t == "tool_result":
                    _in_tool = False
                    self.tool_result_sig.emit(event["content"])
                elif t == "answer":
                    _in_tool = False
                    self.answer_done.emit(
                        event.get("text", ""),
                        event.get("motion", "neutral"),
                        event.get("describe", ""),
                    )
                    self.motion_requested.emit(event.get("motion", "neutral"))
        except Exception as e:
            self.error_sig.emit(str(e))


# ── 紧凑输入条 ───────────────────────────────────────────────────────
_INPUT_BAR_CSS = """
QWidget#bar {
    background: rgba(255, 255, 255, 235);
    border: 1px solid #c4c9d4;
    border-radius: 10px;
    font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
    font-size: 13px;
}
QComboBox {
    background: #edf0f5;
    border: 1px solid #b8bdc9;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 13px;
    color: #1f2937;
    min-width: 68px;
    max-width: 68px;
}
QComboBox::drop-down { border: none; width: 14px; }
QComboBox QAbstractItemView {
    background: #ffffff;
    border: 1px solid #9ca3af;
    selection-background-color: #4f46e5;
    selection-color: #ffffff;
    color: #1f2937;
    font-size: 13px;
    padding: 2px;
}
QPushButton#plus {
    background: #e5e7eb;
    border: none;
    border-radius: 4px;
    color: #1f2937;
    font-size: 13px;
    padding: 1px 5px;
    max-width: 20px;
    min-width: 20px;
}
QPushButton#plus:hover { background: #c7d2fe; color: #3730a3; }
QPushButton#close_btn {
    background: transparent;
    border: none;
    color: #6b7280;
    font-size: 16px;
    padding: 0px 4px;
    max-width: 24px;
    min-width: 24px;
}
QPushButton#close_btn:hover { color: #ef4444; }
QLineEdit#input {
    background: transparent;
    border: none;
    color: #111827;
    font-size: 14px;
    padding: 2px 4px;
}
"""


class InputBar(QWidget):
    """紧凑浮动输入条：[人格▼][+] [对话▼][+] | 输入… [×]"""

    send_requested = pyqtSignal(str, str, str)  # user_text, persona_name, conv_name
    bar_closed = pyqtSignal()

    BAR_H = 42
    BAR_W = 480

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedHeight(self.BAR_H)
        self.setFixedWidth(self.BAR_W)

        self._personas = load_personas()
        self._history_data = load_history()
        self._is_generating = False
        self._answer_buf = ""
        self._drag_pos = None

        self._build_ui()
        self._refresh_conv_combo()

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bar = QWidget(self)
        bar.setObjectName("bar")
        bar.setStyleSheet(_INPUT_BAR_CSS)
        outer.addWidget(bar)

        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 4, 6, 4)
        row.setSpacing(4)

        # 人格选择
        self._persona_combo = QComboBox()
        self._persona_combo.setToolTip("人格")
        for p in self._personas:
            self._persona_combo.addItem(p["name"])
        self._persona_combo.currentTextChanged.connect(self._on_persona_changed)
        row.addWidget(self._persona_combo)

        btn_new_persona = QPushButton("+")
        btn_new_persona.setObjectName("plus")
        btn_new_persona.setToolTip("新建人格")
        btn_new_persona.clicked.connect(self._add_persona)
        row.addWidget(btn_new_persona)

        row.addSpacing(4)

        # 对话选择
        self._conv_combo = QComboBox()
        self._conv_combo.setToolTip("对话存档")
        row.addWidget(self._conv_combo)

        btn_new_conv = QPushButton("+")
        btn_new_conv.setObjectName("plus")
        btn_new_conv.setToolTip("新建对话")
        btn_new_conv.clicked.connect(self._add_conv)
        row.addWidget(btn_new_conv)

        # 分隔线
        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setStyleSheet("color: #d1d5db; margin: 6px 4px;")
        row.addWidget(div)

        # 输入框
        self._input = QLineEdit()
        self._input.setObjectName("input")
        self._input.setPlaceholderText("说点什么… (Enter 发送)")
        self._input.returnPressed.connect(self._send)
        row.addWidget(self._input, stretch=1)

        # 关闭按钮
        btn_close = QPushButton("×")
        btn_close.setObjectName("close_btn")
        btn_close.setToolTip("关闭")
        btn_close.clicked.connect(self._on_close_clicked)
        row.addWidget(btn_close)

        # 右键菜单
        self._persona_combo.setContextMenuPolicy(Qt.CustomContextMenu)
        self._persona_combo.customContextMenuRequested.connect(self._persona_context_menu)
        self._conv_combo.setContextMenuPolicy(Qt.CustomContextMenu)
        self._conv_combo.customContextMenuRequested.connect(self._conv_context_menu)

    def _on_close_clicked(self):
        self.hide()
        self.bar_closed.emit()

    # ── 人格/对话管理 ─────────────────────────────────────────────
    def _on_persona_changed(self, name: str):
        self._refresh_conv_combo()

    def _refresh_conv_combo(self):
        persona = self._persona_combo.currentText()
        convs = list(self._history_data.get(persona, {}).keys())
        if not convs:
            convs = ["对话1"]
        self._conv_combo.blockSignals(True)
        self._conv_combo.clear()
        for c in convs:
            self._conv_combo.addItem(c)
        self._conv_combo.blockSignals(False)

    def _add_persona(self):
        name, ok = QInputDialog.getText(self, "新建人格", "人格名称：")
        if not (ok and name.strip()):
            return
        name = name.strip()
        if any(p["name"] == name for p in self._personas):
            return
        desc, ok2 = QInputDialog.getText(self, "人格描述", "请输入人格描述（可留空）：")
        self._personas.append({"name": name, "description": desc.strip() if ok2 else ""})
        save_personas(self._personas)
        self._persona_combo.addItem(name)
        self._persona_combo.setCurrentText(name)

    def _add_conv(self):
        persona = self._persona_combo.currentText()
        existing = list(self._history_data.get(persona, {}).keys())
        default_name = f"对话{len(existing) + 1}"
        name, ok = QInputDialog.getText(self, "新建对话", "对话名称：", text=default_name)
        if not (ok and name.strip()):
            return
        name = name.strip()
        if persona not in self._history_data:
            self._history_data[persona] = {}
        if name not in self._history_data[persona]:
            self._history_data[persona][name] = []
            save_history(self._history_data)
        self._refresh_conv_combo()
        self._conv_combo.setCurrentText(name)

    def current_persona(self) -> str:
        return self._persona_combo.currentText()

    def current_conv(self) -> str:
        return self._conv_combo.currentText()

    # ── 右键菜单 ───────────────────────────────────────────────────
    def _persona_context_menu(self, pos):
        name = self._persona_combo.currentText()
        if not name:
            return
        menu = QMenu(self)
        act_rename = menu.addAction("重命名")
        act_edit = menu.addAction("编辑描述")
        act_delete = menu.addAction("删除")
        chosen = menu.exec_(self._persona_combo.mapToGlobal(pos))

        if chosen == act_rename:
            new_name, ok = QInputDialog.getText(self, "重命名人格", "新名称：", text=name)
            if not (ok and new_name.strip() and new_name.strip() != name):
                return
            new_name = new_name.strip()
            for p in self._personas:
                if p["name"] == name:
                    p["name"] = new_name
            save_personas(self._personas)
            if name in self._history_data:
                self._history_data[new_name] = self._history_data.pop(name)
                save_history(self._history_data)
            self._persona_combo.setItemText(self._persona_combo.currentIndex(), new_name)

        elif chosen == act_edit:
            persona_obj = next((p for p in self._personas if p["name"] == name), None)
            if not persona_obj:
                return
            desc, ok = QInputDialog.getText(
                self, "编辑描述", "人格描述：", text=persona_obj.get("description", "")
            )
            if ok:
                persona_obj["description"] = desc.strip()
                save_personas(self._personas)

        elif chosen == act_delete:
            if self._persona_combo.count() <= 1:
                return  # 至少保留一个人格
            self._personas = [p for p in self._personas if p["name"] != name]
            save_personas(self._personas)
            self._persona_combo.removeItem(self._persona_combo.currentIndex())

    def _conv_context_menu(self, pos):
        persona = self._persona_combo.currentText()
        conv = self._conv_combo.currentText()
        if not conv:
            return
        menu = QMenu(self)
        act_view = menu.addAction("查看历史")
        act_rename = menu.addAction("重命名")
        act_delete = menu.addAction("删除")
        chosen = menu.exec_(self._conv_combo.mapToGlobal(pos))

        if chosen == act_view:
            self._show_history_dialog(persona, conv)

        elif chosen == act_rename:
            new_name, ok = QInputDialog.getText(self, "重命名对话", "新名称：", text=conv)
            if not (ok and new_name.strip() and new_name.strip() != conv):
                return
            new_name = new_name.strip()
            persona_data = self._history_data.setdefault(persona, {})
            if conv in persona_data:
                persona_data[new_name] = persona_data.pop(conv)
                save_history(self._history_data)
            self._conv_combo.setItemText(self._conv_combo.currentIndex(), new_name)

        elif chosen == act_delete:
            if self._conv_combo.count() <= 1:
                return  # 至少保留一个对话
            persona_data = self._history_data.get(persona, {})
            persona_data.pop(conv, None)
            save_history(self._history_data)
            self._conv_combo.removeItem(self._conv_combo.currentIndex())

    def _show_history_dialog(self, persona: str, conv: str):
        msgs = self._history_data.get(persona, {}).get(conv, [])
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{persona} · {conv}")
        dlg.resize(480, 400)
        layout = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setStyleSheet("font-family: 'Microsoft YaHei', sans-serif; font-size: 13px;")
        lines = []
        for msg in msgs:
            role = "你" if msg.get("role") == "user" else "nuero"
            ts = msg.get("timestamp", "")
            prefix = f"[{ts}] " if ts else ""
            lines.append(f"{prefix}【{role}】\n{msg.get('content', '')}")
        te.setPlainText("\n\n".join(lines) if lines else "（暂无对话记录）")
        layout.addWidget(te)
        dlg.exec_()

    # ── 发送 ──────────────────────────────────────────────────────
    def _send(self):
        text = self._input.text().strip()
        if not text or self._is_generating:
            return
        self._input.clear()
        self._is_generating = True
        self._input.setEnabled(False)
        self._input.setPlaceholderText("思考中…")
        self._answer_buf = ""
        persona = self.current_persona()
        conv = self.current_conv()
        # 保存用户消息
        self._save_message(persona, conv, "user", text)
        self.send_requested.emit(text, persona, conv)

    def on_text_delta(self, delta: str):
        self._answer_buf += delta

    def on_answer_done(self, text: str, motion: str, describe: str):
        self._is_generating = False
        self._input.setEnabled(True)
        self._input.setPlaceholderText("说点什么… (Enter 发送)")
        self._input.setFocus()
        if text.strip():
            self._save_message(self.current_persona(), self.current_conv(), "assistant", text.strip())

    def on_error(self, msg: str):
        self._is_generating = False
        self._input.setEnabled(True)
        self._input.setPlaceholderText("说点什么… (Enter 发送)")

    # ── 拖拽移动 ──────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
        event.accept()

    def _save_message(self, persona: str, conv: str, role: str, content: str):
        if persona not in self._history_data:
            self._history_data[persona] = {}
        if conv not in self._history_data[persona]:
            self._history_data[persona][conv] = []
        self._history_data[persona][conv].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        save_history(self._history_data)

    def get_history(self, persona: str, conv: str) -> list:
        return self._history_data.get(persona, {}).get(conv, [])


# ── 气泡组件 ─────────────────────────────────────────────────────────
class SpeechBubble(QLabel):
    _CSS = """
        QLabel {
            background: white;
            border: 2.5px solid #1c1c2e;
            border-radius: 12px;
            padding: 8px 13px;
            font-size: 13px;
            font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
            color: #1a1a1a;
        }
    """

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setStyleSheet(self._CSS)
        self.setWordWrap(True)
        self.setMaximumWidth(220)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_at(self, text: str, pos: QPoint, duration_ms: int = BUBBLE_DURATION_MS):
        self.setText(text)
        self.adjustSize()
        self.move(pos)
        self.show()
        self._timer.start(duration_ms)


# ── 活跃窗口监控线程 ──────────────────────────────────────────────────
class ActiveWindowWatcher(QThread):
    window_changed = pyqtSignal(str)

    def __init__(self, interval_s=WATCHER_INTERVAL_S, cooldown_s=WATCHER_COOLDOWN_S):
        super().__init__()
        self._interval = interval_s
        self._cooldown = cooldown_s
        self._running = False
        self._last_title = ""
        self._last_emit = 0.0

    def run(self):
        self._running = True
        while self._running:
            title = self._active_title()
            now = time.time()
            if (title
                    and title != self._last_title
                    and now - self._last_emit > self._cooldown):
                self._last_title = title
                self._last_emit = now
                self.window_changed.emit(title)
            # 用短间隔循环代替长 sleep，以便 _running=False 时快速退出
            for _ in range(int(self._interval / 0.3)):
                if not self._running:
                    return
                time.sleep(0.3)

    @staticmethod
    def _active_title() -> str:
        system = platform.system()
        try:
            if system == "Windows":
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                return buf.value
            elif system == "Darwin":
                r = subprocess.run(
                    ["osascript", "-e",
                     "tell application \"System Events\" to get name of "
                     "first process whose frontmost is true"],
                    capture_output=True, text=True, timeout=2,
                )
                return r.stdout.strip()
            else:
                r = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True, text=True, timeout=1,
                )
                return r.stdout.strip()
        except Exception:
            return ""


# ── LLM 气泡评论线程 ──────────────────────────────────────────────────
class LLMWorker(QThread):
    result_ready = pyqtSignal(str)

    _SYS = (
        "你是一个可爱的二次元桌宠。用户正在切换窗口，你悄悄观察到了。"
        "只用一句话、不超过10个字吐槽或调侃，禁止换行，禁止多句。"
    )

    def __init__(self, context: str, api_key: str):
        super().__init__()
        self._ctx = context
        self._key = api_key

    def run(self):
        if not self._key or not _openai_ok:
            return
        try:
            client = OpenAI(api_key=self._key, base_url="https://api.deepseek.com/v1")
            resp = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": self._SYS},
                    {"role": "user",   "content": self._ctx},
                ],
                max_tokens=30,
                extra_body={"thinking": {"type": "disabled"}},
            )
            msg = resp.choices[0].message
            text = (msg.content or "").strip()
            if text:
                self.result_ready.emit(text)
        except Exception as e:
            print(f"[LLM] {e}")


# ── 宠物主窗口 ───────────────────────────────────────────────────────
class PetWindow(QWidget):

    def __init__(self):
        super().__init__()
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        if platform.system() == "Windows":
            flags |= Qt.WindowDoesNotAcceptFocus
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(PET_W, PET_H)

        # 图像状态
        self.current_image_path = Path(IMAGE_BASE_REL_DIR, "000000.png").as_posix()
        self.image_path_queue: deque = deque()
        self.queue_lock = threading.Lock()

        # 凝视状态
        self._gaze_home: str = self.current_image_path  # 中心区时的基准节点
        self._current_gaze_zone: int = -1               # -1 = 中心区，0..N-1 = 扇区编号

        # 窗口拖动
        self._drag_pos: QPoint | None = None

        # LLM 气泡状态
        self._window_change_time = 0.0
        self._pending_text: str | None = None
        self._llm_workers: list = []

        # Agent 相关
        self._agents: dict = {}   # key: (persona_name, conv_name)
        self._agent_workers: list = []
        self._personas = load_personas()
        self._navigator = None
        if _agent_ok:
            try:
                self._navigator = SemanticNavigator("nuero/labels.json")
            except FileNotFoundError:
                pass

        self._build_ui()
        self._load_pixmap(self.current_image_path)

        # 帧消费定时器
        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._consume_queue)
        self._frame_timer.start(1000 // FPS)

        # 全局鼠标追踪定时器（凝视）
        self._gaze_timer = QTimer(self)
        self._gaze_timer.timeout.connect(self._update_gaze_from_mouse)
        self._gaze_timer.start(GAZE_POLL_MS)

        # AnimationController（含空闲动画）
        if _anim_ok:
            self._anim = AnimationController({
                "get_node_path":      lambda: self.current_image_path,
                "get_track_points":   lambda: models.NODE_POSITIONS.get(self.current_image_path, []),
                "get_image_center":   self._get_image_center,
                "add_paths_to_queue": self._anim_add_to_queue,
                "is_queue_empty":     lambda: len(self.image_path_queue) == 0,
                "on_status":          lambda msg: None,
                "on_complete":        lambda: None,
            })
            self._anim.set_idle_enabled(True)
            self._anim.set_return_to_center(False)  # 暂时关闭回中，观察效果
            self._anim_timer = QTimer(self)
            self._anim_timer.timeout.connect(self._anim.tick)
            self._anim_timer.start(ANIM_TICK_MS)
        else:
            self._anim = None
            self._idle_timer = QTimer(self)
            self._idle_timer.timeout.connect(self._idle_step)
            self._idle_timer.start(IDLE_INTERVAL_MS)

        # 活跃窗口监控
        self._watcher = ActiveWindowWatcher()
        self._watcher.window_changed.connect(self._on_window_changed)
        self._watcher.start()

        # 气泡 & 输入条
        self._bubble = SpeechBubble()
        self._input_bar = InputBar()
        self._input_bar.send_requested.connect(self._on_send_requested)
        self._input_bar.bar_closed.connect(self._on_chat_bar_closed)
        self._input_bar._input.textChanged.connect(self._on_chat_input_changed)

        # 对话模式：打开输入条时暂停窗口监测，10s 无输入自动恢复
        self._chat_mode = False
        self._chat_idle_timer = QTimer(self)
        self._chat_idle_timer.setSingleShot(True)
        self._chat_idle_timer.timeout.connect(self._on_chat_idle)

    # ─── UI ──────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._img_label = QLabel(self)
        self._img_label.setFixedSize(PET_W, PET_H)
        self._img_label.setScaledContents(True)
        layout.addWidget(self._img_label)

    # ─── 图像加载 ─────────────────────────────────────────────────
    def _load_pixmap(self, rel_path: str):
        abs_path = os.path.abspath(rel_path)
        if not os.path.exists(abs_path):
            return
        px = QPixmap(abs_path)
        if not px.isNull():
            self._img_label.setPixmap(px)
            self.current_image_path = Path(rel_path).as_posix()

    # ─── 帧队列消费 ───────────────────────────────────────────────
    def _consume_queue(self):
        with self.queue_lock:
            if not self.image_path_queue:
                return
            nxt = self.image_path_queue.popleft()
        if nxt != self.current_image_path:
            self._load_pixmap(nxt)

    # ─── AnimationController 回调 ─────────────────────────────────
    def _get_image_center(self):
        track = models.NODE_POSITIONS.get(self.current_image_path, [])
        if not track:
            return (IMG_W / 2, 0)
        xs = [p[0] for p in track]
        ys = [p[1] for p in track]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _anim_add_to_queue(self, paths: list) -> int:
        with self.queue_lock:
            for p in paths:
                self.image_path_queue.append(p)
        return len(paths)

    # ─── 扇区凝视 ────────────────────────────────────────────────
    def _update_gaze_from_mouse(self):
        if self._drag_pos is not None:
            return
        if self._anim is not None and self._anim.is_active:
            return

        cursor = QCursor.pos()
        center = self.frameGeometry().center()
        dx = cursor.x() - center.x()
        dy = cursor.y() - center.y()
        dist = (dx * dx + dy * dy) ** 0.5

        # 计算当前扇区
        if dist < GAZE_CENTER_THRESHOLD:
            new_zone = -1
        else:
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * math.pi
            new_zone = int(angle / (2 * math.pi) * GAZE_ZONES) % GAZE_ZONES

        # 扇区未变化，无需处理
        if new_zone == self._current_gaze_zone:
            return
        self._current_gaze_zone = new_zone

        if new_zone == -1:
            # 回到中心区：更新 home 节点
            if self._anim is None or not self._anim.is_active:
                self._gaze_home = self.current_image_path
            return

        # 使用鼠标实际方向（扇区仅防抖），限幅后转换到图像坐标
        clamped = min(dist, GAZE_ZONE_RADIUS)
        scale = IMG_W / PET_W
        target_dx = (dx / dist) * clamped * scale
        target_dy = (dy / dist) * clamped * scale

        # 在 home 节点邻域内搜索最佳匹配帧
        origin = self._gaze_home or self.current_image_path
        track = models.NODE_POSITIONS.get(origin, [])
        if not track or models.G is None or origin not in models.G:
            return

        t1 = [(x + target_dx, y + target_dy) for x, y in track]

        try:
            cands = nx.single_source_shortest_path_length(models.G, origin, cutoff=15)
        except Exception:
            return

        best, best_loss = origin, float("inf")
        for node in cands:
            pos = models.NODE_POSITIONS.get(node, [])
            if not pos or len(pos) != len(t1):
                continue
            loss = calculate_loss(t1, pos)
            if loss < best_loss:
                best_loss, best = loss, node

        if best != self.current_image_path:
            with self.queue_lock:
                self.image_path_queue.clear()
            self._load_pixmap(best)

    # ─── 空闲动画（fallback）────────────────────────────────────
    def _idle_step(self):
        with self.queue_lock:
            if self.image_path_queue:
                return

        track = models.NODE_POSITIONS.get(self.current_image_path, [])
        if not track or models.G is None or self.current_image_path not in models.G:
            return

        dx = random.uniform(-20, 20)
        dy = random.uniform(-20, 20)
        t1 = [(x + dx, y + dy) for x, y in track]
        start = self.current_image_path

        node_loss = {n: calculate_loss(t1, models.NODE_POSITIONS.get(n, []))
                     for n in models.G.nodes}
        target = min(node_loss, key=node_loss.get)
        if target == start:
            return

        try:
            fwd = nx.shortest_path(models.G, start, target)
            bwd = nx.shortest_path(models.G, target, start)
        except nx.NetworkXNoPath:
            return

        with self.queue_lock:
            for n in fwd[1:]:
                self.image_path_queue.append(n)
            for n in bwd[1:]:
                self.image_path_queue.append(n)

    # ─── 语义动作 ─────────────────────────────────────────────────
    def _on_motion_requested(self, label: str):
        if not (self._anim and self._navigator and models.G is not None):
            return
        track = models.NODE_POSITIONS.get(self.current_image_path, [])
        try:
            paths = self._navigator.navigate_to(label, self.current_image_path, track)
            if paths:
                self._anim.start_by_path(paths, label)
        except Exception as e:
            print(f"[motion] {e}")

    # ─── Agent 管理 ───────────────────────────────────────────────
    def _get_or_create_agent(self, persona_name: str, conv_name: str):
        if not _agent_ok:
            return None
        key = (persona_name, conv_name)
        if key not in self._agents:
            agent = AgentLLM(
                tool_registry=ToolRegistry(),
                api_key=DEEPSEEK_API_KEY,
            )
            # 注入人格描述
            desc = next(
                (p["description"] for p in self._personas if p["name"] == persona_name),
                "",
            )
            if desc:
                agent._history.append({"role": "user", "content": f"[人格设定] {desc}"})
                agent._history.append({"role": "assistant", "content": "好的，我明白了！"})
            # 注入已有对话历史（仅消息内容，不含 timestamp）
            for msg in self._input_bar.get_history(persona_name, conv_name):
                if msg["role"] in ("user", "assistant"):
                    agent._history.append({"role": msg["role"], "content": msg["content"]})
            self._agents[key] = agent
        return self._agents[key]

    def _on_send_requested(self, text: str, persona_name: str, conv_name: str):
        agent = self._get_or_create_agent(persona_name, conv_name)
        if agent is None:
            self._input_bar.on_error("agent_service 不可用")
            return

        self._agent_workers = [w for w in self._agent_workers if not w.isFinished()]
        worker = AgentWorker(agent, text)
        worker.text_delta.connect(self._input_bar.on_text_delta)
        worker.answer_done.connect(self._input_bar.on_answer_done)
        worker.answer_done.connect(self._on_agent_answer)
        worker.motion_requested.connect(self._on_motion_requested)
        worker.error_sig.connect(self._input_bar.on_error)
        worker.start()
        self._agent_workers.append(worker)

    def _on_agent_answer(self, text: str, motion: str, describe: str):
        # 过滤残留的 ctrl/tool 标记，只保留干净回答文本
        clean = re.sub(r"<ctrl>.*?</ctrl>", "", text, flags=re.DOTALL)
        clean = re.sub(r"\[TOOL_RESULT:.*?\]", "", clean, flags=re.DOTALL)
        # 去掉含 motion/tool 键的 JSON 对象（ctrl 块泄漏）
        # (?:[^{}]|\{[^{}]*\})* 支持一层嵌套花括号（如 tool_args:{}）
        clean = re.sub(
            r'\{(?:[^{}]|\{[^{}]*\})*"(?:motion|tool)"(?:[^{}]|\{[^{}]*\})*\}?',
            "", clean, flags=re.DOTALL
        ).strip()
        if clean:
            self._show_bubble(clean)

    # ─── 气泡位置 ─────────────────────────────────────────────────
    def _reposition_bubble(self):
        geo = self.frameGeometry()
        screen = QApplication.primaryScreen().geometry()
        bw = self._bubble.width() or 220
        # 默认右上角
        x = geo.right() + 8
        y = geo.top()
        # 超出右边界 → 左上角
        if x + bw > screen.right():
            x = geo.left() - bw - 8
        self._bubble.move(QPoint(x, y))

    # ─── 输入条位置 ───────────────────────────────────────────────
    def _reposition_input_bar(self):
        if not self._input_bar.isVisible():
            return
        geo = self.frameGeometry()
        screen = QApplication.primaryScreen().geometry()
        bw = self._input_bar.BAR_W
        bh = self._input_bar.BAR_H
        # 人物正下方，水平居中对齐
        x = geo.center().x() - bw // 2
        y = geo.bottom() + 8
        x = max(screen.left(), min(x, screen.right() - bw))
        y = max(screen.top(), min(y, screen.bottom() - bh))
        self._input_bar.move(x, y)

    # ─── 鼠标事件 ─────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
        elif event.button() == Qt.RightButton:
            self._show_context_menu(event.globalPos())
        event.accept()

    def moveEvent(self, event):
        super().moveEvent(event)
        if self._bubble.isVisible():
            self._reposition_bubble()
        self._reposition_input_bar()

    # ─── 右键菜单 ─────────────────────────────────────────────────
    def _show_context_menu(self, pos: QPoint):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 4px;
                font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
                font-size: 13px;
                color: #111827;
            }
            QMenu::item { padding: 6px 20px; border-radius: 4px; }
            QMenu::item:selected { background: #e0e7ff; color: #4338ca; }
            QMenu::separator { height: 1px; background: #e5e7eb; margin: 3px 8px; }
        """)
        act_chat = menu.addAction("聊天")
        menu.addSeparator()
        act_quit = menu.addAction("退出")
        chosen = menu.exec_(pos)
        if chosen == act_chat:
            self._open_input_bar()
        elif chosen == act_quit:
            self.close()

    def _open_input_bar(self):
        # 先定位再显示，不能依赖 isVisible() 守卫
        geo = self.frameGeometry()
        screen = QApplication.primaryScreen().geometry()
        bw = self._input_bar.BAR_W
        bh = self._input_bar.BAR_H
        x = geo.center().x() - bw // 2
        y = geo.bottom() + 8
        x = max(screen.left(), min(x, screen.right() - bw))
        y = max(screen.top(), min(y, screen.bottom() - bh))
        self._input_bar.move(x, y)
        self._input_bar.show()
        self._input_bar._input.setFocus()
        # 进入对话模式，暂停窗口监测
        self._chat_mode = True
        self._chat_idle_timer.start(10_000)  # 10s 无输入则恢复

    def _on_chat_bar_closed(self):
        self._chat_mode = False
        self._chat_idle_timer.stop()

    def _on_chat_input_changed(self, text: str):
        # 用户正在输入，重置 10s 计时
        if self._chat_mode:
            self._chat_idle_timer.start(10_000)

    def _on_chat_idle(self):
        # 10s 内既没有输入也没有在生成，恢复窗口监测
        if not self._input_bar._is_generating:
            self._chat_mode = False

    # ─── 活跃窗口 → LLM → 气泡 ───────────────────────────────────
    def _on_window_changed(self, title: str):
        print(f"[窗口] -> {title}")
        if self._chat_mode:
            return  # 对话模式中暂停窗口监测
        if not DEEPSEEK_API_KEY:
            print("[窗口] DEEPSEEK_API_KEY 未配置，跳过气泡")
            return
        self._window_change_time = time.time()
        self._pending_text = None
        self._llm_workers = [w for w in self._llm_workers if w.isRunning()]
        w = LLMWorker(f"用户正在使用：{title}", DEEPSEEK_API_KEY)
        w.result_ready.connect(self._on_llm_result)
        w.start()
        self._llm_workers.append(w)

    def _on_llm_result(self, text: str):
        if self._chat_mode:
            return  # 对话模式中丢弃窗口气泡
        self._pending_text = text
        elapsed = time.time() - self._window_change_time
        delay_ms = max(0, int((LLM_DELAY_S - elapsed) * 1000))
        QTimer.singleShot(delay_ms, lambda t=text: self._show_if_fresh(t))

    def _show_if_fresh(self, text: str):
        if self._pending_text == text:
            self._show_bubble(text)

    def _show_bubble(self, text: str):
        self._bubble.setText(text)
        self._bubble.show()          # 先 show 再 adjustSize，确保尺寸正确
        self._bubble.adjustSize()
        self._reposition_bubble()
        self._bubble.raise_()
        self._bubble._timer.start(BUBBLE_DURATION_MS)

    # ─── 关闭清理 ─────────────────────────────────────────────────
    def closeEvent(self, event):
        # 先停定时器，防止关闭期间继续触发
        self._frame_timer.stop()
        self._gaze_timer.stop()
        if self._anim is not None:
            self._anim_timer.stop()
        elif hasattr(self, "_idle_timer"):
            self._idle_timer.stop()

        # 通知监控线程退出
        self._watcher._running = False
        self._watcher.quit()

        # 通知所有 worker 退出
        for w in self._agent_workers + self._llm_workers:
            if w.isRunning():
                w.quit()

        self._bubble.hide()
        self._input_bar.hide()
        event.accept()

        # 强制结束进程：QThread 在 Windows 上对阻塞网络请求无法 terminate，
        # Python 会等待所有非 daemon 线程结束导致卡死。os._exit 直接退出。
        import os as _os
        _os._exit(0)


# ── 首次启动标注向导 ──────────────────────────────────────────────────
class SetupWindow(QWidget):
    done = pyqtSignal()

    _SERVER_URL = "http://localhost:8001"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live2Diff 桌宠 — 初始化设置")
        self.setFixedWidth(420)
        self._proc = None
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._check_server)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("首次使用 · 角色标注设置")
        title.setStyleSheet("font-size:15px; font-weight:bold; margin-bottom:4px;")
        layout.addWidget(title)

        desc = QLabel(
            "桌宠需要先通过 CoTracker 分析角色动作。\n\n"
            "步骤：\n"
            "  1. 点击下方按钮，自动启动标注服务并打开浏览器\n"
            "  2. 在网页中点击 <b>Create</b>，在角色脸部放置若干控制点\n"
            "  3. 点击 <b>CoTracker</b> 按钮生成轨迹（需要约 1–2 分钟）\n"
            "  4. 完成后回到这里，点击 <b>启动宠物</b>"
        )
        desc.setTextFormat(Qt.RichText)
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size:12px; color:#333; line-height:1.5;")
        layout.addWidget(desc)

        self.btn_launch = QPushButton("启动标注服务并打开浏览器")
        self.btn_launch.setStyleSheet(
            "background:#7c3aed; color:white; padding:8px; border-radius:6px;"
            "font-size:13px; font-weight:bold;")
        self.btn_launch.clicked.connect(self._start_server)
        layout.addWidget(self.btn_launch)

        self.btn_done = QPushButton("我已完成标注，启动宠物")
        self.btn_done.setEnabled(False)
        self.btn_done.setStyleSheet(
            "background:#16a34a; color:white; padding:8px; border-radius:6px;"
            "font-size:13px; font-weight:bold;")
        self.btn_done.clicked.connect(self._finish)
        layout.addWidget(self.btn_done)

        btn_skip = QPushButton("跳过（无动画直接启动）")
        btn_skip.setStyleSheet(
            "background:#e5e7eb; color:#555; padding:6px; border-radius:6px; font-size:12px;")
        btn_skip.clicked.connect(self._skip)
        layout.addWidget(btn_skip)

        self.status = QLabel("")
        self.status.setStyleSheet("font-size:11px; color:#888;")
        layout.addWidget(self.status)

    def _start_server(self):
        self.btn_launch.setEnabled(False)
        self.btn_launch.setText("服务启动中，请稍候…")
        self.status.setText("正在启动 main_chat_wu.py …")
        self._proc = subprocess.Popen([sys.executable, "main_chat_wu.py"], env=os.environ.copy())
        self._poll_timer.start(600)

    def _check_server(self):
        try:
            urllib.request.urlopen(self._SERVER_URL, timeout=1)
            self._poll_timer.stop()
            self.btn_launch.setText("标注服务已启动")
            self.status.setText(f"浏览器已打开 {self._SERVER_URL}，请完成标注后点击上方按钮。")
            webbrowser.open(self._SERVER_URL)
            self.btn_done.setEnabled(True)
        except Exception:
            pass

    def _finish(self):
        self._stop_server()
        _preload_track_caches()
        self.done.emit()
        self.close()

    def _skip(self):
        self._stop_server()
        self.done.emit()
        self.close()

    def _stop_server(self):
        self._poll_timer.stop()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc = None

    def closeEvent(self, event):
        self._stop_server()
        event.accept()


# ── 入口 ─────────────────────────────────────────────────────────────
def main():
    print("Live2Diff 桌宠")
    print(f"  角色目录 : {os.path.abspath(BASE_PATH)}")
    print(f"  DeepSeek : {'已配置' if DEEPSEEK_API_KEY else '未配置（对话/气泡功能不可用）'}")
    print(f"  Agent    : {'可用' if _agent_ok else '不可用（缺少 agent_service）'}")
    print(f"  动画控制 : {'AnimationController' if _anim_ok else '简单模式'}")
    print(f"  操作说明 : 左键拖=移动  右键单击=菜单  鼠标移动=凝视追踪")

    app = QApplication(sys.argv)
    app.setApplicationName("Live2Diff Pet")

    _pet: list = []

    def launch_pet():
        pet = PetWindow()
        pet.move(100, 100)
        pet.show()
        _pet.append(pet)

    if check_tracks_exist():
        launch_pet()
    else:
        print("未检测到 track 数据，打开标注向导")
        setup = SetupWindow()
        setup.done.connect(launch_pet)
        setup.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

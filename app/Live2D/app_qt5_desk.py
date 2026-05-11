"""
桌宠版 Live2Diff —— app_qt5_desk.py
从 app_qt5.py 改造，专注于桌面宠物体验：
  - 透明无边框悬浮窗
  - 左键拖：移动宠物位置
  - 右键拖：控制凝视方向，松手自动回中
  - 空闲随机扰动动画（±20px，小幅呼吸感）
  - 监控活跃窗口 → 预取 LLM → 延迟 20s 显示气泡
  - 右键菜单：手动聊天 / 退出

启动（从 app/Live2D/ 目录）:
  DEEPSEEK_API_KEY=<key> python app_qt5_desk.py
"""

import os
import sys
import time
import threading
import random
import subprocess
import platform
import webbrowser
import urllib.request
from pathlib import Path
from collections import deque

import numpy as np
import networkx as nx

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QMenu, QPushButton,
    QVBoxLayout, QInputDialog,
)

try:
    from openai import OpenAI
    _openai_ok = True
except ImportError:
    _openai_ok = False

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

PET_W, PET_H = 300, 410      # 宠物窗口像素大小
IMG_W = 480                   # nuero 原始帧宽（坐标系换算用）
FPS = 30                      # 帧消费频率
IDLE_INTERVAL_MS = 2500       # 空闲动画触发间隔
BUBBLE_DURATION_MS = 8000     # 气泡显示时长
LLM_DELAY_S = 20              # 窗口切换后延迟多少秒显示气泡
WATCHER_INTERVAL_S = 3        # 活跃窗口检测间隔
WATCHER_COOLDOWN_S = 60       # 同一窗口最小触发间隔


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
    print(f"✅ track 缓存预加载：{loaded} 个节点")

_preload_track_caches()


# ── 工具 ─────────────────────────────────────────────────────────────
def calculate_loss(points, node_positions):
    if not points or not node_positions or len(points) != len(node_positions):
        return float("inf")
    p = np.array(points, dtype=np.float32)
    n = np.array(node_positions, dtype=np.float32)
    return float(np.mean((p - n) ** 2))


# ── 活跃窗口监控线程 ──────────────────────────────────────────────────
class ActiveWindowWatcher(QThread):
    """每隔 interval_s 秒查询一次当前活跃窗口标题，变化时发出信号。
    内置 cooldown 防止同一窗口短时间内重复触发。
    依赖 xdotool（Linux）；未安装时静默跳过。
    """
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
            time.sleep(self._interval)

    def stop(self):
        self._running = False
        self.wait()

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
            else:  # Linux
                r = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True, text=True, timeout=1,
                )
                return r.stdout.strip()
        except Exception:
            return ""


# ── LLM 评论线程（同步调用，运行于子线程）────────────────────────────
class LLMWorker(QThread):
    result_ready = pyqtSignal(str)

    _SYS = (
        "你是一个可爱的二次元桌宠。你悄悄观察到用户正在使用电脑。"
        "用一句话（不超过18字，可加 emoji）活泼地评论或提问，俏皮感强。"
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
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self._SYS},
                    {"role": "user",   "content": self._ctx},
                ],
                max_tokens=60,
            )
            text = resp.choices[0].message.content.strip()
            if text:
                self.result_ready.emit(text)
        except Exception as e:
            print(f"[LLM] {e}")


# ── 气泡组件 ─────────────────────────────────────────────────────────
class SpeechBubble(QLabel):
    """漫画风格悬浮气泡，独立于主窗口存在。"""

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

    def popup(self, text: str, global_pos: QPoint, duration_ms: int = BUBBLE_DURATION_MS):
        self.setText(text)
        self.adjustSize()
        self.move(global_pos)
        self.show()
        self._timer.start(duration_ms)


# ── 宠物主窗口 ───────────────────────────────────────────────────────
class PetWindow(QWidget):

    def __init__(self):
        super().__init__()
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        if platform.system() == "Windows":
            # Qt.Tool 在部分 Windows 版本不能完全隐藏任务栏图标，追加此 flag
            flags |= Qt.WindowDoesNotAcceptFocus
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(PET_W, PET_H)

        # 图像状态
        self.current_image_path = Path(IMAGE_BASE_REL_DIR, "000000.png").as_posix()
        self.image_path_queue: deque = deque()
        self.queue_lock = threading.Lock()

        # 凝视状态
        self._gaze_active = False
        self._gaze_origin: str | None = None
        self._right_press_pos: QPoint | None = None
        self._right_moved = False

        # 窗口拖动
        self._drag_pos: QPoint | None = None

        # LLM 状态
        self._window_change_time = 0.0
        self._pending_text: str | None = None
        self._llm_workers: list = []   # 持有引用，防止 GC 回收运行中线程

        self._build_ui()
        self._load_pixmap(self.current_image_path)

        # 帧消费定时器
        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._consume_queue)
        self._frame_timer.start(1000 // FPS)

        # 空闲动画定时器
        self._idle_timer = QTimer(self)
        self._idle_timer.timeout.connect(self._idle_step)
        self._idle_timer.start(IDLE_INTERVAL_MS)

        # 活跃窗口监控
        self._watcher = ActiveWindowWatcher()
        self._watcher.window_changed.connect(self._on_window_changed)
        self._watcher.start()

        # 气泡
        self._bubble = SpeechBubble()

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

    # ─── 空闲动画 ─────────────────────────────────────────────────
    def _idle_step(self):
        if self._gaze_active:
            return
        with self.queue_lock:
            if self.image_path_queue:   # 上一段动画未播完，跳过
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

    # ─── 凝视控制 ─────────────────────────────────────────────────
    def _update_gaze(self, screen_delta: QPoint):
        """根据右键拖动偏移实时切换帧，在凝视起点的 BFS 邻域内搜索。"""
        origin = self._gaze_origin or self.current_image_path
        track = models.NODE_POSITIONS.get(origin, [])
        if not track or models.G is None or origin not in models.G:
            return

        # 屏幕像素 → 图像像素，乘以灵敏度系数 0.4
        scale = (IMG_W / PET_W) * 0.4
        t1 = [(x + screen_delta.x() * scale,
               y + screen_delta.y() * scale) for x, y in track]

        try:
            cands = nx.single_source_shortest_path_length(models.G, origin, cutoff=8)
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
            self._load_pixmap(best)

    def _return_gaze(self):
        """松开右键后，沿图谱把当前帧导航回凝视起点。"""
        origin = self._gaze_origin
        cur = self.current_image_path
        if not origin or origin == cur:
            return
        if models.G is None or origin not in models.G or cur not in models.G:
            return
        try:
            path = nx.shortest_path(models.G, cur, origin)
        except nx.NetworkXNoPath:
            return
        with self.queue_lock:
            for n in path[1:]:
                self.image_path_queue.append(n)

    # ─── 鼠标事件 ─────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
        elif event.button() == Qt.RightButton:
            self._right_press_pos = event.pos()
            self._right_moved = False
            self._gaze_active = True
            self._gaze_origin = self.current_image_path
            with self.queue_lock:
                self.image_path_queue.clear()   # 清掉 idle 残留，凝视立即生效
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
        elif (event.buttons() & Qt.RightButton
              and self._right_press_pos is not None):
            delta = event.pos() - self._right_press_pos
            if abs(delta.x()) > 4 or abs(delta.y()) > 4:
                self._right_moved = True
            if self._right_moved:
                self._update_gaze(delta)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
        elif event.button() == Qt.RightButton:
            if not self._right_moved:
                self._show_context_menu(event.globalPos())
            else:
                self._return_gaze()
            self._right_press_pos = None
            self._right_moved = False
            self._gaze_active = False
        event.accept()

    # ─── 右键菜单 ─────────────────────────────────────────────────
    def _show_context_menu(self, pos: QPoint):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background:white; border:1px solid #ddd;
                    border-radius:6px; padding:4px; font-size:12px; }
            QMenu::item { padding:5px 18px; border-radius:4px; }
            QMenu::item:selected { background:#f3e8ff; color:#7e22ce; }
        """)
        act_chat = menu.addAction("💬 说点什么…")
        menu.addSeparator()
        act_quit = menu.addAction("退出")
        chosen = menu.exec_(pos)
        if chosen == act_chat:
            self._manual_chat()
        elif chosen == act_quit:
            self.close()

    def _manual_chat(self):
        text, ok = QInputDialog.getText(self, "和宠物说话", "说点什么：")
        if not (ok and text.strip()):
            return
        w = LLMWorker(f"用户对你说：{text.strip()}", DEEPSEEK_API_KEY)
        w.result_ready.connect(self._show_bubble)
        w.start()
        self._llm_workers.append(w)

    # ─── 活跃窗口 → LLM → 气泡 ───────────────────────────────────
    def _on_window_changed(self, title: str):
        print(f"[窗口] → {title}")
        self._window_change_time = time.time()
        self._pending_text = None
        # 清理已结束的旧线程
        self._llm_workers = [w for w in self._llm_workers if w.isRunning()]
        w = LLMWorker(f"用户正在使用：{title}", DEEPSEEK_API_KEY)
        w.result_ready.connect(self._on_llm_result)
        w.start()
        self._llm_workers.append(w)

    def _on_llm_result(self, text: str):
        self._pending_text = text
        elapsed = time.time() - self._window_change_time
        delay_ms = max(0, int((LLM_DELAY_S - elapsed) * 1000))
        # 用默认参数捕获当前 text，防止闭包延迟绑定覆盖
        QTimer.singleShot(delay_ms, lambda t=text: self._show_if_fresh(t))

    def _show_if_fresh(self, text: str):
        """只展示最新那条 LLM 结果，丢弃已被新窗口覆盖的旧结果。"""
        if self._pending_text == text:
            self._show_bubble(text)

    def _show_bubble(self, text: str):
        geo = self.frameGeometry()
        bubble_w = 240
        # 优先显示在角色左侧，空间不足则显示在右侧
        x = geo.left() - bubble_w if geo.left() >= bubble_w + 10 else geo.right() + 10
        self._bubble.popup(text, QPoint(x, geo.top() + 20))

    # ─── 关闭清理 ─────────────────────────────────────────────────
    def closeEvent(self, event):
        self._watcher.stop()
        self._frame_timer.stop()
        self._idle_timer.stop()
        self._bubble.hide()
        event.accept()


# ── 首次启动标注向导 ──────────────────────────────────────────────────
class SetupWindow(QWidget):
    """首次使用时引导用户完成 CoTracker 标注的向导窗口。
    流程：启动 main_chat_wu.py → 打开浏览器 → 用户标注 → 点击完成 → 关闭服务 → 启动宠物。
    """
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

        title = QLabel("🐾  首次使用 · 角色标注设置")
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

        self.btn_launch = QPushButton("🚀  启动标注服务并打开浏览器")
        self.btn_launch.setStyleSheet(
            "background:#7c3aed; color:white; padding:8px; border-radius:6px;"
            "font-size:13px; font-weight:bold;")
        self.btn_launch.clicked.connect(self._start_server)
        layout.addWidget(self.btn_launch)

        self.btn_done = QPushButton("✅  我已完成标注，启动宠物")
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
        self.btn_launch.setText("⏳  服务启动中，请稍候…")
        self.status.setText("正在启动 main_chat_wu.py …")
        env = os.environ.copy()
        self._proc = subprocess.Popen(
            [sys.executable, "main_chat_wu.py"],
            env=env,
        )
        self._poll_timer.start(600)

    def _check_server(self):
        try:
            urllib.request.urlopen(self._SERVER_URL, timeout=1)
            self._poll_timer.stop()
            self.btn_launch.setText("✅  标注服务已启动")
            self.status.setText(f"浏览器已打开 {self._SERVER_URL}，请完成标注后点击上方按钮。")
            webbrowser.open(self._SERVER_URL)
            self.btn_done.setEnabled(True)
        except Exception:
            pass  # 还没就绪，继续轮询

    def _finish(self):
        self._stop_server()
        _preload_track_caches()   # 重新读取刚生成的 track 文件
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
    print("🐾 Live2Diff 桌宠")
    print(f"  角色目录 : {os.path.abspath(BASE_PATH)}")
    print(f"  DeepSeek : {'已配置' if DEEPSEEK_API_KEY else '未配置（气泡功能不可用）'}")
    print(f"  操作说明 : 左键拖=移动  右键拖=凝视  右键单击=菜单")

    app = QApplication(sys.argv)
    app.setApplicationName("Live2Diff Pet")

    _pet: list = []   # 用列表持有引用，避免局部变量被 GC

    def launch_pet():
        pet = PetWindow()
        pet.move(100, 100)
        pet.show()
        _pet.append(pet)

    if check_tracks_exist():
        launch_pet()
    else:
        print("⚠️  未检测到 track 数据，打开标注向导")
        setup = SetupWindow()
        setup.done.connect(launch_pet)
        setup.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

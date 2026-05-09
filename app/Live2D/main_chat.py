"""
LLM 驱动的虚拟主播 —— PyQt5 版
用 LLM(DeepSeek) 控制头部运动，在图谱中导航切换图像，实现虚拟主播交互。

流程：
   用户输入 → LLM → JSON(motion_x1~4, motion_y1~4, answer, describe)
     ├─ motion 1~4: T0 + (dx,dy) → 找最佳节点 → 导航队列（快速消费）
     └─ 回中: shortest_path 回到对话起点
"""

import os
import sys
import json
import threading
from pathlib import Path
from collections import deque

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QPainterPath
from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QLineEdit,
    QGroupBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsPathItem,
    QMessageBox,
)

import models
import llm_service
from animation import AnimationController

# ----------------------------------------------------------------------
# 路径配置
# ----------------------------------------------------------------------
BASE_PATH = "nuero"
models.BASE_DIR = BASE_PATH
models.GRAPH_PATH = os.path.join(BASE_PATH, "graph.txt")
models.IMAGES_DIR = os.path.join(BASE_PATH, "images")
models.TRACKS_DIR = os.path.join(BASE_PATH, "track")
models.load_graph()

IMAGE_BASE_REL_DIR = os.path.join(BASE_PATH, "images", "video_00027")
IMAGE_BASE_PHYSICAL_DIR = os.path.abspath(IMAGE_BASE_REL_DIR)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

FPS = 30
DEPTH_LIMIT = 10
QUEUE_LEN = 10
MAX_QUEUE_ADD = 10
MOTION_CHECK_INTERVAL = 80  # ms


# ----------------------------------------------------------------------
# 预加载 track 缓存
# ----------------------------------------------------------------------
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
    print(f"track 缓存预加载完成：{loaded} 个节点有坐标数据")


_preload_track_caches()


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def generate_image_paths():
    if not os.path.exists(IMAGE_BASE_PHYSICAL_DIR):
        print(f"图片物理目录不存在: {IMAGE_BASE_PHYSICAL_DIR}")
        return []
    rels = []
    for fn in os.listdir(IMAGE_BASE_PHYSICAL_DIR):
        if fn.lower().endswith(IMAGE_EXTENSIONS):
            rels.append(Path(IMAGE_BASE_REL_DIR, fn).as_posix())
    rels.sort()
    return rels


# ----------------------------------------------------------------------
# 图像画布（仅展示，无点位编辑）
# ----------------------------------------------------------------------
class ChatCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setBackgroundBrush(self._scene.palette().window())
        self.pixmap_item = None

        self.on_sample = None
        self.on_gesture_complete = None
        self._recording = False
        self._trajectory = []
        self._trajectory_item = None
        self._trajectory_path = QPainterPath()
        self._last_mouse_pos = None
        self._sample_timer = QTimer(self)
        self._sample_timer.timeout.connect(self._on_sample_tick)

    def load_image(self, abs_path):
        if not os.path.exists(abs_path):
            print(f"图片不存在: {abs_path}")
            return False
        pixmap = QPixmap(abs_path)
        if pixmap.isNull():
            print(f"QPixmap 加载失败: {abs_path}")
            return False
        self._scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setZValue(-1)
        self._scene.addItem(self.pixmap_item)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._trajectory_item = None
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self._recording = True
            self._trajectory = [(scene_pos.x(), scene_pos.y())]
            self._last_mouse_pos = event.pos()
            self._trajectory_path = QPainterPath()
            self._trajectory_path.moveTo(scene_pos)
            pen = QPen(
                QColor(212, 160, 138, 128), 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin
            )
            self._trajectory_item = QGraphicsPathItem(self._trajectory_path)
            self._trajectory_item.setPen(pen)
            self._trajectory_item.setZValue(1)
            self._scene.addItem(self._trajectory_item)
            self._sample_timer.start(50)
            if self.on_sample:
                self.on_sample(scene_pos.x(), scene_pos.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._recording:
            self._last_mouse_pos = event.pos()
            scene_pos = self.mapToScene(event.pos())
            self._trajectory.append((scene_pos.x(), scene_pos.y()))
            self._trajectory_path.lineTo(scene_pos)
            if self._trajectory_item:
                self._trajectory_item.setPath(self._trajectory_path)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._recording and event.button() == Qt.LeftButton:
            self._recording = False
            self._sample_timer.stop()
            if self._trajectory_item:
                self._scene.removeItem(self._trajectory_item)
                self._trajectory_item = None
            if self.on_gesture_complete:
                self.on_gesture_complete(self._trajectory)
            self._trajectory = []
        super().mouseReleaseEvent(event)

    def _on_sample_tick(self):
        if not self._recording or self._last_mouse_pos is None:
            return
        scene_pos = self.mapToScene(self._last_mouse_pos)
        if self.on_sample:
            self.on_sample(scene_pos.x(), scene_pos.y())


# ----------------------------------------------------------------------
# 主窗口
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    # 跨线程信号：后台 LLM 线程 → 主线程
    llm_response = pyqtSignal(object)
    llm_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nuero-sama Virtual Anchor (LLM Driven)")
        self.resize(1000, 860)

        # LLM 会话（保留上下文）
        self.llm = llm_service.llm_session()

        # 连接跨线程信号
        self.llm_response.connect(self._on_llm_response)
        self.llm_error.connect(self._on_llm_error)

        # 状态变量
        self.current_image_path = Path(IMAGE_BASE_REL_DIR, "000000.png").as_posix()
        self.image_path_queue = deque()
        self.queue_lock = threading.Lock()
        self._current_t0 = []
        self._llm_busy = False

        self._build_ui()

        self.canvas.on_sample = self._on_mouse_sample
        self.canvas.on_gesture_complete = self._on_gesture_complete

        # 动画控制器
        self.anim = AnimationController(
            {
                "get_node_path": lambda: (self.image_path_queue[-1]
         if self.image_path_queue else self.current_image_path),
                "get_track_points": lambda: self._current_t0,
                "get_image_center": lambda: self._get_img_center(),
                "is_queue_empty": lambda: self._is_queue_empty(),
                "add_paths_to_queue": lambda paths: self._add_paths_to_queue(paths),
                "on_status": lambda text: self.status_label.setText(text),
                "on_complete": lambda: self._on_anim_complete(),
            }
        )

        # 加载初始图像
        self._load_image_from_rel(self.current_image_path)
        self._apply_cached_track()

        self.anim.set_idle_enabled(True)

        # 图像切换定时器（持续运行）
        self.image_timer = QTimer(self)
        self.image_timer.timeout.connect(self._fetch_current_image)
        self.image_timer.start(1000 // FPS)

        # 动画进度检查定时器
        self.motion_check_timer = QTimer(self)
        self.motion_check_timer.timeout.connect(self.anim.tick)
        self.motion_check_timer.start(MOTION_CHECK_INTERVAL)

    # ----- UI 构建 -----
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 状态栏
        self.status_label = QLabel("Ready | Nuero-sama 虚拟主播")
        self.status_label.setStyleSheet("color:#555;padding:4px;font-size:13px;")
        root.addWidget(self.status_label)

        # 图像画布
        self.canvas = ChatCanvas()
        root.addWidget(self.canvas, 3)

        # 聊天面板
        chat_group = QGroupBox("聊天")
        chat_layout = QVBoxLayout(chat_group)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet(
            "font-family:'Segoe UI',sans-serif;font-size:14px;"
        )
        self.chat_history.setMinimumHeight(200)
        chat_layout.addWidget(self.chat_history, 1)

        input_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("输入消息，按 Enter 发送...")
        self.chat_input.returnPressed.connect(self._send_chat)
        self.send_btn = QPushButton("发送")
        self.send_btn.setStyleSheet(
            "background:#6f42c1;color:white;padding:6px 16px;border-radius:4px;font-weight:bold;"
        )
        self.send_btn.clicked.connect(self._send_chat)
        input_row.addWidget(self.chat_input, 1)
        input_row.addWidget(self.send_btn)

        # 额外功能按钮
        self.clear_btn = QPushButton("清除聊天")
        self.clear_btn.clicked.connect(self._clear_chat)
        input_row.addWidget(self.clear_btn)

        chat_layout.addLayout(input_row)
        root.addWidget(chat_group, 2)

        # 初始欢迎消息
        self._add_system_message("欢迎！我是 Nuero-sama，你的虚拟主播！和我聊天吧！")

    # ----- 聊天 -----
    def _add_chat_message(self, sender, text, description=""):
        color = "#9b59b6" if sender == "Nuero-sama" else "#2c3e50"
        align = "left" if sender == "Nuero-sama" else "right"
        desc_html = ""
        if description:
            desc_html = f'<p style="color:#888;font-style:italic;font-size:12px;margin:0 0 4px 0;">动作: {description}</p>'
        html = (
            f'<div style="margin:6px 0;text-align:{align};">'
            f"{desc_html}"
            f'<span style="background:{color};color:white;padding:4px 12px;'
            f'border-radius:12px;display:inline-block;max-width:80%;">'
            f"<b>{sender}</b>: {text}</span></div>"
        )
        self.chat_history.append(html)
        # 滚动到底部
        sb = self.chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _add_system_message(self, text):
        html = (
            f'<div style="margin:6px 0;text-align:center;">'
            f'<span style="color:#999;font-size:13px;">{text}</span></div>'
        )
        self.chat_history.append(html)
        sb = self.chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _clear_chat(self):
        self.chat_history.clear()
        self._add_system_message("聊天已清除")

    def _send_chat(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        if self.anim.is_active:
            self._add_system_message("正在播放动画，请稍候...")
            return
        if self._llm_busy:
            self._add_system_message("正在思考中，请稍候...")
            return

        self._llm_busy = True
        self.chat_input.clear()
        self.chat_input.setEnabled(False)
        self.send_btn.setEnabled(False)

        self._add_chat_message("User", text)
        self.status_label.setText("思考中...")

        # 后台线程调用 LLM，不阻塞 UI
        threading.Thread(target=self._call_llm, args=(text,), daemon=True).start()

    def _call_llm(self, text):
        """在后台线程中运行，通过信号发送结果到主线程"""
        try:
            raw = self.llm.dialogue(text)
            print(f"[LLM] 原始响应: {raw}")
            data = json.loads(raw)
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.llm_error.emit(f"LLM 调用失败: {e}")
            return

        answer = data.get("answer", "")
        describe = data.get("describe", "")
        try:
            motions = [
                (float(data.get("motion_x1", 0)), float(data.get("motion_y1", 0))),
                (float(data.get("motion_x2", 0)), float(data.get("motion_y2", 0))),
                (float(data.get("motion_x3", 0)), float(data.get("motion_y3", 0))),
                (float(data.get("motion_x4", 0)), float(data.get("motion_y4", 0))),
            ]
        except (ValueError, TypeError) as e:
            self.llm_error.emit(f"解析 motion 参数失败: {e}")
            return

        print(f"[LLM] motions={motions}, answer={answer}")
        # 通过信号发送到主线程
        self.llm_response.emit(
            {
                "motions": motions,
                "answer": answer,
                "describe": describe,
            }
        )

    def _on_llm_response(self, data):
        """主线程：处理 LLM 响应，同时显示聊天消息并启动动画"""
        print(f"[Main] 收到 LLM 响应，_current_t0={self._current_t0}")
        self._add_chat_message("Nuero-sama", data["answer"], data["describe"])

        has_motion = any(abs(x) > 0.1 or abs(y) > 0.1 for x, y in data["motions"])

        if not self._current_t0 or not has_motion:
            if not self._current_t0:
                self._add_system_message("当前图像没有 track 数据，请先运行 CoTracker")
            self.chat_input.setEnabled(True)
            self.send_btn.setEnabled(True)
            self._llm_busy = False
            return

        self.anim.start(data["motions"], data["describe"])

    def _on_llm_error(self, msg):
        print(f"[Main] LLM 错误: {msg}")
        self.chat_input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.status_label.setText("Error")
        self._add_system_message(f"错误: {msg}")
        self._llm_busy = False

    def _on_anim_complete(self):
        self.chat_input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.status_label.setText("就绪")
        self._llm_busy = False

    def _on_mouse_sample(self, scene_x, scene_y):
        if self.anim.is_active:
            return
        center = self._get_img_center()
        if not center:
            return
        dx = scene_x - center[0]
        dy = -(scene_y - center[1])
        self.anim.navigate_offset(dx, dy)

    def _on_gesture_complete(self, trajectory):
        self.anim.set_idle_enabled(True)
        if not trajectory or self._llm_busy:
            return
        self._llm_busy = True
        traj_text = f"用户用鼠标在你身上画了轨迹: {trajectory}, 请用可爱的语气回应并解读这个动作"
        threading.Thread(target=self._call_llm, args=(traj_text,), daemon=True).start()

    def _get_img_center(self):
        pixmap = self.canvas.pixmap_item.pixmap()
        if pixmap and not pixmap.isNull():
            return (pixmap.width() / 2.0, pixmap.height() / 2.0)
        return None

    def _is_queue_empty(self):
        with self.queue_lock:
            return not self.image_path_queue

    def _add_paths_to_queue(self, paths):
        added = 0
        with self.queue_lock:
            for n in paths:
                # if len(self.image_path_queue) < QUEUE_LEN:
                self.image_path_queue.append(n)
                added += 1
        return added

    # ----- 图像加载与导航 -----
    def _load_image_from_rel(self, rel_path):
        abs_path = os.path.abspath(rel_path)
        if self.canvas.load_image(abs_path):
            self.current_image_path = Path(rel_path).as_posix()
        else:
            print(f"加载图像失败: {abs_path}")

    def _apply_cached_track(self):
        node_key = Path(self.current_image_path).as_posix().lstrip("/")

        if not models.NODE_POSITIONS.get(node_key):
            try:
                pos = models.get_node_position(node_key)
                if pos:
                    models.NODE_POSITIONS[node_key] = pos
                    print(f"  懒加载 track: {node_key} -> {len(pos)} 个点")
            except Exception:
                pass

        t0_positions = models.NODE_POSITIONS.get(node_key, [])
        if t0_positions:
            self._current_t0 = [(float(p[0]), float(p[1])) for p in t0_positions]
        else:
            self._current_t0 = []

    def _fetch_current_image(self):
        with self.queue_lock:
            if not self.image_path_queue:
                return
            new_path = self.image_path_queue.popleft()

        if new_path == self.current_image_path:
            return

        self._load_image_from_rel(new_path)
        self._apply_cached_track()
        print(f"[切图] 切换到 {new_path}, _current_t0 点数={len(self._current_t0)}")
        self.status_label.setText(
            f"Image: {Path(new_path).parent.name}/{Path(new_path).name}"
            f" | Points: {len(self._current_t0)}"
        )

    # ----- 关闭 -----
    def closeEvent(self, event):
        self.image_timer.stop()
        self.motion_check_timer.stop()
        event.accept()


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main():
    print("Nuero-sama Virtual Anchor (LLM Driven)")
    print(f"基础目录: {BASE_PATH} -> {os.path.abspath(BASE_PATH)}")
    print(f"图片目录: {IMAGE_BASE_PHYSICAL_DIR}")

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

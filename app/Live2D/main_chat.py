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
import time
import threading
from pathlib import Path
from collections import deque

import numpy as np
import networkx as nx
import torch
from PIL import Image

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor
from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QGroupBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMessageBox,
)

import models
import llm_service

# ----------------------------------------------------------------------
# 路径配置
# ----------------------------------------------------------------------
BASE_PATH = "nuero"
models.BASE_DIR = BASE_PATH
models.GRAPH_PATH = os.path.join(BASE_PATH, "graph.txt")
models.IMAGES_DIR = os.path.join(BASE_PATH, "images")
models.TRACKS_DIR = os.path.join(BASE_PATH, "track")
models.load_graph()

IMAGE_BASE_REL_DIR = os.path.join(BASE_PATH, "images", "video_00014")
IMAGE_BASE_PHYSICAL_DIR = os.path.abspath(IMAGE_BASE_REL_DIR)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

FPS = 60
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
def calculate_loss(points, node_positions):
    if not points or not node_positions or len(points) != len(node_positions):
        return float("inf")
    p_np = np.array(points, dtype=np.float32)
    n_np = np.array(node_positions, dtype=np.float32)
    return float(np.mean((p_np - n_np) ** 2))


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
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)


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

        # 动画状态机
        self._is_animating = False
        self._waiting_for_queue = False
        self._motion_sequence = []    # [(dx, dy), ...]
        self._motion_step = 0
        self._origin_node = None
        self._returning_to_center = False
        self._llm_answer = ""
        self._llm_describe = ""
        self._motion_target_node = None
        self._initial_img_center = None

        self._build_ui()

        # 加载初始图像
        self._load_image_from_rel(self.current_image_path)
        self._apply_cached_track()

        # 图像切换定时器（持续运行）
        self.image_timer = QTimer(self)
        self.image_timer.timeout.connect(self._fetch_current_image)
        self.image_timer.start(1000 // FPS)

        # 动画进度检查定时器
        self.motion_check_timer = QTimer(self)
        self.motion_check_timer.timeout.connect(self._check_motion_progress)
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
            "font-family:'Segoe UI',sans-serif;font-size:14px;")
        self.chat_history.setMinimumHeight(200)
        chat_layout.addWidget(self.chat_history, 1)

        input_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("输入消息，按 Enter 发送...")
        self.chat_input.returnPressed.connect(self._send_chat)
        self.send_btn = QPushButton("发送")
        self.send_btn.setStyleSheet(
            "background:#6f42c1;color:white;padding:6px 16px;border-radius:4px;font-weight:bold;")
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
            f'{desc_html}'
            f'<span style="background:{color};color:white;padding:4px 12px;'
            f'border-radius:12px;display:inline-block;max-width:80%;">'
            f'<b>{sender}</b>: {text}</span></div>'
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
        if self._is_animating:
            self._add_system_message("正在播放动画，请稍候...")
            return

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
        self.llm_response.emit({
            "motions": motions,
            "answer": answer,
            "describe": describe,
        })

    def _on_llm_response(self, data):
        """主线程：处理 LLM 响应，启动动画"""
        print(f"[Main] 收到 LLM 响应，_current_t0={self._current_t0}")
        self._start_animation(
            data["motions"],
            data["answer"], data["describe"],
        )

    def _on_llm_error(self, msg):
        print(f"[Main] LLM 错误: {msg}")
        self.chat_input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.status_label.setText("Error")
        self._add_system_message(f"错误: {msg}")

    # ----- 动画状态机 -----
    def _start_animation(self, motions, answer, describe):
        print(f"[动画] _start_animation 被调用, _current_t0={self._current_t0}")
        if not self._current_t0:
            print("[动画] _current_t0 为空，跳过动画")
            self._add_system_message("当前图像没有 track 数据，请先运行 CoTracker")
            self.chat_input.setEnabled(True)
            self.send_btn.setEnabled(True)
            return

        self._is_animating = True
        self._motion_sequence = motions
        self._motion_step = 0
        self._origin_node = self.current_image_path
        self._returning_to_center = False
        self._waiting_for_queue = False
        self._llm_answer = answer
        self._llm_describe = describe
        self._motion_target_node = None

        pixmap = self.canvas.pixmap_item.pixmap()
        if pixmap and not pixmap.isNull():
            self._initial_img_center = (pixmap.width() / 2.0, pixmap.height() / 2.0)
        else:
            self._initial_img_center = None

        print(f"[动画] 开始: origin={self._origin_node}, img_center={self._initial_img_center}")
        for i, (dx, dy) in enumerate(motions, 1):
            print(f"[动画]   motion {i}: dx={dx:.2f}, dy={dy:.2f}")
        self.status_label.setText(f"执行动画... {describe}")
        self._process_next_motion()

    def _process_next_motion(self):
        print(f"[动画] _process_next_motion step={self._motion_step}/{len(self._motion_sequence)}")
        if self._motion_step >= len(self._motion_sequence):
            print("[动画] 所有 motion 完成，准备回中")
            self._start_return_to_center()
            return

        cx, cy = self._motion_sequence[self._motion_step]
        cy = -cy  # LLM y positive=up, image y positive=down

        if not self._current_t0:
            print(f"[动画] _current_t0 为空，跳过 motion {self._motion_step}")
            self._motion_step += 1
            self._process_next_motion()
            return

        # 使用动画初始图像中心作为所有 motion 的参照点，避免累积偏移
        if self._initial_img_center:
            img_cx, img_cy = self._initial_img_center
        else:
            pixmap = self.canvas.pixmap_item.pixmap()
            img_cx = pixmap.width() / 2.0 if pixmap else 0
            img_cy = pixmap.height() / 2.0 if pixmap else 0

        # LLM x,y 是相对图像中心的偏移 → 计算目标中心
        target_cx = img_cx + cx
        target_cy = img_cy + cy

        # 计算当前 T0 中心
        face_cx = sum(p[0] for p in self._current_t0) / len(self._current_t0)
        face_cy = sum(p[1] for p in self._current_t0) / len(self._current_t0)

        # 求 delta 偏移并应用到所有 T0 点
        dx = target_cx - face_cx
        dy = target_cy - face_cy
        t1_target = [(x + dx, y + dy) for x, y in self._current_t0]
        step_label = self._motion_step + 1
        print(f"[动画] motion {step_label}: img_center=({img_cx:.0f},{img_cy:.0f}), llm_offset=({cx:.1f},{cy:.1f}), target=({target_cx:.1f},{target_cy:.1f}), face=({face_cx:.1f},{face_cy:.1f}), delta=({dx:.1f},{dy:.1f})")
        self.status_label.setText(
            f"动作 {step_label}/{len(self._motion_sequence)}: target->({target_cx:.0f},{target_cy:.0f})  {self._llm_describe}")

        self._find_and_add_path(t1_target)
        with self.queue_lock:
            qlen = len(self.image_path_queue)
        print(f"[动画] 加入队列后长度={qlen}")
        self._waiting_for_queue = True

    def _find_and_add_path(self, t1_target):
        start_node = self.current_image_path
        print(f"[寻路] start_node={start_node}, t1_target样本={t1_target[:2]}")
        if models.G is None or start_node not in models.G:
            print(f"[寻路] 起点不在图中: models.G={models.G is None}, in_G={start_node in models.G if models.G else 'N/A'}")
            self._add_system_message(f"寻路失败: 节点 {start_node} 不在图中")
            return

        node_loss = {}
        count = 0
        for node in models.G.nodes:
            node_pos = models.NODE_POSITIONS.get(node, [])
            loss = calculate_loss(t1_target, node_pos)
            node_loss[node] = loss
            count += 1

        print(f"[寻路] 全局搜索了 {count} 个节点")

        if not node_loss:
            print("[寻路] 没有可用的节点")
            return

        target_node = min(node_loss, key=node_loss.get)
        self._motion_target_node = target_node
        min_loss = node_loss[target_node]
        best_pos = models.NODE_POSITIONS.get(target_node, [])
        print(f"[寻路] 目标节点={target_node}, min_loss={min_loss:.4f}, best_pos样本={best_pos[:2]}")
        if min_loss > 10000:
            print(f"[寻路] WARNING: min_loss={min_loss:.1f} 很大，可能未找到匹配节点")

        try:
            shortest = nx.shortest_path(models.G, start_node, target_node)
            print(f"[寻路] 最短路径长度={len(shortest)}: {shortest[:3]}...")
        except nx.NetworkXNoPath:
            print("[寻路] 无路径，使用起点自身")
            shortest = [start_node]

        added = 0
        with self.queue_lock:
            for n in shortest[1:]:
                if len(self.image_path_queue) < QUEUE_LEN:
                    self.image_path_queue.append(n)
                    added += 1
        print(f"[寻路] 队列添加了 {added} 个节点 (目标={target_node})")

    def _start_return_to_center(self):
        print(f"[回中] 开始回中: origin={self._origin_node}, current={self.current_image_path}")
        if not self._origin_node:
            print("[回中] 无 origin 节点")
            self._finish_animation()
            return

        current = self.current_image_path
        if current == self._origin_node:
            print("[回中] 已在起点")
            self._finish_animation()
            return

        if models.G is None or current not in models.G or self._origin_node not in models.G:
            print(f"[回中] 节点不在图中: G={models.G is None}, current_in={current in models.G if models.G else 'N/A'}")
            self._finish_animation()
            return

        try:
            path = nx.shortest_path(models.G, current, self._origin_node)
            print(f"[回中] 路径长度={len(path)}")
        except nx.NetworkXNoPath:
            print("[回中] 无路径")
            self._finish_animation()
            return

        added = 0
        with self.queue_lock:
            for n in path[1:]:
                if len(self.image_path_queue) < QUEUE_LEN:
                    self.image_path_queue.append(n)
                    added += 1
        print(f"[回中] 队列添加了 {added} 个节点")

        self._returning_to_center = True
        self._waiting_for_queue = True
        self.status_label.setText("回中...")

    def _finish_animation(self):
        print(f"[动画] _finish_animation: answer='{self._llm_answer}', describe='{self._llm_describe}'")
        self._is_animating = False
        self._waiting_for_queue = False
        self._returning_to_center = False
        self._motion_target_node = None

        self._add_chat_message("Nuero-sama", self._llm_answer, self._llm_describe)
        self.status_label.setText(f"回答完毕 | {self._llm_describe}")

        self.chat_input.setEnabled(True)
        self.send_btn.setEnabled(True)

    def _continue_motion_path(self):
        """队列已空但尚未到达当前 motion 的目标节点，重新计算并填充路径"""
        current = self.current_image_path
        target = self._motion_target_node
        print(f"[续路] current={current}, target={target}")

        if not target or not models.G or current not in models.G or target not in models.G:
            print(f"[续路] 无法继续寻路，跳过当前 motion")
            self._motion_step += 1
            if self._motion_step >= len(self._motion_sequence):
                self._start_return_to_center()
            else:
                self._process_next_motion()
            return

        try:
            path = nx.shortest_path(models.G, current, target)
            print(f"[续路] 补充路径长度={len(path)}")
        except nx.NetworkXNoPath:
            print(f"[续路] 无路径到目标，跳过当前 motion")
            self._motion_step += 1
            if self._motion_step >= len(self._motion_sequence):
                self._start_return_to_center()
            else:
                self._process_next_motion()
            return

        added = 0
        with self.queue_lock:
            for n in path[1:]:
                if len(self.image_path_queue) < QUEUE_LEN:
                    self.image_path_queue.append(n)
                    added += 1
        print(f"[续路] 补充了 {added} 个节点")
        if added == 0:
            self._waiting_for_queue = True

    def _continue_return_path(self):
        """回中时队列已空但尚未到达起点，重新计算并填充路径"""
        current = self.current_image_path
        target = self._origin_node
        print(f"[续回] current={current}, target={target}")

        if not target or not models.G or current not in models.G or target not in models.G:
            print(f"[续回] 无法继续回中，直接结束")
            self._finish_animation()
            return

        try:
            path = nx.shortest_path(models.G, current, target)
            print(f"[续回] 补充路径长度={len(path)}")
        except nx.NetworkXNoPath:
            print(f"[续回] 无路径到起点，直接结束")
            self._finish_animation()
            return

        added = 0
        with self.queue_lock:
            for n in path[1:]:
                if len(self.image_path_queue) < QUEUE_LEN:
                    self.image_path_queue.append(n)
                    added += 1
        print(f"[续回] 补充了 {added} 个节点")
        if added == 0:
            self._waiting_for_queue = True

    def _check_motion_progress(self):
        if not self._is_animating or not self._waiting_for_queue:
            return

        with self.queue_lock:
            if self.image_path_queue:
                return

        # 队列已空
        self._waiting_for_queue = False
        print(f"[检查] 队列已空, step={self._motion_step}, returning={self._returning_to_center}")

        if self._returning_to_center:
            if self.current_image_path == self._origin_node:
                print("[检查] 回中完成，已到达起点")
                self._finish_animation()
            else:
                print(f"[检查] 回中未完成，当前={self.current_image_path}，目标={self._origin_node}")
                self._waiting_for_queue = True
                self._continue_return_path()
            return

        if self._motion_target_node and self.current_image_path != self._motion_target_node:
            print(f"[检查] 未到达目标 {self._motion_target_node}，当前={self.current_image_path}，继续填充路径")
            self._waiting_for_queue = True
            self._continue_motion_path()
            return

        self._motion_step += 1
        print(f"[检查] 立即执行下一步 (step={self._motion_step})")

        if self._motion_step >= len(self._motion_sequence):
            self.status_label.setText("动作完成，回中...")
            self._start_return_to_center()
        else:
            self._process_next_motion()

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

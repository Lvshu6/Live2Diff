import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import random
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from collections import deque
import asyncio
import networkx as nx
import json
import torch
import numpy as np
import models
from PIL import Image
from pathlib import Path
from openai import AsyncOpenAI

BASE_PATH = "nuero"

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

PERSONAS_FILE = os.path.join(BASE_PATH, "personas.json")


def load_personas() -> dict:
    if os.path.exists(PERSONAS_FILE):
        with open(PERSONAS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_personas(data: dict):
    with open(PERSONAS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


personas_store: dict = load_personas()

_DEFAULT_PERSONA = {
    "name": "default",
    "description": "通用 AI 助手",
    "system_prompt": (
        "你是一个聪明、友好、专业的 AI 助手。"
        "回答问题时简洁清晰，必要时提供详细解释。"
        "你乐于助人，态度积极，擅长写作、分析、编程、问答等各类任务。"
    ),
    "begin_dialogs": [],
}
if "default" not in personas_store:
    personas_store["default"] = _DEFAULT_PERSONA
    save_personas(personas_store)

IMAGE_BASE_REL_DIR = os.path.join(BASE_PATH, "images", "video_00014")
IMAGE_BASE_PHYSICAL_DIR = os.path.abspath(IMAGE_BASE_REL_DIR)
current_image_path = os.path.join(IMAGE_BASE_REL_DIR, "000000.png")
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]

listen_task_started = False
listen_task = None

# DeepSeek client (OpenAI-compatible)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
)

app = FastAPI()

PET_STATIC_DIR = os.path.abspath(BASE_PATH)
if os.path.exists(PET_STATIC_DIR):
    app.mount("/" + BASE_PATH, StaticFiles(directory=PET_STATIC_DIR), name=BASE_PATH)
    print(f"✅ 静态文件挂载成功：/{BASE_PATH} -> {PET_STATIC_DIR}")
else:
    print(f"❌ 警告: 未找到 '{BASE_PATH}' 文件夹，请确保路径是 {PET_STATIC_DIR}")

templates = Jinja2Templates(directory="templates")

points_data = {"t0": [], "t1": []}
last_points_hash = None
image_path_queue = deque()
data_lock = asyncio.Lock()


def get_image_url(rel_path: str) -> str:
    return "/" + rel_path.lstrip("/")


def generate_image_paths() -> List[str]:
    if not os.path.exists(IMAGE_BASE_PHYSICAL_DIR):
        print(f"❌ 警告: 图片物理目录 '{IMAGE_BASE_PHYSICAL_DIR}' 不存在")
        return []
    image_rel_paths = []
    for filename in os.listdir(IMAGE_BASE_PHYSICAL_DIR):
        if any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            rel_path = os.path.join(IMAGE_BASE_REL_DIR, filename)
            image_rel_paths.append(rel_path)
    image_rel_paths.sort()
    return image_rel_paths


def calculate_loss(points: List[List[float]], node_positions: List[List[float]]) -> float:
    if not points or not node_positions or len(points) != len(node_positions):
        return float('inf')
    points_np = np.array(points, dtype=np.float32)
    node_np = np.array(node_positions, dtype=np.float32)
    return float(np.mean((points_np - node_np) ** 2))


def add_image_to_queue(path: Optional[str] = None):
    global image_path_queue, current_image_path
    depth_limit = 20
    if path:
        clean_path = path.lstrip("/")
        image_path_queue.append(clean_path)
        print(f"📥 手动添加路径到队列: {clean_path}")
        return

    if image_path_queue:
        start_node = image_path_queue[-1]
    else:
        start_node = current_image_path

    if models.G is None or start_node not in models.G.nodes:
        print(f"⚠️ 起点 {start_node} 不在图中，使用默认路径")
        all_rel_paths = generate_image_paths()
        if all_rel_paths:
            image_path_queue.append(all_rel_paths[0])
        return

    node_loss = {}
    for node in nx.bfs_tree(models.G, start_node, depth_limit=depth_limit):
        node_pos = models.NODE_POSITIONS.get(node, [])
        loss = calculate_loss(points_data["t1"], node_pos)
        node_loss[node] = loss

    if not node_loss:
        all_rel_paths = generate_image_paths()
        if all_rel_paths:
            image_path_queue.append(all_rel_paths[0])
        return

    sorted_nodes = sorted(node_loss.items(), key=lambda x: x[1])
    target_node = sorted_nodes[0][0]
    min_loss = sorted_nodes[0][1]
    print(f"🎯 找到loss最小的节点: {target_node} (loss={min_loss:.4f})")

    try:
        shortest_path = nx.shortest_path(models.G, start_node, target_node)
    except nx.NetworkXNoPath:
        shortest_path = [start_node]

    MAX_QUEUE_ADD = 10
    for node_path in shortest_path[:MAX_QUEUE_ADD]:
        image_path_queue.append(node_path)
    print(f"📥 路径添加完成，队列状态: {list(image_path_queue)}")


async def listen_points_update():
    global last_points_hash, points_data
    while True:
        async with data_lock:
            current_hash = hash(str(points_data))
            if current_hash != last_points_hash and len(image_path_queue) <= 5:
                last_points_hash = current_hash
                add_image_to_queue()
                print(f"🔄 检测到点数据更新，队列长度: {len(image_path_queue)}")
        await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup_event():
    global listen_task
    if not listen_task_started:
        listen_task = asyncio.create_task(listen_points_update())
        print("✅ 服务启动完成，点数据监听任务已创建")


@app.on_event("shutdown")
async def shutdown_event():
    global listen_task
    if listen_task and not listen_task.done():
        listen_task.cancel()
        await listen_task
    print("✅ 服务已优雅关闭")


# ------------------------------------------------------------------
# 拖拽最近邻帧搜索（直接响应，绕开队列）
# ------------------------------------------------------------------

def find_best_node(t1_points: List) -> str:
    """在预加载的 NODE_POSITIONS 中全局搜索 MSE 最小的帧，纯内存计算"""
    if not t1_points or not models.NODE_POSITIONS:
        return current_image_path
    t1_np = np.array(t1_points, dtype=np.float32)
    n = len(t1_points)
    min_loss = float('inf')
    best = current_image_path
    for node, positions in models.NODE_POSITIONS.items():
        if not positions or len(positions) != n:
            continue
        loss = float(np.mean((t1_np - np.array(positions, dtype=np.float32)) ** 2))
        if loss < min_loss:
            min_loss = loss
            best = node
    return best


@app.post("/api/best-frame")
async def get_best_frame(data: Dict[str, Any]):
    """
    拖动专用接口：接收当前 T1 坐标，直接返回最匹配的帧路径。
    不写队列，不做 BFS，延迟 ≈ 网络 RTT + 内存搜索时间。
    """
    t1 = data.get("t1", [])
    if not t1:
        return JSONResponse(content={"image_src": "/" + current_image_path, "t0_points": []})
    loop = asyncio.get_event_loop()
    best_node = await loop.run_in_executor(None, find_best_node, t1)
    t0_positions = models.NODE_POSITIONS.get(best_node, [])
    return JSONResponse(content={
        "image_src": "/" + best_node,
        "t0_points": t0_positions
    })


# ------------------------------------------------------------------
# 原有路由
# ------------------------------------------------------------------

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index_chat.html", {
        "request": request,
        "image_src": "/" + current_image_path
    })


@app.get("/api/points")
async def get_points():
    async with data_lock:
        return JSONResponse(content={
            "t0": list(points_data["t0"]),
            "t1": list(points_data["t1"])
        })


@app.post("/api/points")
async def update_points(data: Dict[str, Any]):
    global points_data, last_points_hash
    t0 = data.get("t0", [])
    t1 = data.get("t1", [])
    if len(t0) != len(t1):
        return JSONResponse(status_code=400, content={"error": "T0 and T1 length mismatch"})
    async with data_lock:
        points_data["t0"] = t0
        points_data["t1"] = t1
    await asyncio.sleep(0)
    return JSONResponse(content={
        "status": "success",
        "count": len(t0),
        "current_image": "/" + current_image_path
    })


@app.get("/api/current-image")
async def get_current_image():
    global current_image_path
    async with data_lock:
        if image_path_queue:
            current_image_path = image_path_queue.popleft()
            print(f"📤 更新图片路径: {current_image_path}")
    await asyncio.sleep(0)
    current_node_key = current_image_path.lstrip('/')
    t0_positions = models.NODE_POSITIONS.get(current_node_key, [])
    return JSONResponse(content={
        "image_src": "/" + current_image_path,
        "t0_points": t0_positions
    })


@app.post("/api/cotracker-track")
async def receive_cotracker_track(data: Dict[str, Any]):
    t0_points = data.get("t0_points", [])
    if not isinstance(t0_points, list):
        return JSONResponse(status_code=400, content={"status": "error", "message": "t0_points 必须是列表格式"})
    valid = all(isinstance(p, (list, tuple)) and len(p) == 2 for p in t0_points)
    if not valid:
        return JSONResponse(status_code=400, content={"status": "error", "message": "t0_points 必须是 (N,2) 格式"})
    if len(t0_points) == 0:
        return JSONResponse(content={"status": "success", "message": "未提供跟踪点"})

    if models.G is None:
        models.load_graph()

    queries = torch.tensor([[[0, float(x), float(y)] for x, y in t0_points]], dtype=torch.float32, device=models.DEVICE)
    start_node = current_image_path.lstrip('/')
    paths = models.bfs_collect_paths(models.G, start_node=start_node, max_depth=20)

    results_summary = []
    saved_files_count = 0

    for i, path in enumerate(paths):
        video_pil_list = []
        valid_path = True
        for node_path in path:
            if not os.path.exists(node_path):
                valid_path = False
                break
            try:
                img = Image.open(node_path).convert("RGB")
                video_pil_list.append(img)
            except Exception as e:
                print(f"⚠️ 加载图片失败 {node_path}: {e}")
                valid_path = False
                break

        if not valid_path or len(video_pil_list) < 2:
            continue

        try:
            pred_tracks = models.run_cotracker(video_pil_list, queries)
            tracks_np = pred_tracks[0].cpu().numpy()
            T_frames, N_points, _ = tracks_np.shape

            for frame_idx, node_path in enumerate(path):
                if frame_idx >= T_frames:
                    break
                track_path = node_path.replace("images/", "track/", 1)
                track_dir = os.path.dirname(track_path)
                track_basename = os.path.splitext(os.path.basename(track_path))[0]
                full_track_path = os.path.join(track_dir, f"{track_basename}.json")
                Path(track_dir).mkdir(parents=True, exist_ok=True)
                frame_tracks = tracks_np[frame_idx].tolist()
                with open(full_track_path, 'w', encoding='utf-8') as f:
                    json.dump({"track": frame_tracks}, f, ensure_ascii=False, indent=2)
                if os.path.exists(full_track_path):
                    saved_files_count += 1
                    print(f"✅ 保存成功: {full_track_path}")
                else:
                    print(f"❌ 保存失败: {full_track_path}")

            results_summary.append({
                "path_index": i,
                "frames": T_frames,
                "points": N_points,
                "saved_files": min(T_frames, len(path))
            })
            del pred_tracks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ 处理路径 {i} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    for node in models.G.nodes:
        models.NODE_POSITIONS[node] = models.get_node_position(node)

    return JSONResponse(content={
        "status": "success",
        "input_points_count": len(t0_points),
        "paths_processed": len(results_summary),
        "saved_files_total": saved_files_count,
        "details": results_summary,
        "current_image_url": "/" + current_image_path,
        "message": f"成功处理 {len(results_summary)} 条路径，保存 {saved_files_count} 个轨迹文件"
    })


# ------------------------------------------------------------------
# 人格管理路由
# ------------------------------------------------------------------

@app.get("/api/personas")
async def get_personas():
    return JSONResponse(content=personas_store)


@app.post("/api/personas")
async def upsert_persona(data: Dict[str, Any]):
    name = data.get("name", "").strip()
    if not name:
        return JSONResponse(status_code=400, content={"error": "name 不能为空"})
    personas_store[name] = {
        "name": name,
        "description": data.get("description", ""),
        "system_prompt": data.get("system_prompt", ""),
        "begin_dialogs": data.get("begin_dialogs", []),
    }
    save_personas(personas_store)
    return JSONResponse(content={"status": "success", "name": name})


@app.delete("/api/personas/{name}")
async def delete_persona(name: str):
    personas_store.pop(name, None)
    save_personas(personas_store)
    return JSONResponse(content={"status": "success"})


# ------------------------------------------------------------------
# AI 对话路由
# ------------------------------------------------------------------

@app.get("/api/chat/status")
async def chat_status():
    """检查 DeepSeek API 是否已配置"""
    configured = bool(DEEPSEEK_API_KEY)
    return JSONResponse(content={"configured": configured})


@app.post("/api/chat")
async def chat(data: Dict[str, Any]):
    """
    流式对话接口（SSE）。
    请求体: { "messages": [{"role": "user"/"assistant", "content": "..."}] }
    返回: text/event-stream，每条事件形如 data: {"delta": "..."}
    流结束时发送 data: [DONE]
    """
    messages: List[Dict[str, str]] = data.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": "messages 不能为空"})

    if not DEEPSEEK_API_KEY:
        return JSONResponse(
            status_code=503,
            content={"error": "未配置 DEEPSEEK_API_KEY，请设置环境变量后重启服务"}
        )

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            stream = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    payload = json.dumps({"delta": delta}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ------------------------------------------------------------------
# 启动服务
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print(f"🔍 路径验证:")
    print(f"  - 基础目录: {BASE_PATH} -> {'存在' if os.path.exists(os.path.abspath(BASE_PATH)) else '不存在'}")
    print(f"  - 图片目录: {IMAGE_BASE_PHYSICAL_DIR} -> {'存在' if os.path.exists(IMAGE_BASE_PHYSICAL_DIR) else '不存在'}")
    print(f"  - DeepSeek API: {'已配置' if DEEPSEEK_API_KEY else '未配置 (请设置 DEEPSEEK_API_KEY)'}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
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

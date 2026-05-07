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

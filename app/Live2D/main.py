import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import random
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, AsyncGenerator,Tuple
from collections import deque
import asyncio
import networkx as nx
import json
import torch
import numpy as np
import models
from PIL import Image  
from pathlib import Path
BASE_PATH = "pet"# 核心基础路径（无前置/）

# BASE_DIR="pet"
models.BASE_DIR = BASE_PATH
models.GRAPH_PATH = os.path.join(BASE_PATH, "graph.txt")
models.IMAGES_DIR = os.path.join(BASE_PATH, "images")
models.TRACKS_DIR = os.path.join(BASE_PATH, "track")
models.load_graph()
  
IMAGE_BASE_REL_DIR = os.path.join(BASE_PATH, "images", "video_00008")
IMAGE_BASE_PHYSICAL_DIR = os.path.abspath(IMAGE_BASE_REL_DIR)
current_image_path = os.path.join(IMAGE_BASE_REL_DIR, "000000.png")  
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]
# --------------------------
# 2. 全局标志位：确保listen_points_update仅启动一次
# --------------------------
listen_task_started = False  
listen_task = None           

app = FastAPI()

# --------------------------
# 4. 静态文件挂载 & 目录配置
# --------------------------

PET_STATIC_DIR = os.path.abspath(BASE_PATH)
if os.path.exists(PET_STATIC_DIR):
    app.mount("/" + BASE_PATH, StaticFiles(directory=PET_STATIC_DIR), name=BASE_PATH)
    print(f"✅ 静态文件挂载成功：/{BASE_PATH} -> {PET_STATIC_DIR}")
else:
    print(f"❌ 警告: 未找到 '{BASE_PATH}' 文件夹，请确保路径是 {PET_STATIC_DIR}")

templates = Jinja2Templates(directory="templates")

# --------------------------
# 5. 路径配置（仅保留无前置/的格式）
# --------------------------

# GRAPH_PATH = os.path.join(BASE_PATH, "graph.txt")
# IMAGES_DIR = os.path.join(BASE_PATH, "images")
# TRACKS_DIR = os.path.join(BASE_PATH, "track")

# --------------------------
# 6. 全局状态管理
# --------------------------
points_data = {
    "t0": [],
    "t1": []
}
last_points_hash = None
image_path_queue = deque()  
data_lock = asyncio.Lock()

# --------------------------
# 7. 工具函数（核心修改：add_image_to_queue）
# --------------------------
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
    """
    计算points_data与节点坐标的损失（均方误差）
    :param points: 前端传入的points_data["t0"] (N,2)
    :param node_positions: 节点的NODE_POSITIONS (N,2)
    :return: 均方误差损失值（越小越匹配）
    """
    if not points or not node_positions or len(points) != len(node_positions):
        return float('inf')  # 长度不匹配则损失无穷大
    
    points_np = np.array(points, dtype=np.float32)
    node_np = np.array(node_positions, dtype=np.float32)
    mse = np.mean((points_np - node_np) ** 2)
    return float(mse)

def add_image_to_queue(path: Optional[str] = None):
    """
    重写逻辑：
    1. 优先取队列最后一个路径，队列空则取current_image_path作为起点
    2. 广度遍历models.G，计算每个节点与points_data的loss
    3. 找到loss最小的节点，获取到该节点的最短路径
    4. 将路径上的所有节点添加到队列
    """
    global image_path_queue, current_image_path
    depth_limit = 20  # 广度遍历深度限制，避免过深导致性能问题
    # 1. 处理手动传入路径的情况（保留原有逻辑）
    if path:
        clean_path = path.lstrip("/")
        image_path_queue.append(clean_path)
        print(f"📥 手动添加路径到队列: {clean_path}")
        return
    
    # 2. 确定遍历起点：队列最后一个 > current_image_path
    if image_path_queue:
        start_node = image_path_queue[-1]  # 队列最后一个路径作为起点
    else:
        start_node = current_image_path    # 队列为空则用当前图片路径
    
    # 3. 校验起点是否在图中
    if models.G is None or start_node not in models.G.nodes:
        print(f"⚠️ 起点 {start_node} 不在图中，使用默认路径")
        all_rel_paths = generate_image_paths()
        if all_rel_paths:
            image_path_queue.append(all_rel_paths[0])
        return
    
    # 4. 广度遍历所有可达节点，计算每个节点的loss
    node_loss = {}
    # 遍历所有从起点可达的节点
    for node in nx.bfs_tree(models.G, start_node,depth_limit=depth_limit):
        # 获取节点的坐标
        node_pos = models.NODE_POSITIONS.get(node, [])
        # 计算当前节点与points_data["t1"]的损失
        loss = calculate_loss(points_data["t1"], node_pos)
        node_loss[node] = loss
    
    # 5. 找到loss最小的节点（目标节点）
    if not node_loss:
        print("⚠️ 未找到可达节点，使用默认路径")
        all_rel_paths = generate_image_paths()
        if all_rel_paths:
            image_path_queue.append(all_rel_paths[0])
        return
    
    # 按loss升序排序，取最小loss的节点
    sorted_nodes = sorted(node_loss.items(), key=lambda x: x[1])
    target_node = sorted_nodes[0][0]
    min_loss = sorted_nodes[0][1]
    print(f"🎯 找到loss最小的节点: {target_node} (loss={min_loss:.4f})")
    
    # 6. 获取从起点到目标节点的最短路径
    try:
        shortest_path = nx.shortest_path(models.G, start_node, target_node)
        print(f"📏 最短路径: {shortest_path} (长度={len(shortest_path)})")
    except nx.NetworkXNoPath:
        print(f"⚠️ 无路径到达目标节点 {target_node}，使用默认路径")
        shortest_path = [start_node]
    
    # for node_path in shortest_path:
    #         image_path_queue.append(node_path)
    # --- 截断逻辑：只取前 10 个节点 ---
    MAX_QUEUE_ADD = 10
    path_to_add = shortest_path[:MAX_QUEUE_ADD] # 切片操作
    # 7. 将截断后的路径添加到队列
    for node_path in path_to_add:
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

# --------------------------
# 8. 启动监听任务（仅启动一次）
# --------------------------
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

# --------------------------
# 9. API 路由（所有URL显式拼接 "/" + 变量）
# --------------------------
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
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
    
    # 新增：获取当前图片对应的T0坐标
    current_node_key = current_image_path.lstrip('/')
    t0_positions = models.NODE_POSITIONS.get(current_node_key, [])
    
    return JSONResponse(content={
        "image_src": "/" + current_image_path,
        "t0_points": t0_positions  # 返回T0坐标 (N,2)
    })

@app.post("/api/cotracker-track")
async def receive_cotracker_track(data: Dict[str, Any]):
    t0_points = data.get("t0_points", [])
    
    if not isinstance(t0_points, list):
        return JSONResponse(status_code=400, content={"status": "error", "message": "t0_points 必须是列表格式"})
    
    valid = True
    for idx, point in enumerate(t0_points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            valid = False
            break
    
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
            full_physical_path = node_path
            if not os.path.exists(full_physical_path):
                print(f"⚠️ 文件不存在: {full_physical_path}")
                valid_path = False
                break
            
            try:
                img = Image.open(full_physical_path).convert("RGB")
                video_pil_list.append(img)
            except Exception as e:
                print(f"⚠️ 加载图片失败 {full_physical_path}: {str(e)}")
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
                    print(f"⚠️ 帧索引 {frame_idx} 超出推理结果帧数 {T_frames}，跳过")
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
                    file_size = os.path.getsize(full_track_path)
                    saved_files_count += 1
                    print(f"✅ 保存成功: {full_track_path} | 文件大小: {file_size} 字节")
                else:
                    print(f"❌ 保存失败: {full_track_path} (文件不存在)")

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
            print(f"❌ 处理路径 {i} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    for node in models.G.nodes:
        models.NODE_POSITIONS[node] = models.get_node_position(node)
        print(f"预加载节点坐标: {node} -> {models.NODE_POSITIONS[node]}")
    
    return JSONResponse(content={
        "status": "success",
        "input_points_count": len(t0_points),
        "paths_processed": len(results_summary),
        "saved_files_total": saved_files_count,
        "details": results_summary,
        "current_image_url": "/" + current_image_path,
        "message": f"成功处理 {len(results_summary)} 条路径，保存 {saved_files_count} 个轨迹文件"
    })

# --------------------------
# 10. 启动服务
# --------------------------
if __name__ == "__main__":
    import uvicorn
    
    print(f"🔍 路径验证:")
    print(f"  - 基础目录: {BASE_PATH} -> 物理路径={os.path.abspath(BASE_PATH)} -> {'存在' if os.path.exists(os.path.abspath(BASE_PATH)) else '不存在'}")
    print(f"  - 图片目录: {IMAGE_BASE_REL_DIR} -> 物理路径={IMAGE_BASE_PHYSICAL_DIR} -> {'存在' if os.path.exists(IMAGE_BASE_PHYSICAL_DIR) else '不存在'}")
    print(f"  - 初始图片路径: {current_image_path} -> URL=/{current_image_path}")
    
    test_paths = generate_image_paths()
    if test_paths:
        print(f"📝 找到 {len(test_paths)} 张图片，示例: 相对路径={test_paths[0]}, URL=/{test_paths[0]}")
    else:
        print("❌ 未找到任何图片！")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        log_level="info"
    )
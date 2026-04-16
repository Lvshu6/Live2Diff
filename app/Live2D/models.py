import os
import random
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, AsyncGenerator,Tuple
from collections import deque
import asyncio
from contextlib import asynccontextmanager
import networkx as nx
import json
import torch
import numpy as np
from cotracker.utils.visualizer import read_video_from_path
# --------------------------
# 1. 全局变量 & 状态跟踪
# --------------------------
# 先定义生命周期（避免app重新赋值覆盖挂载）

CotrackerModel=None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在加载 CoTracker3 模型到 {DEVICE} ...")
CotrackerModel = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
CotrackerModel = CotrackerModel.to(DEVICE)
CotrackerModel.eval()
print("模型加载完成。")

def run_cotracker(video, queries):
    #video=[PIL.Image, PIL.Image, ...]
    video = np.stack([np.array(img)for img in video], axis=0)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(DEVICE)
    with torch.no_grad():
        pred_tracks, pred_visibility = CotrackerModel(
            video,             #                             
            queries=queries,  #(B,N,3)(t,x,y),t=0
            backward_tracking=False
        )
    return pred_tracks  #(B,T,N,2)


        
BASE_DIR="pet"
GRAPH_PATH = os.path.join(BASE_DIR, "graph.txt")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
TRACKS_DIR = os.path.join(BASE_DIR, "track")
G = None  # 图对象
NODE_POSITIONS = {}  # 预加载节点坐标

def load_graph():
    """加载无向图数据"""
    global G
    if G is not None:
        return G
    
    G = nx.Graph()
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"graph.txt 不存在: {GRAPH_PATH}")
    
    with open(GRAPH_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 分割两个节点路径
        nodes = line.split()
        if len(nodes) != 2:
            continue
        node1, node2 = nodes
        # 添加节点和边
        G.add_node(node1)
        G.add_node(node2)
        G.add_edge(node1, node2)
    
    # 预加载所有节点的轨迹坐标
    for node in G.nodes:
        NODE_POSITIONS[node] = get_node_position(node)
    
    print(f"图加载完成：节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
    return G


def get_node_position(node_path):
    if "images/" not in node_path:
        return []
        
    json_path = node_path.replace("images/", "track/")
    
    # 处理可能的后缀名变化 (.png/.jpg -> .json)
    # 找到最后一个点的位置，截取前缀，加上 .json
    base_name, _ = os.path.splitext(json_path)
    json_path = f"{base_name}.json"
    
    if not os.path.exists(json_path):
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # {"track": [...]} 
        return data.get("track", data) if isinstance(data, dict) else data
    except Exception:
        return []
    


def bfs_collect_paths(graph: nx.Graph, start_node: str, max_depth: int = 20) -> List[List[str]]:
    """
    广度遍历图，仅收集“完整”路径（死胡同或达到最大深度的路径）。
    
    逻辑调整：
    1. 正常 BFS 遍历。
    2. 只有满足以下任一条件时，才将路径加入 all_paths：
       - 条件 A: 当前节点没有未访问的邻居 (死胡同/叶子节点)。
       - 条件 B: 当前路径长度已达到 max_depth (到达最深)。
    
    :return: 符合条件的路径列表 [[start, ..., end], ...]
    """
    if start_node not in graph:
        return []
    
    visited = set()
    visited.add(start_node)
    
    # 队列: (当前节点, 当前路径列表)
    queue = deque([(start_node, [start_node])])
    
    all_paths = []
    
    while queue:
        current_node, current_path = queue.popleft()
        
        # 获取所有邻居
        neighbors = list(graph.neighbors(current_node))
        # 过滤出未访问的邻居
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        
        current_len = len(current_path)
        
        # 判断是否应该记录当前路径
        is_leaf = (len(unvisited_neighbors) == 0)
        is_max_depth = (current_len >= max_depth)
        
        if is_leaf or is_max_depth:
            all_paths.append(current_path)
            
        
        # 如果还没到最大深度，且有未访问邻居，继续扩展
        # 注意：如果是 is_max_depth 为真，即使有邻居也不再扩展（因为已经到头了）
        if not is_max_depth and unvisited_neighbors:
            for neighbor in unvisited_neighbors:
                # 标记并入队
                visited.add(neighbor)
                new_path = current_path + [neighbor]
                queue.append((neighbor, new_path))
    
    return all_paths

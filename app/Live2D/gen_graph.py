import os
import re
import cv2
import networkx as nx
import numpy as np
import torch
import itertools
import multiprocessing
from typing import List, Tuple, Set, Dict
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import hashlib
# 导入 model.py 中的模型和数据集处理逻辑
from train import AdjacencyCNN, TrackGridDataset  # 假设 model.py 在当前目录或 PYTHONPATH 中

# 导入 cotracker
try:
    from cotracker.utils.visualizer import read_video_from_path
    # 注意：cotracker 通常需要视频输入，这里我们需要适配单帧图像对
    # 如果 cotracker 不支持直接输入两张图，可能需要构造一个临时的 2 帧视频 tensor
except ImportError:
    print("Warning: cotracker not found. Please install it or adjust the import path.")
    raise

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 🔥 【全局唯一 gridsize】所有地方统一用这个
# ==========================================
GRID_SIZE = 48  # 只改这里！！！
\
    
    
def analyze_graph_components(graph_file):
    """分析无向图的连通分量及每个分量包含的文件夹"""
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"图文件 {graph_file} 不存在")

    # 构建无向图
    G = nx.Graph()

    # 读取边列表并添加到图中
    with open(graph_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                u, v = line.split()
                G.add_edge(u, v)
            except ValueError:
                print(f"跳过无效行: {line}")

    # 提取每个节点对应的文件夹
    node_to_folder = {}
    for node in G.nodes():
        path = Path(node)
        folder = path.parent.as_posix()
        node_to_folder[node] = folder

    # 查找所有连通分量
    components = list(nx.connected_components(G))

    # 收集每个连通分量中的文件夹（去重）
    folder_components = []
    for component in components:
        folders = set()
        for node in component:
            folders.add(node_to_folder[node])
        folder_components.append(folders)

    return len(folder_components), folder_components

    
# ================= 辅助函数：图像加载与预处理 =================
def load_image_as_tensor(img_path: str, device: str) -> torch.Tensor:
    """加载单张图像并转换为 cotracker 所需的格式 (1, 3, H, W)"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        return img_tensor
    except Exception as e:
        print(f"加载图像失败 {img_path}: {e}")
        return None

# ================= 辅助函数：轨迹转 Grid (复用 model.py 逻辑) =================
def rasterize_to_grid(points, displacements, grid_size=GRID_SIZE):
    """
    将点和位移转换为 grid 表示 (复用 model.py 中的逻辑)
    points: (N, 2) numpy 或 tensor
    displacements: (N, 2) numpy 或 tensor
    """
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()
    if isinstance(displacements, np.ndarray):
        displacements = torch.from_numpy(displacements).float()

    # 计算缩放比例
    max_x = max(points[:, 0].max().item(), 1.0)
    max_y = max(points[:, 1].max().item(), 1.0)
    scale_x = grid_size / max_x
    scale_y = grid_size / max_y

    grid = torch.zeros((2, grid_size, grid_size), dtype=torch.float32)
    xs = (points[:, 0] * scale_x).long().clamp(0, grid_size - 1)
    ys = (points[:, 1] * scale_y).long().clamp(0, grid_size - 1)
    dx = displacements[:, 0] * scale_x
    dy = displacements[:, 1] * scale_y

    flat_grid = grid.view(2, -1)
    indices = ys * grid_size + xs
    # 使用 scatter_ 累加位移（如果有多个点映射到同一格）
    flat_grid[0, :].scatter_(0, indices, dx)
    flat_grid[1, :].scatter_(0, indices, dy)
    return grid

# ================= 核心逻辑：Cotracker 预测两帧轨迹 =================
def predict_tracks_cotracker(img1_path, img2_path, model_cotracker, device, grid_size=GRID_SIZE):
    """
    使用 cotracker 预测两张图像之间的轨迹
    返回: points_t1 (N, 2), displacements (N, 2)
    """
    # 1. 加载两张图像
    img1 = load_image_as_tensor(img1_path, device)
    img2 = load_image_as_tensor(img2_path, device)
    if img1 is None or img2 is None:
        return None, None

    # 2. 拼接成 2 帧视频 (1, 2, 3, H, W)
    video = torch.cat([img1, img2], dim=0).unsqueeze(0).to(device)  # (B=1, T=2, C=3, H, W)

    # 3. 运行 cotracker
    with torch.no_grad():
        # grid_query_frame=0 表示从第一帧开始采样网格点
        pred_tracks, pred_visibility = model_cotracker(
            video,
            grid_size=grid_size,
            grid_query_frame=0,
            backward_tracking=False
        )
    
    # pred_tracks 形状: (1, T=2, N, 2)
    if pred_tracks.shape[1] < 2:
        return None, None
    
    points_t1 = pred_tracks[0, 0, :, :].cpu().numpy()  # (N, 2)
    points_t2 = pred_tracks[0, 1, :, :].cpu().numpy()  # (N, 2)
    displacements = points_t2 - points_t1
    
    return points_t1, displacements

# ================= 模型加载 =================
def load_adjacency_model(model_path, device, grid_size=GRID_SIZE):
    """加载 model.py 中的 AdjacencyCNN 模型"""
    checkpoint = torch.load(model_path, map_location=device)
    # 兼容不同保存格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        loaded_grid_size = checkpoint.get('grid_size', grid_size)
    else:
        state_dict = checkpoint
        loaded_grid_size = grid_size
    
    model = AdjacencyCNN(grid_size=loaded_grid_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded adjacency model from {model_path} (grid_size={loaded_grid_size})")
    return model, loaded_grid_size

def load_cotracker_model(device):
    """加载 cotracker 模型"""
    print("Loading CoTracker model...")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    model.eval()
    print("CoTracker model loaded.")
    return model

# ================= 同文件夹处理 (保持原有逻辑) =================
def natural_key(filename):
    return int(re.findall(r'\d+', filename)[0])

def process_same_folder_edges(frame_dict, folders_to_process, same_folder_threshold=1):
    edges = set()
    for folder in tqdm(folders_to_process, desc="处理同文件夹边"):
        paths = frame_dict.get(folder, [])
        if not paths:
            continue
        folder_path = Path(folder)
        filenames = [os.path.basename(path) for path in paths]
        sorted_filenames = sorted(filenames, key=natural_key)
        paths = [(folder_path / filename).as_posix() for filename in sorted_filenames]

        for i in range(len(paths)):
            start = max(0, i - same_folder_threshold)
            end = min(len(paths), i + same_folder_threshold + 1)
            for j in range(start, end):
                if i != j:
                    edge = tuple(sorted((paths[i], paths[j])))
                    edges.add(edge)
    return edges

# ================= 跨文件夹处理 (使用 Cotracker + Adjacency Model) =================
def process_cross_folder_pairs(args):
    (paths1, paths2, model_adj, grid_size, model_cotracker, device) = args
    
    edges = set()
    batch_grids = []
    batch_pairs = []
    batch_size = 4  # 根据显存调整

    for i, path1 in enumerate(paths1):
        for j, path2 in enumerate(paths2):
            # 1. 使用 cotracker 预测轨迹
            points_t1, displacements = predict_tracks_cotracker(
                path1, path2, model_cotracker, device, grid_size=GRID_SIZE
            )
            if points_t1 is None:
                continue
            
            # 2. 转换为 grid 输入
            grid = rasterize_to_grid(points_t1, displacements, grid_size=GRID_SIZE)
            batch_grids.append(grid)
            batch_pairs.append((path1, path2))

            # 3. 批次推理
            if len(batch_grids) >= batch_size or (i == len(paths1) - 1 and j == len(paths2) - 1):
                batch_tensor = torch.stack(batch_grids).to(device)  # (B, 2, H, W)
                with torch.no_grad():
                    probs = model_adj(batch_tensor)  # (B, 1)
                    preds = (probs > 0.5).cpu().numpy()
                
                for idx, (p1, p2) in enumerate(batch_pairs):
                    if preds[idx]:
                        edges.add(tuple(sorted((p1, p2))))
                
                # 清理
                del batch_tensor, probs, preds
                torch.cuda.empty_cache()
                batch_grids = []
                batch_pairs = []

    return edges

# ================= 主函数：构建图 =================
def build_video_frame_graph(
        folders: List[str],
        model_path: str,
        device: str = "cuda",
        output_file: str = "graph.txt",
        same_folder_threshold: int = 1,
        num_workers: int = 4,
        use_graph: bool = False,
        graph: str = None
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model_adj, grid_size = load_adjacency_model(model_path, device, grid_size=GRID_SIZE)
    model_cotracker = load_cotracker_model(device)

    # 2. 处理旧图逻辑 (复用原有逻辑)
    new_folders = None
    old_edges = set()
    old_folders = set()

    if use_graph and graph is not None:
        loaded_edges, loaded_folders = read_and_print_graph(graph)
        old_edges = set(tuple(sorted(edge)) for edge in loaded_edges)
        old_folders = {Path(f).as_posix() for f in loaded_folders if Path(f).is_dir()}
        input_folders = {Path(f).as_posix() for f in folders if Path(f).is_dir()}
        new_folders = input_folders - old_folders
        folders = list(input_folders)
        print(f"复用旧图：{len(old_folders)} 个旧文件夹，{len(new_folders)} 个新文件夹")

        if not new_folders and same_folder_threshold == 1:
            print("没有新文件夹需要处理")
            with open(output_file, 'w') as f:
                for u, v in old_edges:
                    f.write(f"{u} {v}\n")
            return old_edges

    # 3. 构建帧字典
    frame_dict = {}
    if use_graph and graph is not None:
        for folder in tqdm(old_folders, desc="扫描旧文件夹"):
            folder_path = Path(folder)
            frame_dict[folder] = [
                Path(f).as_posix()
                for f in folder_path.iterdir()
                if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
            ]

    folders_to_scan = new_folders if (use_graph and graph is not None and new_folders) else folders
    for folder in tqdm(folders_to_scan, desc="扫描新文件夹"):
        folder_path = Path(folder)
        if folder_path.is_dir():
            frame_dict[folder] = [
                Path(f).as_posix()
                for f in folder_path.iterdir()
                if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
            ]

    edges = old_edges.copy()

    # 4. 同文件夹边
    folders_to_process = new_folders if (use_graph and graph is not None and new_folders) else frame_dict.keys()
    same_folder_edges = process_same_folder_edges(frame_dict, folders_to_process, same_folder_threshold)
    edges.update(same_folder_edges)

    if use_graph and graph is not None and not new_folders:
        with open(output_file, 'w') as f:
            for u, v in edges:
                f.write(f"{u} {v}\n")
        print(f"生成完成，共 {len(edges)} 条边，保存至 {output_file}")
        return edges

    # 5. 跨文件夹边
    folder_pairs = []
    if use_graph and graph is not None and new_folders:
        new_folder_list = list(new_folders)
        folder_pairs.extend(itertools.combinations(new_folder_list, 2))
        for new_folder in new_folder_list:
            for old_folder in old_folders:
                if new_folder in frame_dict and old_folder in frame_dict:
                    folder_pairs.append((new_folder, old_folder))
    else:
        folder_list = list(frame_dict.keys())
        folder_pairs = list(itertools.combinations(folder_list, 2))

    tasks = []
    for f1, f2 in folder_pairs:
        if not frame_dict.get(f1, []) or not frame_dict.get(f2, []):
            continue
        tasks.append((
            frame_dict[f1], frame_dict[f2],
            model_adj, GRID_SIZE, model_cotracker, device
        ))

    if tasks:
        # 注意：multiprocessing 传递模型需要小心，这里简化为单进程或需使用 spawn
        # 如果多进程报错，可暂时设置 num_workers=1
        if num_workers > 1:
            try:
                multiprocessing.set_start_method("spawn", force=True)
                with multiprocessing.Pool(num_workers) as pool:
                    for results in tqdm(pool.imap_unordered(process_cross_folder_pairs, tasks),
                                        total=len(tasks), desc="跨文件夹预测"):
                        for result in results:
                            edges.add(tuple(sorted(result)))
            except Exception as e:
                print(f"多进程出错，回退到单进程：{e}")
                for task in tqdm(tasks, desc="跨文件夹预测 (单进程)"):
                    results = process_cross_folder_pairs(task)
                    for result in results:
                        edges.add(tuple(sorted(result)))
        else:
            for task in tqdm(tasks, desc="跨文件夹预测 (单进程)"):
                results = process_cross_folder_pairs(task)
                for result in results:
                    edges.add(tuple(sorted(result)))

    unique_edges = set(tuple(sorted(edge)) for edge in edges)
    with open(output_file, 'w') as f:
        for u, v in unique_edges:
            f.write(f"{u} {v}\n")

    print(f"生成完成，共 {len(unique_edges)} 条边，保存至 {output_file}")
    return unique_edges

# ================= 辅助函数：读写图 =================
def read_and_print_graph(graph_file):
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"图文件 {graph_file} 不存在")
    graph = nx.Graph()
    with open(graph_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                u, v = line.split()
                graph.add_edge(u, v)
            except ValueError:
                continue
    folders = set()
    for edge in graph.edges():
        for node in edge:
            path = Path(node)
            folders.add(path.parent.as_posix())
    return graph.edges(), folders

def generate_output_filename(model_path, input_folder):
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    first_folder = os.path.normpath(input_folder).split('/')[0]
    return f"{base_name}-{first_folder}.txt"

# ================= 主程序入口 =================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    base=Path("pet")
    model_path = base / "best_model.pth"  # 替换为你的 model.py 训练出的模型路径
    
    input_folder = base / "images"
    graph = base / "graph.txt"  # 可选：旧图路径
    device = "cuda"
    # output_file = generate_output_filename(model_path, input_folder)
    output_file = base / "graph.txt"
    use_graph = False
    num_workers = 1  # 如果 cotracker 多进程有问题，建议设为 1
    
    folders = [
        input_folder,
    ]
    
    # 递归获取子目录 (如果需要)
    def get_subdirectories(folders):
        subdirs = []
        for folder in folders:
            if os.path.exists(folder) and os.path.isdir(folder):
                for item in os.listdir(folder):
                    item_path = os.path.join(folder, item)
                    if os.path.isdir(item_path):
                        subdirs.append(item_path)
                        subdirs.extend(get_subdirectories([item_path]))
        return subdirs
    
    folders = get_subdirectories(folders)
    print(f"处理文件夹：{folders}")
    print(f"加载模型：{model_path}, 设备：{device}")
    
    build_video_frame_graph(
        folders=folders,
        model_path=model_path,
        device=device,
        output_file=output_file,
        same_folder_threshold=1,
        num_workers=num_workers,
        use_graph=use_graph,
        graph=graph
    )
    print("处理完成！")
    
    try:
        component_count, folders_in_components = analyze_graph_components(output_file)
        print(f"无向图被分为 {component_count} 个连通分量:")
        for i, folders in enumerate(folders_in_components, 1):
            print(f"\n第 {i} 个连通分量（包含 {len(folders)} 个文件夹）:")
            for folder in sorted(folders):
                print(f'"{folder}",')
    except Exception as e:
        print(f"分析过程中出错：{e}")
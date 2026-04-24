import os
import re
import cv2
import json
import networkx as nx
import numpy as np
import torch
import itertools
import multiprocessing
from typing import List, Set
from tqdm import tqdm
from pathlib import Path

# 导入 AdjacencyCNN
from train import AdjacencyCNN, TrackGridDataset

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 🔥 【全局唯一 gridsize】所有地方统一用这个
# ==========================================
GRID_SIZE = 48  # 只改这里！！！

BASE         = Path("nuero")
IMAGE_ROOT   = BASE / "images"
QUERIES_ROOT = BASE / "queries"

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def natural_key(filename):
    """按文件名中第一个数字排序（兼容字符串和 Path）"""
    stem = Path(filename).stem
    nums = re.findall(r'\d+', stem)
    return int(nums[0]) if nums else int(1e9)


def get_sorted_images(folder: Path) -> List[Path]:
    """返回文件夹内按自然序排列的图片路径列表"""
    imgs = [f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTS]
    return sorted(imgs, key=lambda p: natural_key(p))


def get_all_image_folders(root: Path) -> List[Path]:
    """
    递归获取 root 下所有含图片的子文件夹，
    按 POSIX 路径升序排列，确保顺序跨平台一致。
    """
    result = []
    for folder in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
        if folder.is_dir():
            if any(f.suffix.lower() in IMG_EXTS for f in folder.iterdir() if f.is_file()):
                result.append(folder)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CoTracker 相关（合并自 gen_queries.py）
# ══════════════════════════════════════════════════════════════════════════════

def load_cotracker(device: torch.device):
    print("Loading CoTracker3 (offline) ...")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    model.eval()
    print("CoTracker loaded.")
    return model


def load_video_tensor(img_paths: List[Path], device: torch.device) -> torch.Tensor:
    """
    加载一组图片为 CoTracker 所需格式：(1, T, 3, H, W)，float32，值域 [0, 255]。
    各帧尺寸不同时统一 resize 到第一帧分辨率。
    """
    frames = []
    ref_h, ref_w = None, None
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {p}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        if ref_h is None:
            ref_h, ref_w = h, w
        elif h != ref_h or w != ref_w:
            img_rgb = cv2.resize(img_rgb, (ref_w, ref_h))
        frames.append(torch.from_numpy(img_rgb).permute(2, 0, 1).float())  # (C,H,W)
    return torch.stack(frames, dim=0).unsqueeze(0).to(device)  # (1,T,C,H,W)


def _queries_dir_for_folder(folder: Path, image_root: Path, queries_root: Path) -> Path:
    """计算 folder 对应的 queries 输出目录"""
    try:
        rel = folder.relative_to(image_root)
    except ValueError:
        rel = Path(folder.name)
    return queries_root / rel


def folder_queries_exist(folder: Path, image_root: Path, queries_root: Path) -> bool:
    """
    判断某个文件夹的 queries 缓存是否**完整**存在：
    要求文件夹内每张图片都有对应的 .json 文件。
    """
    imgs = get_sorted_images(folder)
    if not imgs:
        return False
    out_dir = _queries_dir_for_folder(folder, image_root, queries_root)
    return all((out_dir / (img.stem + ".json")).exists() for img in imgs)


def _save_folder_queries(
    folder: Path,
    queries: torch.Tensor,          # (1, N, 3)  [t=0, x, y]
    model_cotracker,
    device: torch.device,
    image_root: Path,
    queries_root: Path,
    grid_size: int,
):
    """
    对单个文件夹用给定 queries 运行 CoTracker，
    将每帧的 (N,2) 保存为 <queries_root>/<rel>/<img_stem>.json。
    """
    imgs = get_sorted_images(folder)
    if not imgs:
        return

    out_dir = _queries_dir_for_folder(folder, image_root, queries_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    video = load_video_tensor(imgs, device)   # (1, T, 3, H, W)
    T_folder = video.shape[1]

    with torch.no_grad():
        pred_tracks, _ = model_cotracker(
            video,
            grid_size=grid_size,
            grid_query_frame=0,
            backward_tracking=False,
            queries=queries,          # (1, N, 3)  [t=0, x, y]
        )
    # pred_tracks: (1, T_folder, N, 2)
    assert pred_tracks.shape[1] == T_folder, (
        f"{folder}: pred_tracks T={pred_tracks.shape[1]} ≠ 图片数 {T_folder}"
    )

    for t, img_path in enumerate(imgs):
        pts = pred_tracks[0, t, :, :].cpu().numpy()   # (N, 2)
        out_file = out_dir / (img_path.stem + ".json")
        with open(out_file, "w") as f:
            json.dump(pts.tolist(), f)


def ensure_queries(
    all_folders: List[Path],
    device: torch.device,
    image_root: Path,
    queries_root: Path,
    grid_size: int,
):
    """
    检查所有文件夹的 queries 缓存，按需生成缺失部分。

    策略：
    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ 情况                │ 处理方式                                           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ 全部都是新文件夹    │ 所有文件夹的第一帧拼成视频跑 CoTracker，           │
    │                     │ T=i 的输出作为第 i 个文件夹的 queries              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ 有旧文件夹          │ 取第一个旧文件夹 + 所有新文件夹的第一帧拼视频，    │
    │                     │ 跑 CoTracker；T=0 是旧文件夹锚点（仅用于坐标对齐, │
    │                     │ 不重新保存），T=1..k 对应各新文件夹                │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ 没有新文件夹        │ 全部缓存已存在，直接跳过                           │
    └─────────────────────┴────────────────────────────────────────────────────┘
    """
    old_folders = [f for f in all_folders if folder_queries_exist(f, image_root, queries_root)]
    new_folders = [f for f in all_folders if not folder_queries_exist(f, image_root, queries_root)]

    print(f"\n[ensure_queries] 旧文件夹(queries已存在): {len(old_folders)}  "
          f"新文件夹(需生成): {len(new_folders)}")

    if not new_folders:
        print("[ensure_queries] 所有 queries 缓存均已存在，跳过生成。")
        return

    # ── 加载 CoTracker（仅在需要时才加载）──────────────────────────────────
    model_cotracker = load_cotracker(device)

    if not old_folders:
        # ── 情况 A：全部新文件夹，从零开始 ──────────────────────────────────
        print(f"[ensure_queries] 全量生成，共 {len(new_folders)} 个文件夹 ...")
        target_folders = new_folders
        anchor_folder  = None
    else:
        # ── 情况 B：有旧文件夹，取第一个作锚点 ──────────────────────────────
        anchor_folder = old_folders[0]
        target_folders = new_folders
        print(f"[ensure_queries] 增量生成：锚点={anchor_folder.name}，"
              f"新文件夹 {len(new_folders)} 个 ...")

    # ── Step 1：构建首帧视频 → 获得每个 target_folder 的 initial queries ──
    #   视频帧顺序：[anchor（若有）] + [new_0, new_1, ..., new_k]
    frame_list: List[Path] = []
    if anchor_folder is not None:
        anchor_imgs = get_sorted_images(anchor_folder)
        assert anchor_imgs, f"锚点文件夹无图片: {anchor_folder}"
        frame_list.append(anchor_imgs[0])

    for folder in target_folders:
        imgs = get_sorted_images(folder)
        assert imgs, f"新文件夹无图片: {folder}"
        frame_list.append(imgs[0])

    print(f"[ensure_queries] Step1 视频帧数: {len(frame_list)} "
          f"({'含锚点' if anchor_folder else '无锚点'})")
    video = load_video_tensor(frame_list, device)  # (1, T_all, 3, H, W)

    with torch.no_grad():
        pred_tracks, _ = model_cotracker(
            video,
            grid_size=grid_size,
            grid_query_frame=0,
            backward_tracking=False,
        )
    # pred_tracks: (1, T_all, N, 2)
    print(f"[ensure_queries] Step1 pred_tracks: {pred_tracks.shape}")

    T_all = pred_tracks.shape[1]
    N     = pred_tracks.shape[2]

    # 计算 target_folders 对应的时间步偏移
    offset = 1 if anchor_folder is not None else 0
    assert T_all == offset + len(target_folders), (
        f"T_all={T_all} ≠ offset({offset}) + new_folders({len(target_folders)})"
    )

    # 为每个 target_folder 构造 queries (1, N, 3) [t=0, x, y]
    folder_queries = {}
    for i, folder in enumerate(target_folders):
        pts   = pred_tracks[0, offset + i, :, :]          # (N, 2)
        t_col = torch.zeros(N, 1, dtype=pts.dtype, device=pts.device)
        q     = torch.cat([t_col, pts], dim=1).unsqueeze(0)  # (1, N, 3)
        folder_queries[folder] = q

    # ── Step 2：逐文件夹运行 CoTracker → 保存 JSON ──────────────────────────
    print(f"[ensure_queries] Step2 逐文件夹追踪并保存 JSON ...")
    for folder in tqdm(target_folders, desc="生成 queries"):
        _save_folder_queries(
            folder=folder,
            queries=folder_queries[folder],
            model_cotracker=model_cotracker,
            device=device,
            image_root=image_root,
            queries_root=queries_root,
            grid_size=grid_size,
        )

    print(f"[ensure_queries] 完成，新增 {len(new_folders)} 个文件夹的 queries。")


# ══════════════════════════════════════════════════════════════════════════════
# queries 缓存读取 & 位移计算
# ══════════════════════════════════════════════════════════════════════════════

def load_queries_for_img(img_path: str,
                         queries_root: Path = QUERIES_ROOT,
                         image_root: Path = IMAGE_ROOT) -> np.ndarray:
    """从缓存读取单张图片对应的 (N,2) 点坐标；不存在则返回 None。"""
    img_path = Path(img_path)
    try:
        rel = img_path.parent.relative_to(image_root)
    except ValueError:
        rel = Path(img_path.parent.name)
    json_path = queries_root / rel / (img_path.stem + ".json")
    if not json_path.exists():
        print(f"[WARN] queries 缓存不存在: {json_path}")
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(data, dtype=np.float32)  # (N, 2)


def load_displacement_from_cache(img1_path: str, img2_path: str,
                                  queries_root: Path = QUERIES_ROOT,
                                  image_root: Path = IMAGE_ROOT):
    """
    从缓存读取两张图的 (N,2)，相减得位移。
    返回 (points_t1, displacements)，缺失时返回 (None, None)。
    """
    pts1 = load_queries_for_img(img1_path, queries_root, image_root)
    pts2 = load_queries_for_img(img2_path, queries_root, image_root)
    if pts1 is None or pts2 is None:
        return None, None
    if pts1.shape != pts2.shape:
        print(f"[WARN] 点数不一致: {img1_path}({pts1.shape}) vs {img2_path}({pts2.shape})")
        return None, None
    return pts1, pts2 - pts1


# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数：轨迹转 Grid
# ══════════════════════════════════════════════════════════════════════════════

def rasterize_to_grid(points, displacements, grid_size=GRID_SIZE):
    """
    将点和位移转换为 (2, grid_size, grid_size) 的 grid 表示。
    points / displacements: (N, 2) numpy 或 tensor
    """
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()
    if isinstance(displacements, np.ndarray):
        displacements = torch.from_numpy(displacements).float()

    max_x = max(points[:, 0].max().item(), 1.0)
    max_y = max(points[:, 1].max().item(), 1.0)
    scale_x = grid_size / max_x
    scale_y = grid_size / max_y

    grid = torch.zeros((2, grid_size, grid_size), dtype=torch.float32)
    xs = (points[:, 0] * scale_x).long().clamp(0, grid_size - 1)
    ys = (points[:, 1] * scale_y).long().clamp(0, grid_size - 1)

    flat_grid = grid.view(2, -1)
    indices = ys * grid_size + xs
    flat_grid[0].scatter_(0, indices, displacements[:, 0] * scale_x)
    flat_grid[1].scatter_(0, indices, displacements[:, 1] * scale_y)
    return grid


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_adjacency_model(model_path, device, grid_size=GRID_SIZE):
    """加载 AdjacencyCNN"""
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict      = checkpoint['model_state_dict']
        loaded_grid_size = checkpoint.get('grid_size', grid_size)
    else:
        state_dict      = checkpoint
        loaded_grid_size = grid_size
    model = AdjacencyCNN(grid_size=loaded_grid_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded adjacency model from {model_path} (grid_size={loaded_grid_size})")
    return model, loaded_grid_size


# ══════════════════════════════════════════════════════════════════════════════
# 同文件夹边
# ══════════════════════════════════════════════════════════════════════════════

def process_same_folder_edges(frame_dict, folders_to_process, same_folder_threshold=1):
    edges = set()
    for folder in tqdm(folders_to_process, desc="处理同文件夹边"):
        paths = frame_dict.get(folder, [])
        if not paths:
            continue
        folder_path = Path(folder)
        sorted_paths = [(folder_path / os.path.basename(p)).as_posix()
                        for p in sorted(paths, key=natural_key)]
        for i in range(len(sorted_paths)):
            for j in range(max(0, i - same_folder_threshold),
                           min(len(sorted_paths), i + same_folder_threshold + 1)):
                if i != j:
                    edges.add(tuple(sorted((sorted_paths[i], sorted_paths[j]))))
    return edges


# ══════════════════════════════════════════════════════════════════════════════
# 跨文件夹边（从 queries 缓存读取位移，无需 CoTracker）
# ══════════════════════════════════════════════════════════════════════════════

def process_cross_folder_pairs(args):
    """
    args: (paths1, paths2, model_adj, grid_size, device, queries_root, image_root)
    """
    (paths1, paths2, model_adj, grid_size, device, queries_root, image_root) = args

    edges       = set()
    batch_grids = []
    batch_pairs = []
    batch_size  = 4

    total = len(paths1) * len(paths2)
    idx_global = 0
    for i, path1 in enumerate(paths1):
        for j, path2 in enumerate(paths2):
            idx_global += 1
            points_t1, displacements = load_displacement_from_cache(
                path1, path2, queries_root=queries_root, image_root=image_root
            )
            if points_t1 is not None:
                batch_grids.append(rasterize_to_grid(points_t1, displacements, grid_size))
                batch_pairs.append((path1, path2))

            # 当批次满 或 是最后一对 时推理
            last = (idx_global == total)
            if (len(batch_grids) >= batch_size or last) and batch_grids:
                batch_tensor = torch.stack(batch_grids).to(device)
                with torch.no_grad():
                    probs = model_adj(batch_tensor)          # (B, 1)
                    preds = (probs > 0.5).cpu().numpy()
                for k, (p1, p2) in enumerate(batch_pairs):
                    if preds[k]:
                        edges.add(tuple(sorted((p1, p2))))
                del batch_tensor, probs, preds
                torch.cuda.empty_cache()
                batch_grids = []
                batch_pairs = []

    return edges


# ══════════════════════════════════════════════════════════════════════════════
# 图分析
# ══════════════════════════════════════════════════════════════════════════════

def analyze_graph_components(graph_file):
    """分析无向图的连通分量及每个分量包含的文件夹"""
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"图文件 {graph_file} 不存在")
    G = nx.Graph()
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
    node_to_folder = {node: Path(node).parent.as_posix() for node in G.nodes()}
    folder_components = []
    for component in nx.connected_components(G):
        folder_components.append({node_to_folder[n] for n in component})
    return len(folder_components), folder_components


def read_and_print_graph(graph_file):
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"图文件 {graph_file} 不存在")
    graph = nx.Graph()
    with open(graph_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                u, v = line.split()
                graph.add_edge(u, v)
            except ValueError:
                continue
    folders = set()
    for u, v in graph.edges():
        folders.add(Path(u).parent.as_posix())
        folders.add(Path(v).parent.as_posix())
    return graph.edges(), folders


def generate_output_filename(model_path, input_folder):
    base_name    = os.path.splitext(os.path.basename(model_path))[0]
    first_folder = os.path.normpath(input_folder).split('/')[0]
    return f"{base_name}-{first_folder}.txt"


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

def build_video_frame_graph(
        folders: List[str],
        model_path: str,
        device: str = "cuda",
        output_file: str = "graph.txt",
        same_folder_threshold: int = 1,
        num_workers: int = 1,
        use_graph: bool = False,
        graph: str = None,
        queries_root: Path = QUERIES_ROOT,
        image_root: Path = IMAGE_ROOT,
):
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # ── 1. 加载 AdjacencyCNN ─────────────────────────────────────────────────
    model_adj, grid_size = load_adjacency_model(model_path, device_obj, grid_size=GRID_SIZE)

    # ── 2. 处理旧图逻辑 ───────────────────────────────────────────────────────
    new_folders_graph = None
    old_edges         = set()
    old_folders_graph = set()

    if use_graph and graph is not None:
        loaded_edges, loaded_folders = read_and_print_graph(graph)
        old_edges         = set(tuple(sorted(e)) for e in loaded_edges)
        old_folders_graph = {Path(f).as_posix() for f in loaded_folders if Path(f).is_dir()}
        input_folders     = {Path(f).as_posix() for f in folders if Path(f).is_dir()}
        new_folders_graph = input_folders - old_folders_graph
        folders           = list(input_folders)
        print(f"复用旧图：{len(old_folders_graph)} 个旧文件夹，"
              f"{len(new_folders_graph)} 个新文件夹")

        if not new_folders_graph and same_folder_threshold == 1:
            print("没有新文件夹需要处理")
            with open(output_file, 'w') as f:
                for u, v in old_edges:
                    f.write(f"{u} {v}\n")
            return old_edges

    # ── 3. 构建帧字典 ─────────────────────────────────────────────────────────
    frame_dict = {}
    scan_set = (new_folders_graph
                if (use_graph and graph is not None and new_folders_graph)
                else folders)

    if use_graph and graph is not None:
        for folder in tqdm(old_folders_graph, desc="扫描旧文件夹"):
            fp = Path(folder)
            frame_dict[folder] = [
                p.as_posix() for p in fp.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            ]

    for folder in tqdm(scan_set, desc="扫描新文件夹"):
        fp = Path(folder)
        if fp.is_dir():
            frame_dict[folder] = [
                p.as_posix() for p in fp.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            ]

    # ── 4. 确保所有需要处理的文件夹都有 queries 缓存 ─────────────────────────
    #   all_folders_path: 全部涉及的文件夹（Path 对象，用于 ensure_queries）
    all_involved = (list(new_folders_graph) if (use_graph and graph is not None and new_folders_graph)
                    else list(frame_dict.keys()))
    all_involved_paths = [Path(f) for f in all_involved if Path(f).is_dir()]

    queries_root.mkdir(parents=True, exist_ok=True)
    ensure_queries(
        all_folders=all_involved_paths,
        device=device_obj,
        image_root=image_root,
        queries_root=queries_root,
        grid_size=grid_size,
    )

    # ── 5. 同文件夹边 ─────────────────────────────────────────────────────────
    edges = old_edges.copy()
    folders_to_process = (new_folders_graph
                          if (use_graph and graph is not None and new_folders_graph)
                          else frame_dict.keys())
    edges.update(process_same_folder_edges(frame_dict, folders_to_process,
                                           same_folder_threshold))

    if use_graph and graph is not None and not new_folders_graph:
        with open(output_file, 'w') as f:
            for u, v in edges:
                f.write(f"{u} {v}\n")
        print(f"生成完成，共 {len(edges)} 条边，保存至 {output_file}")
        return edges

    # ── 6. 跨文件夹边 ─────────────────────────────────────────────────────────
    if use_graph and graph is not None and new_folders_graph:
        new_list = list(new_folders_graph)
        folder_pairs = list(itertools.combinations(new_list, 2))
        for nf in new_list:
            for of in old_folders_graph:
                if nf in frame_dict and of in frame_dict:
                    folder_pairs.append((nf, of))
    else:
        folder_pairs = list(itertools.combinations(list(frame_dict.keys()), 2))

    tasks = [
        (frame_dict[f1], frame_dict[f2],
         model_adj, grid_size, device_obj,
         queries_root, image_root)
        for f1, f2 in folder_pairs
        if frame_dict.get(f1) and frame_dict.get(f2)
    ]

    if tasks:
        if num_workers > 1:
            try:
                multiprocessing.set_start_method("spawn", force=True)
                with multiprocessing.Pool(num_workers) as pool:
                    for results in tqdm(pool.imap_unordered(process_cross_folder_pairs, tasks),
                                        total=len(tasks), desc="跨文件夹预测"):
                        for r in results:
                            edges.add(tuple(sorted(r)))
            except Exception as e:
                print(f"多进程出错，回退到单进程：{e}")
                for task in tqdm(tasks, desc="跨文件夹预测 (单进程)"):
                    for r in process_cross_folder_pairs(task):
                        edges.add(tuple(sorted(r)))
        else:
            for task in tqdm(tasks, desc="跨文件夹预测 (单进程)"):
                for r in process_cross_folder_pairs(task):
                    edges.add(tuple(sorted(r)))

    unique_edges = set(tuple(sorted(e)) for e in edges)
    with open(output_file, 'w') as f:
        for u, v in unique_edges:
            f.write(f"{u} {v}\n")
    print(f"生成完成，共 {len(unique_edges)} 条边，保存至 {output_file}")
    return unique_edges


# ══════════════════════════════════════════════════════════════════════════════
# 主程序入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    base         = Path("nuero/f2l")
    model_path   = base / "best_model.pth"
    input_folder = base / "images"
    output_file  = base / "graph.txt"
    graph_file   = base / "graph.txt"   # 旧图路径（use_graph=True 时使用）
    device       = "cuda"
    use_graph    = False
    num_workers  = 1

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

    folders = get_subdirectories([input_folder])
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
        graph=graph_file,
        queries_root=base / "queries",
        image_root=base / "images",
    )
    print("处理完成！")

    try:
        component_count, folders_in_components = analyze_graph_components(output_file)
        print(f"无向图被分为 {component_count} 个连通分量:")
        for i, fset in enumerate(folders_in_components, 1):
            print(f"\n第 {i} 个连通分量（包含 {len(fset)} 个文件夹）:")
            for folder in sorted(fset):
                print(f'  "{folder}"')
    except Exception as e:
        print(f"分析过程中出错：{e}")

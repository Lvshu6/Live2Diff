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
from rife import load_rife_model, interpolate_images

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = ""

# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def natural_key(filename):
    stem = Path(filename).stem
    nums = re.findall(r'\d+', stem)
    return int(nums[0]) if nums else int(1e9)


def get_sorted_images(folder: Path, img_exts: set) -> List[Path]:
    imgs = [f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in img_exts]
    return sorted(imgs, key=lambda p: natural_key(p))


def get_all_image_folders(root: Path, img_exts: set) -> List[Path]:
    result = []
    for folder in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
        if folder.is_dir():
            if any(f.suffix.lower() in img_exts for f in folder.iterdir() if f.is_file()):
                result.append(folder)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CoTracker 相关
# ══════════════════════════════════════════════════════════════════════════════

def load_cotracker(device: torch.device):
    print("Loading CoTracker3 (offline) ...")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    model.eval()
    print("CoTracker loaded.")
    return model


def load_video_tensor(img_paths: List[Path], device: torch.device) -> torch.Tensor:
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
        frames.append(torch.from_numpy(img_rgb).permute(2, 0, 1).float())
    return torch.stack(frames, dim=0).unsqueeze(0).to(device)


def _queries_dir_for_folder(folder: Path, image_root: Path, queries_root: Path) -> Path:
    try:
        rel = folder.relative_to(image_root)
    except ValueError:
        rel = Path(folder.name)
    return queries_root / rel


def folder_queries_exist(folder: Path, image_root: Path, queries_root: Path, img_exts: set) -> bool:
    imgs = get_sorted_images(folder, img_exts)
    if not imgs:
        return False
    out_dir = _queries_dir_for_folder(folder, image_root, queries_root)
    return all((out_dir / (img.stem + ".json")).exists() for img in imgs)


# ------------------------------
# 完整处理一个文件夹：保存所有帧的 queries
# ------------------------------

def process_full_folder(
    folder: Path,
    anchor_frame: Path,
    cotracker,
    rife_model,
    device,
    image_root: Path,
    queries_root: Path,
    grid_size: int,
    img_exts: set,
    rife_exp=2,
    rife_cache="./cache"
):
    imgs = get_sorted_images(folder, img_exts)
    out_dir = _queries_dir_for_folder(folder, image_root, queries_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 最简插帧逻辑（你原来的，保留） ---
    interp_paths = interpolate_images(str(anchor_frame), str(imgs[0]), rife_model, exp=rife_exp, cache_root=rife_cache)
    mid_frames = [Path(p) for p in interp_paths[1:-1]]
    video_paths = [anchor_frame] + mid_frames + imgs
    video = load_video_tensor(video_paths, device)

    # ======================
    # 🔴 【你要求的核心修复】
    # ======================
    queries_input = None  # 默认 None（第一个文件夹用）

    # 👇 只有 anchor_frame 不是本文件夹第一帧时，才读取保存的 queries
    anchor_folder = anchor_frame.parent
    if anchor_folder != folder:
        # 读取 anchor_frame 对应的轨迹文件
        anchor_query_path = _queries_dir_for_folder(anchor_folder, image_root, queries_root) / f"{anchor_frame.stem}.json"
        if anchor_query_path.exists():
            with open(anchor_query_path, "r") as f:
                points = json.load(f)  # shape [N, 2] (x,y)
            
            # 构造 CoTracker 需要的格式：(B, N, 3) → (t=0, x, y)
            t = torch.zeros(len(points), dtype=torch.float32, device=device)
            xy = torch.tensor(points, dtype=torch.float32, device=device)
            queries_input = torch.cat([t.unsqueeze(1), xy], dim=1).unsqueeze(0)  # [1, N, 3]

    # ======================
    # 🔴 传入 queries！
    # ======================
    with torch.no_grad():
        tracks, _ = cotracker(
            video,
            grid_size=grid_size,
            grid_query_frame=0,
            backward_tracking=False,
            queries=queries_input  # ✅ 第一个文件夹=None，后续=已保存点
        )

    # 你原来的保存逻辑（不变）
    offset = 1 + len(mid_frames)
    for i, img in enumerate(imgs):
        pts = tracks[0, i + offset].cpu().numpy()
        with open(out_dir / f"{img.stem}.json", "w") as f:
            json.dump(pts.tolist(), f)

    del video, tracks
    torch.cuda.empty_cache()
    return True


# ------------------------------
# 计算两帧之间的跟踪 loss
# ------------------------------
def compute_pair_loss(
    imgA: Path,
    imgB: Path,
    cotracker,
    device,
    grid_size,
):

    interp = [str(imgA), str(imgB)]

    video = load_video_tensor([Path(p) for p in interp], device)
    with torch.no_grad():
        tracks, _ = cotracker(video, grid_size=grid_size, grid_query_frame=0)

    t0 = tracks[0, 0].cpu().numpy()
    t1 = tracks[0, -1].cpu().numpy()
    loss = np.mean(np.abs(t1 - t0))

    del video, tracks
    torch.cuda.empty_cache()
    return loss


# ------------------------------
# 【你要的核心逻辑】渐进式最小误差构建 queries
# ------------------------------
# ------------------------------
def ensure_queries_progressive(
    all_folders: List[Path],
    device: torch.device,
    image_root: Path,
    queries_root: Path,
    grid_size: int,
    img_exts: set,
    rife_exp: int = 2,
    rife_cache: str = "./cache",
    sample_step: int = 3  # 步长3采样
):
    cotracker = load_cotracker(device)
    rife_model = load_rife_model('train_log')

    # ===================== 【新加】自动识别已完成文件夹 =====================
    completed = []
    remaining = []

    for folder in all_folders:
        # 构建该文件夹对应的 queries 输出目录
        rel_path = folder.relative_to(image_root)
        query_folder = queries_root / rel_path

        # 判断：是否已经存在跟踪结果（任意一个json存在就算完成）
        has_json = list(query_folder.rglob("*.json"))
        if query_folder.exists() and len(has_json) > 0:
            completed.append(folder)
            print(f"✅ 已存在（跳过）: {folder.name}")
        else:
            remaining.append(folder)
    # ======================================================================

    # 如果没有任何已完成，必须先处理第一个
    if not completed and remaining:
        first = remaining.pop(0)
        first_imgs = get_sorted_images(first, img_exts)
        process_full_folder(
            folder=first,
            anchor_frame=first_imgs[0],
            cotracker=cotracker,
            rife_model=rife_model,
            device=device,
            image_root=image_root,
            queries_root=queries_root,
            grid_size=grid_size,
            img_exts=img_exts,
            rife_exp=rife_exp,
            rife_cache=rife_cache
        )
        completed.append(first)
        print(f"✅ 初始文件夹完成: {first.name}")

    # 循环：每次选全局最小 loss 的文件夹处理
    while remaining:
        print(f"\n【迭代】剩余文件夹: {len(remaining)}")
        best_loss = float('inf')
        best_folder = None
        best_anchor_frame = None

        # 遍历所有未处理文件夹
        for folder in remaining:
            folder_imgs = get_sorted_images(folder, img_exts)
            if not folder_imgs:
                continue
            target_first = folder_imgs[0]

            # 遍历已完成的所有文件夹
            for comp in completed:
                comp_imgs = get_sorted_images(comp, img_exts)
                # 步长 3 采样
                sampled = comp_imgs[::sample_step]
                for anchor_candidate in sampled:
                    loss = compute_pair_loss(
                        imgA=anchor_candidate,
                        imgB=target_first,
                        cotracker=cotracker,
                        device=device,
                        grid_size=grid_size,
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best_folder = folder
                        best_anchor_frame = anchor_candidate

        # 处理最优文件夹
        print(f"🎯 选定最小loss文件夹: {best_folder.name}, loss={best_loss:.4f}")
        process_full_folder(
            folder=best_folder,
            anchor_frame=best_anchor_frame,
            cotracker=cotracker,
            rife_model=rife_model,
            device=device,
            image_root=image_root,
            queries_root=queries_root,
            grid_size=grid_size,
            img_exts=img_exts,
            rife_exp=rife_exp,
            rife_cache=rife_cache
        )

        # 移动到已完成
        remaining.remove(best_folder)
        completed.append(best_folder)

    print("\n🎉 所有文件夹已按【最小误差策略】构建完成！")


# ══════════════════════════════════════════════════════════════════════════════
# 以下函数完全不变，保持兼容
# ══════════════════════════════════════════════════════════════════════════════

def load_queries_for_img(img_path: str, queries_root: Path, image_root: Path) -> np.ndarray:
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
    return np.array(data, dtype=np.float32)


def load_displacement_from_cache(img1_path: str, img2_path: str, queries_root: Path, image_root: Path):
    pts1 = load_queries_for_img(img1_path, queries_root, image_root)
    pts2 = load_queries_for_img(img2_path, queries_root, image_root)
    if pts1 is None or pts2 is None:
        return None, None
    if pts1.shape != pts2.shape:
        print(f"[WARN] 点数不一致: {img1_path}({pts1.shape}) vs {img2_path}({pts2.shape})")
        return None, None
    return pts1, pts2 - pts1


def rasterize_to_grid(points, displacements, grid_size: int):
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


def load_adjacency_model(model_path, device, grid_size: int):
    checkpoint = torch.load(model_path, map_location=device)
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


def process_cross_folder_pairs(args):
    paths1, paths2, model_adj, grid_size, device, queries_root, image_root = args
    edges = set()
    batch_grids = []
    batch_pairs = []
    batch_size = 4
    total = len(paths1) * len(paths2)
    idx_global = 0

    for path1 in paths1:
        for path2 in paths2:
            idx_global += 1
            points_t1, displacements = load_displacement_from_cache(path1, path2, queries_root, image_root)
            if points_t1 is not None:
                batch_grids.append(rasterize_to_grid(points_t1, displacements, grid_size))
                batch_pairs.append((path1, path2))

            last = (idx_global == total)
            if (len(batch_grids) >= batch_size or last) and batch_grids:
                batch_tensor = torch.stack(batch_grids).to(device)
                with torch.no_grad():
                    probs = model_adj(batch_tensor)
                    preds = (probs > 0.5).cpu().numpy()
                for k, (p1, p2) in enumerate(batch_pairs):
                    if preds[k]:
                        edges.add(tuple(sorted((p1, p2))))
                del batch_tensor, probs, preds
                torch.cuda.empty_cache()
                batch_grids = []
                batch_pairs = []
    return edges


def analyze_graph_components(graph_file):
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


# ══════════════════════════════════════════════════════════════════════════════
# 主函数（已替换新版）
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
        queries_root: Path = None,
        image_root: Path = None,
        grid_size: int = 48,
        img_exts: set = None,
        rife_exp: int = 2,
        rife_cache: str = "./cache"
):
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    img_exts = img_exts or {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    model_adj, grid_size = load_adjacency_model(model_path, device_obj, grid_size=grid_size)

    all_folder_paths = [Path(f) for f in folders if Path(f).is_dir()]
    ensure_queries_progressive(
        all_folders=all_folder_paths,
        device=device_obj,
        image_root=image_root,
        queries_root=queries_root,
        grid_size=grid_size,
        img_exts=img_exts,
        rife_exp=rife_exp,
        rife_cache=rife_cache,
        sample_step=3
    )

    frame_dict = {}
    for folder in tqdm(folders, desc="扫描文件夹"):
        fp = Path(folder)
        if fp.is_dir():
            frame_dict[folder] = [p.as_posix() for p in fp.iterdir()
                                   if p.is_file() and p.suffix.lower() in img_exts]

    edges = set()
    edges.update(process_same_folder_edges(frame_dict, frame_dict.keys(), same_folder_threshold))
    folder_pairs = list(itertools.combinations(list(frame_dict.keys()), 2))
    tasks = [(frame_dict[f1], frame_dict[f2], model_adj, grid_size, device_obj, queries_root, image_root)
             for f1, f2 in folder_pairs if frame_dict.get(f1) and frame_dict.get(f2)]

    if tasks:
        if num_workers > 1:
            multiprocessing.set_start_method("spawn", force=True)
            with multiprocessing.Pool(num_workers) as pool:
                for res in tqdm(pool.imap_unordered(process_cross_folder_pairs, tasks), total=len(tasks)):
                    edges.update(res)
        else:
            for task in tqdm(tasks, desc="跨文件夹预测"):
                edges.update(process_cross_folder_pairs(task))

    unique_edges = set(tuple(sorted(e)) for e in edges)
    with open(output_file, 'w') as f:
        for u, v in unique_edges:
            f.write(f"{u} {v}\n")
    print(f"生成完成，共 {len(unique_edges)} 条边")
    return unique_edges


# ══════════════════════════════════════════════════════════════════════════════
# 主程序入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    GRID_SIZE    = 48
    RIFE_EXP     = 3
    BASE         = Path("nuero")
    RIFE_CACHE   = BASE / "cache"
    IMAGE_ROOT   = BASE / "images"
    QUERIES_ROOT = BASE / "queries"
    IMG_EXTS     = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    model_path   = BASE / "best_model.pth"
    input_folder = BASE / "images"
    output_file  = BASE / "graph.txt"
    device       = "cuda"
    num_workers  = 1

    def get_subdirectories(folders):
        subdirs = []
        for folder in folders:
            if os.path.isdir(folder):
                for item in os.listdir(folder):
                    p = os.path.join(folder, item)
                    if os.path.isdir(p):
                        subdirs.append(p)
                        subdirs.extend(get_subdirectories([p]))
        return subdirs

    folders = get_subdirectories([input_folder])
    print(f"待处理文件夹数量：{len(folders)}")

    build_video_frame_graph(
        folders=folders,
        model_path=model_path,
        device=device,
        output_file=output_file,
        queries_root=QUERIES_ROOT,
        image_root=IMAGE_ROOT,
        grid_size=GRID_SIZE,
        img_exts=IMG_EXTS,
        rife_exp=RIFE_EXP,
        rife_cache=RIFE_CACHE,
        num_workers=num_workers
    )

    try:
        cnt, comps = analyze_graph_components(output_file)
        print(f"\n连通分量：{cnt}")
        for i, c in enumerate(comps, 1):
            print(f"{i}: {sorted(c)}")
    except:
        pass